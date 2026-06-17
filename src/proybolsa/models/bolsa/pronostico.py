"""Pronosticador multi-horizonte de precio de bolsa.

Interfaz principal del modulo. Une el modelo de nivel diario con el perfil horario.

Uso tipico:
    from proybolsa.models.bolsa.pronostico import PronosticadorBolsa

    p = PronosticadorBolsa()
    p.fit(df_diario_train, df_perfil)
    resultado = p.pronosticar(horizonte_dias=30, df_futuro=df_drivers_futuros)
    # resultado tiene columnas: fecha, pred_diaria, hora, pred_horaria, ci_lo90, ci_hi90
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from proybolsa.features.calendar import agregar_features_fecha
from proybolsa.features.hourly_profile import aplicar_perfil_horario, cargar_perfil
from proybolsa.models.bolsa.nivel_diario import EnsembleNivel, LGBNivel, SARIMAXNivel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Escenarios hidrologicos
# ---------------------------------------------------------------------------

ESCENARIOS_HIDRO = {
    "seco":     {"aportes_pct": 65.0,  "volumen_util_pct": 40.0},
    "promedio": {"aportes_pct": 100.0, "volumen_util_pct": 60.0},
    "humedo":   {"aportes_pct": 140.0, "volumen_util_pct": 75.0},
}


def _construir_df_futuro(
    horizonte_dias: int,
    fecha_inicio: pd.Timestamp,
    escenario: str = "promedio",
    precio_escasez: float = 906.0,
    oni_asumido: float = 0.0,
    df_referencia: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construye un DataFrame de drivers futuros para el pronostico.

    Si df_referencia se provee, toma la ultima fila disponible para propagar
    los drivers observados mas recientes. Si no, usa los valores del escenario.
    """
    fechas = pd.date_range(fecha_inicio, periods=horizonte_dias, freq="D")
    df = pd.DataFrame({"fecha": fechas.date})
    df = agregar_features_fecha(df)

    # Hidrologia por escenario
    hidro = ESCENARIOS_HIDRO.get(escenario, ESCENARIOS_HIDRO["promedio"])
    df["aportes_pct"] = hidro["aportes_pct"]
    df["volumen_util_pct"] = hidro["volumen_util_pct"]
    df["precio_escasez"] = precio_escasez

    # ENSO
    df["oni_lag"] = oni_asumido
    df["enso_el_nino"] = int(oni_asumido >= 0.5)
    df["enso_la_nina"] = int(oni_asumido <= -0.5)

    # Trig mes
    df["cos_mes"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["sin_mes"] = np.sin(2 * np.pi * df["mes"] / 12)

    # Propagar el ultimo precio observado como lag inicial
    if df_referencia is not None and len(df_referencia) > 0:
        ultimo_precio = df_referencia["precio_bolsa_mean"].iloc[-1]
        df["precio_bolsa_mean_lag1d"] = ultimo_precio
        df["precio_bolsa_mean_lag7d"] = df_referencia["precio_bolsa_mean"].iloc[-7:].mean() if len(df_referencia) >= 7 else ultimo_precio
        df["precio_bolsa_mean_lag30d"] = df_referencia["precio_bolsa_mean"].iloc[-30:].mean() if len(df_referencia) >= 30 else ultimo_precio
        df["precio_bolsa_mean_roll7d_std"] = df_referencia["precio_bolsa_mean"].iloc[-7:].std() if len(df_referencia) >= 7 else 0.0
        df["precio_bolsa_mean_roll30d_std"] = df_referencia["precio_bolsa_mean"].iloc[-30:].std() if len(df_referencia) >= 30 else 0.0
        # Lags hidrologicos (ultimos observados)
        for col in ("aportes_pct_lag1d", "aportes_pct_lag7d", "volumen_util_pct_lag1d", "volumen_util_pct_lag7d"):
            if col in df_referencia.columns:
                df[col] = df_referencia[col].iloc[-1]

    return df


# ---------------------------------------------------------------------------
# Pronosticador principal
# ---------------------------------------------------------------------------

@dataclass
class PronosticadorBolsa:
    """Pronosticador multi-horizonte del precio de bolsa de energia.

    Flujo:
      1. fit(df_diario, perfil) → ajusta EnsembleNivel
      2. pronosticar(horizonte_dias, ...) → serie horaria con bandas

    Horizontes reconocidos:
      corto   1-7 dias      (usa lags del precio; precision maxima)
      tactico 8-90 dias     (usa hidrologia + ENSO; escenarios disponibles)
      largo   91-720 dias   (usa estructura estacional; escenarios obligatorios)
    """

    modelo: EnsembleNivel = field(default_factory=EnsembleNivel)
    perfil: pd.DataFrame | None = field(default=None, repr=False)
    _df_train: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        df_diario: pd.DataFrame,
        perfil: pd.DataFrame | None = None,
        ruta_perfil: str = "data/processed/perfil_horario.parquet",
        val_fraccion: float = 0.1,
    ) -> "PronosticadorBolsa":
        """Ajusta el ensemble con los datos historicos diarios.

        Parametros
        ----------
        df_diario    : Feature matrix diaria (salida de construir_features.py)
        perfil       : DataFrame del perfil horario (si None, carga desde ruta_perfil)
        val_fraccion : Fraccion del fin del historico usada para calibrar pesos
        """
        # Perfil horario
        if perfil is not None:
            self.perfil = perfil
        else:
            p = Path(ruta_perfil)
            if p.exists():
                self.perfil = cargar_perfil(str(p))
            else:
                logger.warning("Perfil horario no encontrado en %s. Pronostico horario no disponible.", ruta_perfil)

        # Ordenar por fecha
        df = df_diario.sort_values("fecha").reset_index(drop=True)
        df = df.dropna(subset=["precio_bolsa_mean"])
        self._df_train = df

        # Split train / val para calibrar pesos
        n_val = max(1, int(len(df) * val_fraccion))
        df_train = df.iloc[:-n_val]
        df_val = df.iloc[-n_val:]

        logger.info(
            "Ajustando ensemble: %d dias train, %d dias val (%.0f%%)",
            len(df_train), len(df_val), val_fraccion * 100,
        )
        self.modelo.fit(df_train, df_val=df_val)
        return self

    def pronosticar(
        self,
        horizonte_dias: int,
        fecha_inicio: pd.Timestamp | None = None,
        escenario: str = "promedio",
        precio_escasez: float | None = None,
        oni_asumido: float = 0.0,
        escenarios_multiples: bool = False,
        devolver_horario: bool = True,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Genera el pronostico para los proximos `horizonte_dias` dias.

        Parametros
        ----------
        horizonte_dias      : Numero de dias a pronosticar
        fecha_inicio        : Primer dia del pronostico (default: dia siguiente al ultimo en train)
        escenario           : 'seco', 'promedio' o 'humedo' (hidrologia asumida)
        precio_escasez      : Precio de escasez asumido (default: ultimo observado)
        oni_asumido         : Valor de ONI asumido para el horizonte
        escenarios_multiples: Si True, calcula los 3 escenarios y devuelve un dict
        devolver_horario    : Si True (default), expande a 24h usando perfil

        Returns
        -------
        DataFrame con columnas: fecha, pred_diaria, ci_lo90, ci_hi90
        Si devolver_horario=True: agrega timestamp (horario), pred_horaria
        Si escenarios_multiples=True: devuelve dict{'seco':..., 'promedio':..., 'humedo':...}
        """
        if escenarios_multiples:
            return {
                esc: self.pronosticar(
                    horizonte_dias, fecha_inicio, escenario=esc,
                    precio_escasez=precio_escasez, oni_asumido=oni_asumido,
                    escenarios_multiples=False, devolver_horario=devolver_horario,
                )
                for esc in ("seco", "promedio", "humedo")
            }

        # Fecha de inicio
        if fecha_inicio is None and self._df_train is not None:
            ultima_fecha = pd.to_datetime(self._df_train["fecha"].max())
            fecha_inicio = ultima_fecha + pd.Timedelta(days=1)
        elif fecha_inicio is None:
            raise ValueError("Debe especificar fecha_inicio si el modelo no fue ajustado.")

        # Precio de escasez
        if precio_escasez is None and self._df_train is not None and "precio_escasez" in self._df_train.columns:
            precio_escasez = self._df_train["precio_escasez"].dropna().iloc[-1]
        precio_escasez = precio_escasez or 906.0

        # Construir drivers futuros
        df_futuro = _construir_df_futuro(
            horizonte_dias=horizonte_dias,
            fecha_inicio=fecha_inicio,
            escenario=escenario,
            precio_escasez=precio_escasez,
            oni_asumido=oni_asumido,
            df_referencia=self._df_train,
        )

        # Pronostico del nivel diario
        pred_df = self.modelo.predict(
            df_futuro,
            horizon=horizonte_dias,
            exog_future=df_futuro,
        )
        pred_df["fecha"] = [r.date() for r in pd.date_range(fecha_inicio, periods=horizonte_dias)]
        pred_df = pred_df.rename(columns={"pred": "pred_diaria"})

        # Expandir a horario con perfil
        if devolver_horario and self.perfil is not None:
            pred_df = agregar_features_fecha(pred_df)   # agrega tipo_dia, mes para el perfil
            pred_horario = aplicar_perfil_horario(
                pred_df,
                self.perfil,
                col_nivel_diario="pred_diaria",
            )
            # Columna que devuelve aplicar_perfil_horario
            col_h = "precio_bolsa_pred_horario"

            # Agregar fecha (date) para el merge
            pred_horario["_fecha"] = pred_horario["timestamp"].dt.date

            # Merge bandas diarias al horario
            pred_horario = pred_horario.merge(
                pred_df[["fecha", "pred_diaria", "ci_lo90", "ci_hi90"]].rename(columns={"fecha": "_fecha"}),
                on="_fecha",
                how="left",
            ).drop(columns=["_fecha"])

            # Escalar CI proporcionalmente al perfil
            ratio = pred_horario[col_h] / pred_horario["pred_diaria"].replace(0, np.nan)
            pred_horario["ci_lo90_horario"] = pred_horario["ci_lo90"] * ratio
            pred_horario["ci_hi90_horario"] = pred_horario["ci_hi90"] * ratio
            pred_horario = pred_horario.rename(columns={col_h: "pred_horaria"})
            return pred_horario

        return pred_df

    def resumen_modelo(self) -> dict:
        """Devuelve un diccionario con informacion diagnostica del modelo ajustado."""
        info = {
            "w_sarimax": round(self.modelo.w_sarimax, 3),
            "w_lgb": round(self.modelo.w_lgb, 3),
            "n_train": len(self._df_train) if self._df_train is not None else None,
            "perfil_celdas": len(self.perfil) if self.perfil is not None else None,
        }
        if hasattr(self.modelo.sarimax, "_result") and self.modelo.sarimax._result is not None:
            info["aic_sarimax"] = round(self.modelo.sarimax._result.aic, 1)
        if self.modelo.lgb._model is not None:
            top = self.modelo.lgb.feature_importance.head(5)
            info["top_features_lgb"] = top.to_dict()
        return info
