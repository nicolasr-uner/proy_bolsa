"""Modelo de nivel diario del precio de bolsa.

Pipeline:
    log(precio_diario) = SARIMAX(exog) + LightGBM(features) -> ensemble -> exp()

La descomposicion es:
    precio_horario(dia, hora) = nivel_diario(dia) x perfil(hora | tipo_dia, mes)

Este modulo resuelve el primer factor: nivel_diario(dia).

Horizontes:
    corto   1-7 dias      (lags de precio son la señal dominante)
    tactico 8-90 dias     (hidrologia + ENSO dominan)
    largo   91-720 dias   (estructura estacional + reversión a media + escenarios)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

# Variables exogenas para el SARIMAX — solo las mas robustas y disponibles
_EXOG_SARIMAX = [
    "aportes_pct",
    "volumen_util_pct",
    "precio_escasez",
    "enso_el_nino",
    "enso_la_nina",
    "cos_mes",
    "sin_mes",
]

# Features para LightGBM — incluye todo lo disponible
_FEATS_LGB = [
    # Hidrologia
    "aportes_pct", "aportes_pct_lag1d", "aportes_pct_lag7d", "aportes_pct_lag30d",
    "volumen_util_pct", "volumen_util_pct_lag1d", "volumen_util_pct_lag7d", "volumen_util_pct_lag30d",
    "vertimientos_ratio",
    # Precio de escasez (techo structural)
    "precio_escasez",
    # ENSO
    "enso_el_nino", "enso_la_nina", "oni_lag",
    # Calendario
    "mes", "dia_semana", "es_festivo", "es_fin_semana",
    "cos_mes", "sin_mes",
    # Lags del precio
    "precio_bolsa_mean_lag1d", "precio_bolsa_mean_lag7d", "precio_bolsa_mean_lag30d",
    # Volatilidad rolling
    "precio_bolsa_mean_roll7d_std", "precio_bolsa_mean_roll30d_std",
]

_SARIMAX_ORDER = (1, 1, 1)
_SARIMAX_SEAS = (1, 0, 1, 7)   # ciclo semanal (precios colombianos tienen patron dia)

_LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}

HORIZONTE = Literal["corto", "tactico", "largo"]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _agregar_trig_mes(df: pd.DataFrame, col_mes: str = "mes") -> pd.DataFrame:
    """Agrega cos_mes y sin_mes si no existen ya."""
    if "cos_mes" not in df.columns:
        df = df.copy()
        df["cos_mes"] = np.cos(2 * np.pi * df[col_mes] / 12)
        df["sin_mes"] = np.sin(2 * np.pi * df[col_mes] / 12)
    return df


def _feats_disponibles(df: pd.DataFrame, lista: list[str]) -> list[str]:
    """Devuelve solo las columnas de `lista` que existen en `df`."""
    return [c for c in lista if c in df.columns]


def _log_precio(s: pd.Series) -> pd.Series:
    return np.log(s.clip(lower=1.0))


def _exp_precio(s: pd.Series) -> pd.Series:
    return np.exp(s)


# ---------------------------------------------------------------------------
# Modelo SARIMAX
# ---------------------------------------------------------------------------

@dataclass
class SARIMAXNivel:
    """Wrapper ligero sobre statsmodels.SARIMAX para precio de bolsa diario.

    Trabaja en log-espacio. El pronostico se devuelve en escala original.
    """
    order: tuple = _SARIMAX_ORDER
    seasonal_order: tuple = _SARIMAX_SEAS
    _result: object = field(default=None, init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "SARIMAXNivel":
        df = _agregar_trig_mes(df_train.copy())
        y = _log_precio(df["precio_bolsa_mean"])
        exog_cols = _feats_disponibles(df, _EXOG_SARIMAX)
        exog = df[exog_cols].ffill().fillna(0) if exog_cols else None

        model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = model.fit(disp=False)
        self._exog_cols = exog_cols
        logger.debug("SARIMAX ajustado. AIC=%.1f", self._result.aic)
        return self

    def forecast(self, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        """Pronostica `horizon` pasos hacia adelante.

        Returns DataFrame con: pred_log, pred, ci_lo90, ci_hi90.
        """
        if self._result is None:
            raise RuntimeError("Llamar fit() primero")

        exog_f = None
        if exog_future is not None and self._exog_cols:
            exog_f = _agregar_trig_mes(exog_future.copy())[self._exog_cols].ffill().fillna(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self._result.get_forecast(steps=horizon, exog=exog_f)

        mu = pred.predicted_mean
        ci = pred.conf_int(alpha=0.10)   # 90% interval

        return pd.DataFrame({
            "pred_log": mu.values,
            "pred": _exp_precio(mu).values,
            "ci_lo90": _exp_precio(ci.iloc[:, 0]).values,
            "ci_hi90": _exp_precio(ci.iloc[:, 1]).values,
        })


# ---------------------------------------------------------------------------
# Modelo LightGBM
# ---------------------------------------------------------------------------

@dataclass
class LGBNivel:
    """LightGBM sobre features tabulares para nivel diario del precio.

    Trabaja en log-espacio. Predice un paso (o varios con features preparadas).
    """
    params: dict = field(default_factory=lambda: dict(_LGB_PARAMS))
    _model: lgb.Booster | None = field(default=None, init=False, repr=False)
    _feature_cols: list[str] = field(default_factory=list, init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "LGBNivel":
        df = _agregar_trig_mes(df_train.copy())
        feature_cols = _feats_disponibles(df, _FEATS_LGB)
        self._feature_cols = feature_cols

        X = df[feature_cols].fillna(-9999)
        y = _log_precio(df["precio_bolsa_mean"])

        # Eliminar filas con NaN en el target
        mask = y.notna()
        dtrain = lgb.Dataset(X[mask], label=y[mask])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = lgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params.get("n_estimators", 400),
                valid_sets=[dtrain],
                callbacks=[lgb.log_evaluation(period=-1)],
            )
        logger.debug("LGB ajustado. %d features.", len(feature_cols))
        return self

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        """Devuelve predicciones en escala original."""
        if self._model is None:
            raise RuntimeError("Llamar fit() primero")
        df = _agregar_trig_mes(df_pred.copy())
        # Usar solo los features con los que fue entrenado
        avail = [c for c in self._feature_cols if c in df.columns]
        X = df[avail].fillna(-9999)
        # Features faltantes con respecto al entrenamiento -> rellenar con -9999
        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = -9999
        pred_log = self._model.predict(X[self._feature_cols])
        return _exp_precio(pd.Series(pred_log)).values

    @property
    def feature_importance(self) -> pd.Series:
        if self._model is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self._model.feature_importance(importance_type="gain"),
            index=self._model.feature_name(),
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

@dataclass
class EnsembleNivel:
    """Ensemble de SARIMAX + LightGBM con pesos adaptativos.

    Los pesos se calibran por inverse-MSE sobre un periodo de validacion
    o se pasan manualmente.

    Atributos publicos despues de fit():
        w_sarimax, w_lgb  -- pesos normalizados
        sarimax, lgb      -- modelos individuales
    """
    sarimax: SARIMAXNivel = field(default_factory=SARIMAXNivel)
    lgb: LGBNivel = field(default_factory=LGBNivel)
    w_sarimax: float = 0.5
    w_lgb: float = 0.5
    # Correccion de sesgo por horizonte: clave=h, valor=COP/kWh a restar del pronostico.
    # Se actualiza desde el loop mensual con la media de errores recientes.
    # No se estima automaticamente en fit() porque el sesgo depende del regimen actual
    # y puede cambiar entre entrenamiento y despliegue (El Nino vs post-El Nino).
    sesgo_por_horizonte: dict = field(default_factory=dict)

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame | None = None,
        df_full: pd.DataFrame | None = None,
    ) -> "EnsembleNivel":
        """Ajusta ambos modelos y calibra pesos si se pasa df_val.

        Si se pasa df_full, re-ajusta los componentes sobre toda la serie tras
        calibrar pesos: el pronostico de produccion debe partir del ultimo dato
        observado, no del fin de df_train. (En backtest no se pasa df_full, para
        no contaminar el walk-forward.)
        """
        self.sarimax.fit(df_train)
        self.lgb.fit(df_train)

        if df_val is not None and len(df_val) > 0:
            self._calibrar_pesos(df_val)

        if df_full is not None and len(df_full) > len(df_train):
            self.sarimax.fit(df_full)
            self.lgb.fit(df_full)
        return self

    def actualizar_sesgo(self, sesgo: dict) -> None:
        """Actualiza la correccion de sesgo desde el loop mensual.

        Parametros
        ----------
        sesgo : dict {horizonte_dias: sesgo_medio_COP}
            Sesgo medio (pred - real) de las ultimas N semanas de rolling-origin.
            Ejemplo: {1: 50.2, 7: 80.5, 14: 90.0, 30: 110.0}
        """
        self.sesgo_por_horizonte.update(sesgo)
        logger.info("Sesgos actualizados: %s", {k: f"{v:+.1f}" for k, v in self.sesgo_por_horizonte.items()})

    def _calibrar_pesos(self, df_val: pd.DataFrame) -> None:
        """Pesos inverse-MSE sobre el conjunto de validacion."""
        y_real = df_val["precio_bolsa_mean"].values

        # Predicciones SARIMAX en val (se pasan los exog del periodo de validacion)
        pred_sar = self.sarimax.forecast(len(df_val), exog_future=df_val)["pred"].values
        mse_sar = np.mean((y_real - pred_sar) ** 2)

        # Predicciones LGB en val
        pred_lgb = self.lgb.predict(df_val)
        mse_lgb = np.mean((y_real - pred_lgb) ** 2)

        # Inverse-MSE weights (evitar division por cero)
        eps = 1e-6
        w_sar_raw = 1.0 / (mse_sar + eps)
        w_lgb_raw = 1.0 / (mse_lgb + eps)
        total = w_sar_raw + w_lgb_raw
        self.w_sarimax = w_sar_raw / total
        self.w_lgb = w_lgb_raw / total

        logger.info(
            "Pesos ensemble: SARIMAX=%.2f  LGB=%.2f  "
            "(MSE: sarimax=%.1f  lgb=%.1f)",
            self.w_sarimax, self.w_lgb, mse_sar, mse_lgb,
        )

    def predict(
        self,
        df_val_o_test: pd.DataFrame,
        horizon: int | None = None,
        exog_future: pd.DataFrame | None = None,
        devolver_componentes: bool = False,
    ) -> pd.DataFrame:
        """Pronostico del ensemble.

        Para horizonte 1 (in-sample con features tabulares): usa predict() de LGB
        y forecast(1) de SARIMAX.
        Para horizonte > 1 (out-of-sample puro): exog_future debe contener los
        drivers futuros.

        Returns DataFrame con: pred, ci_lo90, ci_hi90.
        Si devolver_componentes=True, agrega pred_sarimax, pred_lgb.
        """
        n = len(df_val_o_test) if horizon is None else horizon

        # SARIMAX requiere exog futuros; si no se pasan explicitamente, usar df_val_o_test
        exog_for_sarimax = exog_future if exog_future is not None else df_val_o_test
        sar_df = self.sarimax.forecast(n, exog_future=exog_for_sarimax)
        pred_sar = sar_df["pred"].values

        # Componente LGB
        if horizon is None:
            pred_lgb = self.lgb.predict(df_val_o_test)
        else:
            # Si hay exog_future, usarla; si no, usar el ultimo estado
            df_for_lgb = exog_future if exog_future is not None else df_val_o_test.tail(n)
            pred_lgb = self.lgb.predict(df_for_lgb)

        # Ensemble
        pred = self.w_sarimax * pred_sar + self.w_lgb * pred_lgb

        # Correccion de sesgo por horizonte (si fue actualizada desde el loop mensual)
        if self.sesgo_por_horizonte and n in self.sesgo_por_horizonte:
            pred = pred - self.sesgo_por_horizonte[n]

        # Ancho del CI en log-espacio desde SARIMAX, centrado en el ensemble
        log_pred_sar = sar_df["pred_log"].values
        log_lo90 = np.log(np.maximum(sar_df["ci_lo90"].values, 1.0))
        log_hi90 = np.log(np.maximum(sar_df["ci_hi90"].values, 1.0))
        half_width = (log_hi90 - log_lo90) / 2.0
        log_ensemble = np.log(np.maximum(pred, 1.0))

        result = pd.DataFrame({
            "pred": pred,
            "ci_lo90": np.exp(log_ensemble - half_width),
            "ci_hi90": np.exp(log_ensemble + half_width),
        })
        if devolver_componentes:
            result["pred_sarimax"] = pred_sar
            result["pred_lgb"] = pred_lgb
        return result
