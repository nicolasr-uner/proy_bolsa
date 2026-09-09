"""Backtesting rolling-origin (walk-forward) para los modelos de bolsa e IPP.

Metodologia:
  1. Se define un origen inicial (inicio_evaluacion).
  2. En cada iteracion se ajusta el modelo con todos los datos hasta la fecha de corte.
  3. Se generan pronosticos a h = 1, 7, 14, 30 dias hacia adelante.
  4. Se desplaza la ventana un paso (step_dias) y se repite.
  5. Se calculan metricas por horizonte y se devuelven los errores detallados.

Diferencia vs hold-out unico: el rolling-origin evita que el resultado dependa
de una sola division y mide la degradacion real del error con el horizonte.

Referencias:
  Tashman (2000) - Out-of-sample tests of forecasting accuracy
  Hyndman & Athanasopoulos (2021) - Forecasting: Principles & Practice, Ch. 5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (y_true != 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _sesgo(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sesgo = media del error (positivo = sobreestima, negativo = subestima)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(y_pred[mask] - y_true[mask]))


def _cobertura_ci(
    y_true: np.ndarray, ci_lo: np.ndarray, ci_hi: np.ndarray
) -> float:
    """Fraccion de observaciones dentro del CI. Objetivo ~90% para CI al 90%."""
    mask = ~(np.isnan(y_true) | np.isnan(ci_lo) | np.isnan(ci_hi))
    if mask.sum() == 0:
        return np.nan
    dentro = (y_true[mask] >= ci_lo[mask]) & (y_true[mask] <= ci_hi[mask])
    return float(dentro.mean() * 100)


def _ancho_ci(ci_lo: np.ndarray, ci_hi: np.ndarray) -> float:
    mask = ~(np.isnan(ci_lo) | np.isnan(ci_hi))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(ci_hi[mask] - ci_lo[mask]))


def _winkler(y_true: np.ndarray, ci_lo: np.ndarray, ci_hi: np.ndarray,
             alpha: float = 0.10) -> float:
    """Winkler score del intervalo (menor es mejor).

    Penaliza el ancho y castiga las observaciones que caen fuera. Es la métrica que hace
    falta para no premiar a un intervalo por ser ancho: la cobertura sola se maximiza con
    bandas absurdas, y el ancho solo se minimiza con bandas que no cubren nada.
    """
    mask = ~(np.isnan(y_true) | np.isnan(ci_lo) | np.isnan(ci_hi))
    if mask.sum() == 0:
        return np.nan
    y, lo, hi = y_true[mask], ci_lo[mask], ci_hi[mask]
    score = hi - lo
    score = np.where(y < lo, score + 2.0 / alpha * (lo - y), score)
    score = np.where(y > hi, score + 2.0 / alpha * (y - hi), score)
    return float(np.mean(score))


def calcular_metricas(errores: pd.DataFrame, *, variable: str | None = None,
                      modelo: str | None = None) -> pd.DataFrame:
    """Agrega los errores detallados en metricas por horizonte.

    Parametros
    ----------
    errores  : DataFrame con columnas: horizonte, y_real, y_pred, ci_lo90, ci_hi90
    variable : etiqueta de la serie ('bolsa', 'ipp'). Se emite como columna si se pasa, para
               que un mismo parquet pueda consolidar backtests de varias series.
    modelo   : clave del modelo. Debe coincidir con la del registro de modelos.
    """
    grupos = errores.groupby("horizonte")
    rows = []
    for h, g in grupos:
        tiene_ci = "ci_lo90" in g.columns
        # `n_predicciones` cuenta filas; `n_validas` cuenta las que realmente produjeron un
        # numero. La distincion importa: un modelo que solo pudo pronosticar en 1 de 56
        # origenes (p. ej. el VECM cuando su regla de activacion casi nunca se cumple) muestra
        # un RMSE calculado sobre esa unica observacion, que parece excelente y no significa
        # nada. Sin esta columna, ese numero viaja al parquet sin ninguna advertencia.
        n_validas = int((~(g["y_real"].isna() | g["y_pred"].isna())).sum())
        fila = {
            "horizonte": h,
            "n_predicciones": len(g),
            "n_validas": n_validas,
            "rmse": _rmse(g["y_real"].values, g["y_pred"].values),
            "mae": _mae(g["y_real"].values, g["y_pred"].values),
            "mape_pct": _mape(g["y_real"].values, g["y_pred"].values),
            "sesgo": _sesgo(g["y_real"].values, g["y_pred"].values),
            "cobertura_ci90_pct": _cobertura_ci(
                g["y_real"].values, g["ci_lo90"].values, g["ci_hi90"].values
            ) if tiene_ci else np.nan,
            "ancho_ci90_medio": _ancho_ci(
                g["ci_lo90"].values, g["ci_hi90"].values
            ) if tiene_ci else np.nan,
            "winkler_90": _winkler(
                g["y_real"].values, g["ci_lo90"].values, g["ci_hi90"].values
            ) if tiene_ci else np.nan,
        }
        if variable is not None:
            fila["variable"] = variable
        if modelo is not None:
            fila["modelo"] = modelo
        rows.append(fila)
    return pd.DataFrame(rows).sort_values("horizonte").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Test de Diebold-Mariano
# ---------------------------------------------------------------------------

def diebold_mariano(
    errores_a: pd.Series,
    errores_b: pd.Series,
    h: int = 1,
    alternativa: str = "dos_colas",
) -> dict:
    """Test de Diebold-Mariano para comparar dos modelos.

    H0: ambos modelos tienen la misma precision.
    H1: modelo A es mejor (alternativa='a_mejor'), B es mejor ('b_mejor'),
        o son distintos ('dos_colas').

    Parametros
    ----------
    errores_a : errores al cuadrado del modelo A
    errores_b : errores al cuadrado del modelo B
    h         : horizonte de pronostico (para correccion HAC)
    alternativa: 'dos_colas', 'a_mejor', 'b_mejor'

    Devuelve dict con: estadistico_dm, p_valor, conclusion.
    """
    d = errores_a.values - errores_b.values
    n = len(d)
    if n < 5:
        return {"estadistico_dm": np.nan, "p_valor": np.nan, "conclusion": "insuficientes datos"}

    d_mean = np.mean(d)
    # Varianza HAC (Newey-West) para corregir autocorrelacion hasta lag h-1
    gamma0 = np.var(d, ddof=0)
    gammas = sum(
        (1 - lag / h) * np.cov(d[lag:], d[:-lag], bias=True)[0, 1]
        for lag in range(1, h)
    ) if h > 1 else 0
    var_hac = (gamma0 + 2 * gammas) / n
    if var_hac <= 0:
        return {"estadistico_dm": np.nan, "p_valor": np.nan, "conclusion": "varianza nula"}

    dm_stat = d_mean / np.sqrt(var_hac)

    if alternativa == "dos_colas":
        p_valor = 2 * stats.t.sf(abs(dm_stat), df=n - 1)
    elif alternativa == "a_mejor":  # A < B (menores errores)
        p_valor = stats.t.sf(-dm_stat, df=n - 1)
    else:  # b_mejor
        p_valor = stats.t.sf(dm_stat, df=n - 1)

    if p_valor < 0.05:
        mejor = "A" if dm_stat < 0 else "B"
        conclusion = f"Modelo {mejor} es significativamente mejor (p={p_valor:.3f})"
    else:
        conclusion = f"Sin diferencia significativa (p={p_valor:.3f})"

    return {
        "estadistico_dm": round(dm_stat, 3),
        "p_valor": round(p_valor, 4),
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# Motor de rolling-origin
# ---------------------------------------------------------------------------

@dataclass
class RollingOriginBacktest:
    """Evalua un modelo con rolling-origin sobre una serie historica.

    El modelo se reajusta en cada corte (por defecto). Para modelos pesados
    (SARIMAX), se puede usar refit_cada=N para reajustar solo cada N pasos.

    Parametros
    ----------
    horizonte_max      : Numero maximo de dias hacia adelante a evaluar
    step_dias          : Cuantos dias avanza la ventana en cada iteracion
    min_train_dias     : Minimo de dias de entrenamiento para el primer origen
    horizontes_evaluar : Lista de horizontes h para registrar errores
    refit_cada         : Refitear cada N iteraciones (1 = siempre, util para ahorrar tiempo)
    """

    horizonte_max: int = 30
    step_dias: int = 7
    min_train_dias: int = 365
    horizontes_evaluar: list[int] = field(default_factory=lambda: [1, 7, 14, 30])
    refit_cada: int = 1

    # --- Alias neutros de periodo ---
    # El motor cuenta FILAS, no días: con una fila por mes, `min_train_dias=60` significa 60
    # meses. Solo los nombres mienten. Estos alias permiten usarlo con series mensuales sin
    # escribir `min_train_dias` para hablar de meses, y sin romper a los llamadores actuales.
    min_train: int | None = None
    step: int | None = None
    etiqueta_periodo: str = "dias"

    # Registrar los pasos 1..h y no solo el paso h. Necesario para calibrar la incertidumbre
    # por horizonte: con solo el último paso, un backtest de h=24 aporta un dato por origen.
    registrar_todos_los_pasos: bool = False

    # `range(min_train, n - horizonte_max, step)` descartaba el último origen válido: con
    # n filas, el corte `n - horizonte_max` todavía deja exactamente `horizonte_max` filas de
    # test. Con 54 orígenes mensuales, perder uno es el 2% de la muestra.
    incluir_ultimo_origen: bool = True

    def __post_init__(self) -> None:
        self._min_train = self.min_train if self.min_train is not None else self.min_train_dias
        self._step = self.step if self.step is not None else self.step_dias

    def evaluar(
        self,
        df: pd.DataFrame,
        col_fecha: str,
        col_target: str,
        fn_fit: Callable,
        fn_predict: Callable,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Ejecuta el backtesting rolling-origin.

        Parametros
        ----------
        df         : Dataset completo con fechas y target
        col_fecha  : Columna de fecha (date o datetime)
        col_target : Columna del target (y)
        fn_fit     : Funcion fit(df_train) -> modelo
        fn_predict : Funcion predict(modelo, df_futuro, h) -> DataFrame con 'pred'
                     (y opcionalmente 'ci_lo90', 'ci_hi90')

        Returns
        -------
        errores_detalle : DataFrame con una fila por (origen, horizonte)
        metricas_resumen: DataFrame con metricas agregadas por horizonte
        """
        df = df.copy().sort_values(col_fecha).reset_index(drop=True)
        fechas = pd.to_datetime(df[col_fecha])

        n = len(df)
        tope = n - self.horizonte_max + (1 if self.incluir_ultimo_origen else 0)
        origenes = range(self._min_train, tope, self._step)
        n_origenes = len(list(origenes))

        if n_origenes == 0:
            raise ValueError(
                f"Sin suficientes datos. Necesitas al menos "
                f"{self._min_train + self.horizonte_max} filas, tienes {n}."
            )

        logger.info(
            "Rolling-origin: %d origenes, step=%d %s, horizonte_max=%d",
            n_origenes, self._step, self.etiqueta_periodo, self.horizonte_max,
        )

        errores = []
        modelo = None

        for i, corte in enumerate(origenes):
            df_train = df.iloc[:corte]
            df_test = df.iloc[corte: corte + self.horizonte_max]
            fecha_corte = fechas.iloc[corte - 1]

            if i % self.refit_cada == 0:
                modelo = fn_fit(df_train)

            if self.registrar_todos_los_pasos:
                # UNA proyección a horizonte_max, registrando cada paso 1..h_max. No se
                # itera `horizontes_evaluar` porque entonces el paso 1 quedaría registrado
                # una vez por cada h de la lista.
                h_max = min(self.horizonte_max, len(df_test))
                if h_max == 0:
                    continue
                df_h = df_test.iloc[:h_max]
                try:
                    pred_df = fn_predict(modelo, df_h, h_max)
                    y_real = df_h[col_target].values
                    y_pred = (pred_df["pred"].values if "pred" in pred_df.columns
                              else np.asarray(pred_df).ravel())
                    ci_lo = (pred_df["ci_lo90"].values if "ci_lo90" in pred_df.columns
                             else np.full(h_max, np.nan))
                    ci_hi = (pred_df["ci_hi90"].values if "ci_hi90" in pred_df.columns
                             else np.full(h_max, np.nan))
                    for paso in range(1, min(h_max, len(y_pred)) + 1):
                        j = paso - 1
                        errores.append({
                            "fecha_corte": fecha_corte,
                            "horizonte": paso,
                            "y_real": float(y_real[j]),
                            "y_pred": float(y_pred[j]),
                            "ci_lo90": float(ci_lo[j]),
                            "ci_hi90": float(ci_hi[j]),
                            "error": float(y_pred[j]) - float(y_real[j]),
                        })
                except Exception as exc:
                    logger.warning("Error en origen=%d (todos los pasos): %s", corte, exc)
                if verbose and (i + 1) % max(1, n_origenes // 10) == 0:
                    logger.info("  Progreso: %d/%d origenes", i + 1, n_origenes)
                continue

            for h in self.horizontes_evaluar:
                if h > len(df_test):
                    continue
                df_h = df_test.iloc[:h]
                try:
                    pred_df = fn_predict(modelo, df_h, h)
                    y_real = df_h[col_target].values
                    y_pred = pred_df["pred"].values if "pred" in pred_df.columns else pred_df.values
                    ci_lo = pred_df.get("ci_lo90", pd.Series([np.nan] * h)).values if hasattr(pred_df, "get") else np.nan
                    ci_hi = pred_df.get("ci_hi90", pd.Series([np.nan] * h)).values if hasattr(pred_df, "get") else np.nan

                    # Solo el ultimo punto del horizonte h
                    errores.append({
                        "fecha_corte": fecha_corte,
                        "horizonte": h,
                        "y_real": float(y_real[-1]),
                        "y_pred": float(y_pred[-1]) if len(y_pred) >= h else np.nan,
                        "ci_lo90": float(ci_lo[-1]) if hasattr(ci_lo, "__len__") and len(ci_lo) >= h else np.nan,
                        "ci_hi90": float(ci_hi[-1]) if hasattr(ci_hi, "__len__") and len(ci_hi) >= h else np.nan,
                        "error": float(y_pred[-1]) - float(y_real[-1]) if len(y_pred) >= h else np.nan,
                    })
                except Exception as exc:
                    logger.warning("Error en origen=%d h=%d: %s", corte, h, exc)
                    errores.append({
                        "fecha_corte": fecha_corte, "horizonte": h,
                        "y_real": np.nan, "y_pred": np.nan,
                        "ci_lo90": np.nan, "ci_hi90": np.nan, "error": np.nan,
                    })

            if verbose and (i + 1) % max(1, n_origenes // 10) == 0:
                logger.info("  Progreso: %d/%d origenes", i + 1, n_origenes)

        errores_df = pd.DataFrame(errores)
        metricas = calcular_metricas(errores_df)
        return errores_df, metricas


def skill_vs_naive_por_horizonte(metricas: pd.DataFrame, modelo: str, naive: str,
                                 col_modelo: str = "modelo", col_h: str = "horizonte",
                                 col_rmse: str = "rmse") -> dict:
    """Skill score (1 - RMSE_modelo/RMSE_naive) por horizonte, desde las metricas del backtest.

    Positivo = el modelo le gana al naive; 0 = empata; negativo = pierde. Se publica en los
    resumen_*.json del ciclo mensual para que cada corrida diga, sin adornos, si el modelo
    desplegado supera a "no hacer casi nada". Devuelve {horizonte: skill_redondeado}.
    """
    if metricas is None or metricas.empty:
        return {}
    req = {col_modelo, col_h, col_rmse}
    if not req <= set(metricas.columns):
        return {}
    rmse_m = metricas[metricas[col_modelo] == modelo].set_index(col_h)[col_rmse]
    rmse_n = metricas[metricas[col_modelo] == naive].set_index(col_h)[col_rmse]
    out: dict[int, float] = {}
    for h in rmse_m.index:
        if h in rmse_n.index and float(rmse_n[h]) > 0:
            out[int(h)] = round(float(1 - float(rmse_m[h]) / float(rmse_n[h])), 3)
    return out
