"""Backtest rolling-origin mensual del IPP, con benchmarks.

Por qué existe: **el IPP nunca se validó out-of-sample**. `scripts/ejecutar_backtest.py` es
solo de bolsa, `outputs/backtest/*.parquet` solo tenía bolsa, y los tests del IPP son de forma
y dirección (shapes, positividad, que los escenarios se separen), no de error. Los pesos
inverse-MSE que calcula `EnsembleIPP._calibrar_pesos` no sustituyen esto: comparan un
forecast multi-paso del SARIMA contra un LGB de ~1 paso alimentado con los lags reales, y le
pasan al SARIMAX las exógenas realizadas (foresight perfecto).

Sin backtest no se puede afirmar que un cambio mejora nada. Este módulo es el prerrequisito
de todo lo demás.

La pieza crítica es `congelar_drivers_futuros`: el análogo mensual de `_fix_lags_precio` de
`scripts/ejecutar_backtest.py`. Sin ella, el "backtest" le entrega al modelo los valores
futuros reales de TRM, Brent y los lags del propio IPP, y mide una precisión que en producción
no existe.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from proybolsa.backtest.rolling_origin import RollingOriginBacktest, calcular_metricas

logger = logging.getLogger(__name__)

Z90 = 1.6448536269514722  # cuantil normal al 95% (intervalo central del 90%)

# Columnas que en producción NO se conocen para el futuro y que por tanto hay que congelar.
# Incluye los lags del propio IPP: en un pronóstico a h meses, `ipp_lag1m` del mes 2 depende
# de la predicción del mes 1, no del valor real.
_COLS_DRIVER_FUTURO = [
    "ipp_lag1m", "ipp_lag2m", "ipp_lag3m", "ipp_lag12m",
    "trm", "brent", "brent_cop",
    "brent_cop_lag1m", "brent_cop_lag2m", "brent_cop_lag3m",
    "trm_yoy", "trm_yoy_lag1m", "brent_yoy", "brent_yoy_lag1m",
    "oni_lag", "enso_el_nino", "enso_la_nina",
    "ppi_usa", "ppi_usa_lag1m", "ppi_usa_lag2m", "ppi_usa_lag3m",
]

# Columnas deterministas del calendario: SÍ se conocen para el futuro.
_COLS_CALENDARIO = ["fecha", "mes", "cos_mes", "sin_mes"]

ModoDrivers = Literal["congelado", "perfecto", "escenario"]


def congelar_drivers_futuros(
    df_h: pd.DataFrame,
    df_train: pd.DataFrame,
    modo: ModoDrivers = "congelado",
    df_escenario: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepara las exógenas del tramo futuro sin filtrar información del futuro.

    modo="congelado" (default y el único honesto para medir precisión): cada driver se fija
        en su último valor observado en `df_train`. Es lo que el modelo tendría en producción
        con `_construir_futuro_naive`.
    modo="perfecto": deja los valores reales. NO mide precisión alcanzable — existe para
        cuantificar la cota superior del valor de los drivers: si con foresight perfecto un
        modelo no le gana a un univariado, sus drivers no aportan señal predictiva.
    modo="escenario": usa una trayectoria provista.
    """
    fut = df_h.copy()

    if modo == "perfecto":
        return fut
    if modo == "escenario":
        if df_escenario is None:
            raise ValueError("modo='escenario' requiere df_escenario")
        for c in df_escenario.columns:
            if c in fut.columns and c != "fecha":
                fut[c] = df_escenario[c].values[: len(fut)]
        return fut

    ultima = df_train.iloc[-1]
    for c in _COLS_DRIVER_FUTURO:
        if c in fut.columns:
            fut[c] = ultima[c] if c in df_train.columns else np.nan
    return fut


# ---------------------------------------------------------------------------
# Benchmarks: modelos de primera clase, con la misma interfaz fit/forecast
# ---------------------------------------------------------------------------

@dataclass
class RandomWalkIPP:
    """`pred = último valor observado`. La vara mínima: ¿el modelo aporta sobre no hacer nada?

    Banda desde la desviación de los cambios log mensuales, escalada con sqrt(h), que es la
    varianza correcta para un paseo aleatorio.
    """
    _ultimo: float = field(default=np.nan, init=False, repr=False)
    _sd_dlog: float = field(default=np.nan, init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "RandomWalkIPP":
        y = df_train["ipp"].dropna()
        self._ultimo = float(y.iloc[-1])
        self._sd_dlog = float(np.log(y).diff().dropna().std(ddof=1))
        return self

    def forecast(self, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        pasos = np.arange(1, horizon + 1)
        pred = np.full(horizon, self._ultimo, dtype=float)
        half = Z90 * self._sd_dlog * np.sqrt(pasos)
        log_p = np.log(np.maximum(pred, 1e-9))
        return pd.DataFrame({
            "pred": pred,
            "ci_lo90": np.exp(log_p - half),
            "ci_hi90": np.exp(log_p + half),
        })


@dataclass
class DriftIPP:
    """Random walk con deriva: `pred_h = y_T * exp(h * media(dlog))`.

    Equivale a ARIMA(0,1,0) con `trend="c"`. Con `ventana=12` la deriva es la media de los
    últimos 12 cambios mensuales, que es el benchmark "media de los últimos 12 dlog".
    """
    ventana: int | None = None
    _ultimo: float = field(default=np.nan, init=False, repr=False)
    _drift: float = field(default=0.0, init=False, repr=False)
    _sd: float = field(default=np.nan, init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "DriftIPP":
        y = df_train["ipp"].dropna()
        dlog = np.log(y).diff().dropna()
        if self.ventana:
            dlog = dlog.tail(self.ventana)
        self._ultimo = float(y.iloc[-1])
        self._drift = float(dlog.mean())
        self._sd = float(dlog.std(ddof=1)) if len(dlog) > 1 else 0.0
        return self

    def forecast(self, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        pasos = np.arange(1, horizon + 1)
        log_p = np.log(max(self._ultimo, 1e-9)) + self._drift * pasos
        half = Z90 * self._sd * np.sqrt(pasos)
        return pd.DataFrame({
            "pred": np.exp(log_p),
            "ci_lo90": np.exp(log_p - half),
            "ci_hi90": np.exp(log_p + half),
        })


@dataclass
class ThetaIPP:
    """Método Theta (Assimakopoulos & Nikolopoulos, 2000), sobre log."""
    _res: object = field(default=None, init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "ThetaIPP":
        from statsmodels.tsa.forecasting.theta import ThetaModel

        y = np.log(df_train["ipp"].dropna()).reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = ThetaModel(y, period=12, deseasonalize=False).fit()
        return self

    def forecast(self, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = np.asarray(self._res.forecast(horizon), dtype=float)
            try:
                ci = self._res.prediction_intervals(horizon, alpha=0.10)
                lo = np.exp(np.asarray(ci.iloc[:, 0], dtype=float))
                hi = np.exp(np.asarray(ci.iloc[:, 1], dtype=float))
            except Exception:
                lo = hi = np.full(horizon, np.nan)
        return pd.DataFrame({"pred": np.exp(mu), "ci_lo90": lo, "ci_hi90": hi})


# ---------------------------------------------------------------------------
# Adaptador para el motor genérico
# ---------------------------------------------------------------------------

def _llamar_forecast(modelo, horizon: int, exog: pd.DataFrame | None) -> pd.DataFrame:
    """Invoca forecast/predict aceptando las firmas heterogéneas que ya existen.

    `SARIMABaselineIPP.forecast(horizon)` no acepta exógenas, `SARIMAXDriversIPP.forecast`
    sí, y `EnsembleIPP` expone `predict(horizon, exog_future=...)` en vez de `forecast`. En
    vez de uniformar las firmas ahora (que rompería tests y pickles), se resuelve aquí.
    """
    fn = getattr(modelo, "forecast", None) or getattr(modelo, "predict")
    params = inspect.signature(fn).parameters
    if "exog_future" in params and exog is not None:
        return fn(horizon, exog_future=exog)
    return fn(horizon)


@dataclass
class _Adaptador:
    """Traduce (fábrica de modelo, modo de drivers) al par fn_fit/fn_predict del motor."""
    fabrica: Callable[[], object]
    modo_drivers: ModoDrivers = "congelado"
    _df_train: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def fit(self, df_train: pd.DataFrame):
        self._df_train = df_train
        modelo = self.fabrica()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo.fit(df_train)
        return modelo

    def predict(self, modelo, df_h: pd.DataFrame, h: int) -> pd.DataFrame:
        exog = congelar_drivers_futuros(df_h, self._df_train, self.modo_drivers)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _llamar_forecast(modelo, h, exog)


def _fabricas() -> dict[str, Callable[[], object]]:
    """Registro de modelos backtesteables. La clave es la que va a la columna `modelo`."""
    from proybolsa.models.ipp.modelo_ipp import (
        EnsembleIPP,
        SARIMABaselineIPP,
        SARIMAXDriversIPP,
    )

    fabricas: dict[str, Callable[[], object]] = {
        # Benchmarks
        "ipp_bench_rw": RandomWalkIPP,
        "ipp_bench_drift": DriftIPP,
        "ipp_bench_drift12": lambda: DriftIPP(ventana=12),
        "ipp_bench_theta": ThetaIPP,
        # Estado actual (línea base "antes")
        "ipp_sarima_actual": SARIMABaselineIPP,
        "ipp_sarimax_actual": SARIMAXDriversIPP,
        "ipp_ensemble_actual": EnsembleIPP,
    }

    # Modelos nuevos: se registran solo si el módulo existe, para que este archivo funcione
    # tanto antes como después de la Fase 2.3.
    try:
        from proybolsa.models.ipp import componentes as _c
    except ImportError:
        return fabricas

    for clave, attr in (
        ("ipp_arima_drift", "ARIMADriftIPP"),
        ("ipp_ets_damped", "ETSDampedIPP"),
        ("ipp_lgb_dlog", "LGBDlogIPP"),
    ):
        if hasattr(_c, attr):
            fabricas[clave] = getattr(_c, attr)
    return fabricas


def backtest_ipp(
    df: pd.DataFrame,
    claves: list[str] | None = None,
    *,
    horizontes: tuple[int, ...] = (1, 3, 6, 12, 24),
    min_train: int = 60,
    step: int = 1,
    modo_drivers: ModoDrivers = "congelado",
    refit_cada: int = 1,
    registrar_todos_los_pasos: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corre el rolling-origin mensual para las `claves` pedidas.

    Devuelve `(errores, metricas)` con columnas `variable='ipp'` y `modelo=<clave>`, listos
    para concatenarse con el backtest de bolsa en un único parquet.
    """
    fabricas = _fabricas()
    claves = list(claves) if claves else list(fabricas)
    desconocidas = [c for c in claves if c not in fabricas]
    if desconocidas:
        raise KeyError(f"modelos desconocidos: {desconocidas}. Disponibles: {sorted(fabricas)}")

    base = df.sort_values("fecha").dropna(subset=["ipp"]).reset_index(drop=True)
    h_max = max(horizontes)

    errores_all, metricas_all = [], []
    for clave in claves:
        logger.info("Backtest IPP: %s (%d obs, min_train=%d, h_max=%d, drivers=%s)",
                    clave, len(base), min_train, h_max, modo_drivers)
        motor = RollingOriginBacktest(
            horizonte_max=h_max,
            horizontes_evaluar=list(horizontes),
            min_train=min_train,
            step=step,
            etiqueta_periodo="meses",
            refit_cada=refit_cada,
            registrar_todos_los_pasos=registrar_todos_los_pasos,
        )
        ad = _Adaptador(fabricas[clave], modo_drivers)
        try:
            err, _ = motor.evaluar(base, "fecha", "ipp", ad.fit, ad.predict, verbose=False)
        except Exception as exc:
            logger.error("  %s: backtest fallo -> %s", clave, exc)
            continue

        err = err.assign(variable="ipp", modelo=clave, modo_drivers=modo_drivers)
        met = calcular_metricas(err, variable="ipp", modelo=clave)
        met = met.assign(modo_drivers=modo_drivers)
        errores_all.append(err)
        metricas_all.append(met)

        resumen = "  ".join(
            f"h{int(r.horizonte)}={r.rmse:.2f}" for r in met.itertuples() if not np.isnan(r.rmse)
        )
        logger.info("  %s RMSE: %s", clave, resumen)

    if not errores_all:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.concat(errores_all, ignore_index=True),
            pd.concat(metricas_all, ignore_index=True))


def comparar_contra_benchmark(errores: pd.DataFrame, clave_modelo: str,
                              clave_benchmark: str = "auto") -> pd.DataFrame:
    """Diebold-Mariano del modelo contra un benchmark, por horizonte.

    Con `clave_benchmark='auto'` se elige, para cada horizonte, el benchmark con menor RMSE.
    Es la comparación exigente: no basta ganarle al random walk si el drift es mejor.
    """
    from proybolsa.backtest.rolling_origin import diebold_mariano

    benchmarks = [c for c in errores["modelo"].unique() if c.startswith("ipp_bench_")]
    filas = []
    for h in sorted(errores["horizonte"].unique()):
        eh = errores[errores["horizonte"] == h]
        em = eh[eh["modelo"] == clave_modelo].sort_values("fecha_corte")
        if em.empty:
            continue

        if clave_benchmark == "auto":
            rmses = {b: np.sqrt(np.mean(eh[eh["modelo"] == b]["error"].dropna() ** 2))
                     for b in benchmarks if not eh[eh["modelo"] == b].empty}
            if not rmses:
                continue
            bench = min(rmses, key=rmses.get)
        else:
            bench = clave_benchmark

        eb = eh[eh["modelo"] == bench].sort_values("fecha_corte")
        comun = set(em["fecha_corte"]) & set(eb["fecha_corte"])
        if len(comun) < 5:
            continue
        em2 = em[em["fecha_corte"].isin(comun)].sort_values("fecha_corte")
        eb2 = eb[eb["fecha_corte"].isin(comun)].sort_values("fecha_corte")

        dm = diebold_mariano(em2["error"] ** 2, eb2["error"] ** 2, h=int(h), alternativa="a_mejor")
        rmse_m = float(np.sqrt(np.mean(em2["error"] ** 2)))
        rmse_b = float(np.sqrt(np.mean(eb2["error"] ** 2)))
        filas.append({
            "horizonte": int(h),
            "modelo": clave_modelo,
            "benchmark": bench,
            "rmse_modelo": rmse_m,
            "rmse_benchmark": rmse_b,
            "skill": 1 - rmse_m / rmse_b if rmse_b else np.nan,
            "dm_stat": dm["estadistico_dm"],
            "p_valor": dm["p_valor"],
            "n": len(comun),
        })
    return pd.DataFrame(filas)
