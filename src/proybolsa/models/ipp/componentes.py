"""Componentes nuevos del modelo de IPP (Fase 2 del plan de mejoras).

Qué problema resuelven, medido con `backtest_ipp` sobre 56 orígenes idénticos (muestra
2015-01 a 2026-07, min_train=60, drivers congelados):

    modelo                       h=1    h=6    h=12   h=24
    ipp_bench_drift             1.62   7.60   14.02  24.25   <- random walk con deriva
    ipp_sarima_actual (1,1,1)   1.58   8.16   18.46  49.72   <- el componente en produccion

El componente en producción tiene **más del doble de error que un random walk con deriva** a
24 meses. La causa es que `SARIMABaselineIPP` fija `order=(1,1,1)` sin `trend`, así que el
pronóstico converge a una recta plana: proyecta +0.4% a 12 meses contra una deriva histórica
de +5.1%/año sostenida durante 27 años.

`ARIMADriftIPP` corrige exactamente eso: incluye `trend="c"` entre los candidatos y elige el
orden por AIC/BIC **dentro de `fit`**, no una vez sobre la serie completa (seleccionar el
orden viendo todos los datos y después backtestear contamina el resultado).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from proybolsa.models.sarimax_pickle import PickleCompactoSARIMAX

logger = logging.getLogger(__name__)

Z90 = 1.6448536269514722


def _terminos_fourier(fechas: pd.Series, k: int) -> pd.DataFrame | None:
    """Armónicos anuales como exógenas deterministas.

    Se prefiere estacionalidad determinista sobre un componente SARIMA (P,D,Q,12): en la
    muestra 2015+ el F-test de dummies de mes sobre dlog no rechaza (p=0.33), y un componente
    estacional estocástico gasta parámetros en un patrón que apenas existe.
    """
    if k <= 0:
        return None
    mes = pd.to_datetime(fechas).dt.month.to_numpy()
    cols = {}
    for j in range(1, k + 1):
        cols[f"fourier_cos{j}"] = np.cos(2 * np.pi * j * mes / 12)
        cols[f"fourier_sin{j}"] = np.sin(2 * np.pi * j * mes / 12)
    return pd.DataFrame(cols, index=fechas.index)


@dataclass
class ARIMADriftIPP(PickleCompactoSARIMAX):
    """ARIMA sobre log(IPP) con término de deriva y orden elegido por criterio de información.

    Diferencias con `SARIMABaselineIPP`:
      - `trend="c"` está SIEMPRE entre los candidatos (con d=1 eso es una deriva).
      - El orden se busca en una rejilla y se elige por AIC/BIC, dentro de `fit`.
      - La estacionalidad, si se usa, es determinista (Fourier) y no un (P,D,Q,12).

    ATENCIÓN — el default es orden FIJO (1,1,0) con deriva, no búsqueda. Medido sobre los
    mismos 56 orígenes:

        variante                     h=1    h=6    h=12   h=24
        (1,1,0)+drift  [default]    1.55   7.08   13.50  24.11
        (1,1,0) sin drift           1.58   7.90   15.69  28.20
        (1,1,1)+drift               1.61   7.77   15.54  32.03
        busqueda por AIC            1.63   8.24   17.88  48.01
        ipp_bench_drift             1.77   7.60   14.02  24.25

    La búsqueda por AIC dentro de cada origen **empeora** el pronóstico: casi duplica el
    error a 24 meses. El conteo de órdenes elegidos explica por qué: la rejilla escoge
    (1,1,0)+c en 34 de 56 orígenes, pero en 7 elige un modelo **sin deriva** y en el resto
    órdenes más altos. AIC mide ajuste dentro de muestra; a 24 meses lo único que pesa es que
    el término de deriva esté, y la varianza que introduce elegir por origen cuesta mucho más
    de lo que aporta la flexibilidad. Ampliar la rejilla de p_max/q_max=1 a 3 no cambió nada
    (RMSE idéntico): el daño viene de que el `trend` se voltee, no del orden.

    `reseleccionar_orden=True` se conserva para el estudio (§5 del informe), no para producción.
    """
    criterio: Literal["aic", "bic"] = "aic"
    p_max: int = 3
    q_max: int = 3
    fourier_k: int = 0
    con_drift: bool = True
    reseleccionar_orden: bool = False
    orden_fijo: tuple | None = (1, 1, 0)
    trend_fijo: str | None = "c"

    _result: object = field(default=None, init=False, repr=False)
    _endog: object = field(default=None, init=False, repr=False)
    _exog: object = field(default=None, init=False, repr=False)
    _order: tuple = field(default=(1, 1, 0), init=False, repr=False)
    _seasonal_order: tuple = field(default=(0, 0, 0, 0), init=False, repr=False)
    _trend: str | None = field(default="c", init=False, repr=False)
    _diagnostico: dict = field(default_factory=dict, init=False, repr=False)

    # `PickleCompactoSARIMAX._reconstruir_result` no conoce `trend`, así que se sobreescribe.
    def _reconstruir_result(self, params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = SARIMAX(
                self._endog, exog=self._exog, order=self._order,
                seasonal_order=self._seasonal_order, trend=self._trend,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            return modelo.filter(params)

    def _candidatos(self) -> list[tuple[tuple, str | None]]:
        # `reseleccionar_orden` manda: con el default (orden_fijo poblado) hay que poder pedir
        # la búsqueda explícitamente para el estudio, sin tener que anular `orden_fijo`.
        if not self.reseleccionar_orden and self.orden_fijo is not None:
            return [(self.orden_fijo, self.trend_fijo)]
        trends: list[str | None] = ["c", None] if self.con_drift else [None]
        return [((p, 1, q), t)
                for p in range(self.p_max + 1)
                for q in range(self.q_max + 1)
                for t in trends
                if (p, q) != (0, 0) or t is not None]

    def fit(self, df_train: pd.DataFrame) -> "ARIMADriftIPP":
        s = df_train.dropna(subset=["ipp"])
        y = np.log(s["ipp"].reset_index(drop=True))
        exog = _terminos_fourier(s["fecha"].reset_index(drop=True), self.fourier_k) \
            if "fecha" in s.columns else None

        candidatos = self._candidatos()

        mejor = None
        evaluados = 0
        for order, trend in candidatos:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = SARIMAX(y, exog=exog, order=order, seasonal_order=(0, 0, 0, 0),
                                trend=trend, enforce_stationarity=False,
                                enforce_invertibility=False)
                    r = m.fit(disp=False)
                crit = r.aic if self.criterio == "aic" else r.bic
                if not np.isfinite(crit):
                    continue
                evaluados += 1
                if mejor is None or crit < mejor[0]:
                    mejor = (crit, order, trend, r)
            except Exception:
                # Los órdenes que no convergen se descartan; con la rejilla completa suele
                # fallar ~25% y es esperable, no un error.
                continue

        if mejor is None:
            raise RuntimeError("ARIMADriftIPP: ningún orden candidato convergió")

        crit, order, trend, r = mejor
        self._order, self._trend, self._result = order, trend, r
        self._endog, self._exog = y, exog
        self._diagnostico = {
            "order": order, "trend": trend, "criterio": self.criterio, "valor": float(crit),
            "aic": float(r.aic), "bic": float(r.bic),
            "n_candidatos_evaluados": evaluados,
            "drift": float(r.params.get("intercept", np.nan)) if hasattr(r.params, "get") else np.nan,
        }
        logger.debug("ARIMADriftIPP: order=%s trend=%s %s=%.2f", order, trend, self.criterio, crit)
        return self

    def forecast(self, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("Llamar fit() primero")
        exog_f = None
        if self._exog is not None:
            if exog_future is not None and "fecha" in exog_future.columns:
                exog_f = _terminos_fourier(
                    exog_future["fecha"].reset_index(drop=True), self.fourier_k
                ).iloc[:horizon]
            else:
                exog_f = self._exog.iloc[:horizon]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self._result.get_forecast(steps=horizon, exog=exog_f)
        mu = pred.predicted_mean.values
        ci = pred.conf_int(alpha=0.10)
        return pd.DataFrame({
            "pred": np.exp(mu),
            "ci_lo90": np.exp(ci.iloc[:, 0].values),
            "ci_hi90": np.exp(ci.iloc[:, 1].values),
        })

    @property
    def diagnostico(self) -> dict:
        return dict(self._diagnostico)


@dataclass
class ETSDampedIPP:
    """Suavizamiento exponencial con tendencia amortiguada, sobre log(IPP).

    Se usa `statsmodels.tsa.exponential_smoothing.ets`, NO `statsforecast`: statsforecast está
    en pyproject pero **no** en requirements.txt, que es lo que instala Streamlit Cloud, así
    que meterlo como dependencia de un componente rompería el despliegue.

    La amortiguación es deliberada: proyecta la tendencia reciente pero la va apagando, lo que
    lo hace competitivo a horizontes cortos y conservador a los largos.
    """
    damped: bool = True
    _res: object = field(default=None, init=False, repr=False)
    _n: int = field(default=0, init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "ETSDampedIPP":
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        y = np.log(df_train["ipp"].dropna()).reset_index(drop=True)
        self._n = len(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = ETSModel(y, error="add", trend="add",
                                 damped_trend=self.damped, seasonal=None).fit(disp=False)
        return self

    def forecast(self, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._res is None:
            raise RuntimeError("Llamar fit() primero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self._res.get_prediction(start=self._n, end=self._n + horizon - 1)
            mu = np.asarray(pred.predicted_mean, dtype=float)
            try:
                ci = pred.summary_frame(alpha=0.10)
                lo = np.exp(np.asarray(ci["pi_lower"], dtype=float))
                hi = np.exp(np.asarray(ci["pi_upper"], dtype=float))
            except Exception:
                lo = hi = np.full(horizon, np.nan)
        return pd.DataFrame({"pred": np.exp(mu), "ci_lo90": lo, "ci_hi90": hi})


@dataclass
class LGBDlogIPP:
    """LightGBM sobre la DIFERENCIA log del IPP, con cuantiles propios.

    Reemplaza a `LGBDriversIPP`, que modelaba el **nivel** de log(IPP) con árboles. Un árbol
    no extrapola: predice promedios de hojas vistas en entrenamiento, así que con una serie
    con tendencia su pronóstico es estructuralmente plano. Combinado con los lags del IPP
    congelados en el futuro, su aporte era casi constante — coherente con el peso de 0.046 que
    le asignaba el ensemble.

    Modelando dlog el objetivo sí es aproximadamente estacionario, y el nivel se reconstruye
    integrando. Además entrena tres cuantiles (0.05/0.50/0.95), así que por primera vez este
    componente aporta su propia incertidumbre en vez de heredar la del SARIMAX.
    """
    lags_dlog: tuple[int, ...] = (1, 2, 3, 6, 12)
    cuantiles: tuple[float, float, float] = (0.05, 0.50, 0.95)
    usar_drivers: bool = True
    params: dict = field(default_factory=lambda: {
        "objective": "quantile", "n_estimators": 300, "learning_rate": 0.05,
        "num_leaves": 7, "min_child_samples": 10, "verbose": -1,
    })
    _boosters: dict = field(default_factory=dict, init=False, repr=False)
    _cols: list[str] = field(default_factory=list, init=False, repr=False)
    _hist: pd.DataFrame | None = field(default=None, init=False, repr=False)

    _DRIVERS = ("brent_yoy_lag1m", "trm_yoy_lag1m", "oni_lag", "enso_el_nino", "enso_la_nina")

    def _matriz(self, dlog: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=dlog.index)
        for lag in self.lags_dlog:
            X[f"dlog_lag{lag}"] = dlog.shift(lag)
        X["mes_cos"] = np.cos(2 * np.pi * pd.to_datetime(df["fecha"]).dt.month / 12)
        X["mes_sin"] = np.sin(2 * np.pi * pd.to_datetime(df["fecha"]).dt.month / 12)
        if self.usar_drivers:
            for c in self._DRIVERS:
                if c in df.columns:
                    X[c] = df[c].to_numpy()
        return X

    def fit(self, df_train: pd.DataFrame) -> "LGBDlogIPP":
        import lightgbm as lgb

        s = df_train.dropna(subset=["ipp"]).reset_index(drop=True)
        dlog = np.log(s["ipp"]).diff()
        X = self._matriz(dlog, s)

        # Se recortan las filas sin lags completos en vez de imputar. La version anterior
        # rellenaba con -9999, que para un arbol es un valor mas y ensucia los cortes.
        ok = X.notna().all(axis=1) & dlog.notna()
        X, y = X[ok], dlog[ok]
        self._cols = list(X.columns)
        self._hist = s

        if len(X) < 30:
            raise RuntimeError(f"LGBDlogIPP: solo {len(X)} filas utiles tras recortar")

        self._boosters = {}
        for q in self.cuantiles:
            p = dict(self.params)
            p["alpha"] = q
            self._boosters[q] = lgb.LGBMRegressor(**p).fit(X, y)
        return self

    def forecast(self, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self._boosters:
            raise RuntimeError("Llamar fit() primero")

        s = self._hist
        dlog_hist = list(np.log(s["ipp"]).diff().dropna().to_numpy())
        nivel = float(s["ipp"].iloc[-1])

        fechas = (pd.to_datetime(exog_future["fecha"]).reset_index(drop=True)
                  if exog_future is not None and "fecha" in exog_future.columns
                  else pd.Series(pd.date_range(pd.to_datetime(s["fecha"].iloc[-1]),
                                               periods=horizon + 1, freq="MS")[1:]))

        q_lo, q_med, q_hi = self.cuantiles
        preds, los, his = [], [], []
        acum_lo = acum_hi = 0.0

        for t in range(horizon):
            fila = {}
            for lag in self.lags_dlog:
                fila[f"dlog_lag{lag}"] = dlog_hist[-lag] if len(dlog_hist) >= lag else 0.0
            mes = fechas.iloc[t].month if t < len(fechas) else 1
            fila["mes_cos"] = np.cos(2 * np.pi * mes / 12)
            fila["mes_sin"] = np.sin(2 * np.pi * mes / 12)
            if self.usar_drivers and exog_future is not None:
                for c in self._DRIVERS:
                    if c in self._cols:
                        fila[c] = (float(exog_future[c].iloc[min(t, len(exog_future) - 1)])
                                   if c in exog_future.columns else 0.0)
            X1 = pd.DataFrame([fila])[self._cols]

            d_med = float(self._boosters[q_med].predict(X1)[0])
            dlog_hist.append(d_med)
            nivel *= np.exp(d_med)
            preds.append(nivel)

            # La banda acumula la dispersión de los cuantiles paso a paso: la incertidumbre de
            # un pronóstico integrado crece con el horizonte, no es la de un solo paso.
            acum_lo += float(self._boosters[q_lo].predict(X1)[0]) - d_med
            acum_hi += float(self._boosters[q_hi].predict(X1)[0]) - d_med
            los.append(nivel * np.exp(acum_lo))
            his.append(nivel * np.exp(acum_hi))

        return pd.DataFrame({"pred": preds, "ci_lo90": los, "ci_hi90": his})

    # Alias de compatibilidad: `EnsembleIPP._calibrar_pesos` llama `.predict(df_val)`.
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.forecast(len(df), exog_future=df)["pred"].to_numpy()
