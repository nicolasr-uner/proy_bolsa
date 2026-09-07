"""Modelo de IPP mensual (Indice de Precios al Productor, Colombia).

Pipeline:
  1. SARIMA univariado (baseline robusto)
  2. SARIMAX con drivers ortogonales (brent_yoy, trm_yoy con rezagos, ENSO, estacionalidad)
  3. VECM si hay cointegración Johansen entre IPP y brent_cop (sistema 2-variable)
  4. LightGBM sobre features tabulares
  5. Ensemble por inverse-MSE con shrinkage sobre los errores del rolling-origin, por horizonte

El IPP colombiano tiene:
  - Estacionalidad débil (la manufactura no es tan estacional como el precio de bolsa)
  - Cointegración de largo plazo con el costo del insumo importado en pesos (literatura BanRep)
  - Rezago típico de 1-3 meses (transmisión importaciones -> precios internos)

Drivers (post-estudio; ver docs/estudio_drivers_ipp.md y docs/METODOLOGIA.md):
  brent_cop = Brent_USD x TRM (costo importado en pesos, VIF~=1.04) y las variaciones
  anuales brent_yoy / trm_yoy. PPI USA se descartó por multicolinealidad (VIF=7.65).

Nota: Requiere ipp > 0 (índice base=100, DANE dic-2014). El modelo trabaja en
diferencias logarítmicas (log_dif) para estacionariedad y el pronóstico se
reconstruye por integración.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from proybolsa.models.ipp.componentes import ARIMADriftIPP
from proybolsa.models.sarimax_pickle import PickleCompactoSARIMAX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Drivers ortogonales post-estudio (ver docs/estudio_drivers_ipp.md)
# brent_cop = Brent_USD × TRM elimina multicolinealidad (PPI_USA VIF=7.65 → descartado)
# brent_yoy / trm_yoy en % anual escalan bien con dlog(IPP): coef interpretable como elasticidad parcial
# (brent_cop en nivel escala mal con dlog IPP: coef→0, sin efecto en escenarios)
_EXOG_SARIMAX = ["brent_yoy_lag1m", "trm_yoy_lag1m", "enso_el_nino", "cos_mes", "sin_mes"]

_FEATS_LGB = [
    "brent_cop", "brent_cop_lag1m", "brent_cop_lag2m", "brent_cop_lag3m",
    "trm_yoy", "trm_yoy_lag1m",
    "brent_yoy", "brent_yoy_lag1m",
    "ipp_lag1m", "ipp_lag2m", "ipp_lag3m", "ipp_lag12m",
    "enso_el_nino", "enso_la_nina", "oni_lag",
    "mes", "cos_mes", "sin_mes",
]

_VECM_VARS = ["ipp", "brent_cop"]  # sistema 2-variable: más estable con n≈137

# Errores del rolling-origin de donde salen los pesos por horizonte. Lo produce
# scripts/ejecutar_backtest_ipp.py y se versiona, para que Streamlit Cloud —que no puede
# correr un backtest— use los mismos pesos que se calibraron en local.
_RUTA_BACKTEST_IPP = Path(__file__).resolve().parents[3].parent / "outputs" / "backtest" / "errores_ipp.parquet"


def _cargar_errores_backtest(ruta: str | Path | None) -> pd.DataFrame | None:
    """Lee los errores del backtest si existen. Devuelve None si no, sin ruido."""
    if ruta is None:
        return None
    p = Path(ruta)
    if not p.exists():
        logger.info("Sin backtest en %s: los pesos usaran el metodo antiguo", p.name)
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        logger.warning("No se pudo leer %s (%s)", p.name, exc)
        return None

# Columnas sin las cuales los componentes con drivers (SARIMAX, VECM, LGB) no son
# estimables. `construir_features.py` mergea el IPP con los macros en how="outer", así que
# el parquet trae la historia completa del IPP (1999-06+) con drivers NaN antes de 2015-01.
# Sin recortar, el SARIMAX imputaría 0 en las YoY y el LGB -9999 sobre ~200 meses: inventar
# señal en vez de reconocer que falta.
_DRIVERS_REQUERIDOS = ["brent_cop"]


def recortar_a_drivers(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Recorta el DataFrame al tramo final donde los drivers requeridos existen.

    Los modelos univariados (SARIMA/ARIMA, ETS) pueden aprovechar toda la historia; los que
    consumen drivers, no. Esta función define la muestra común del ensemble actual.
    """
    cols = [c for c in (cols or _DRIVERS_REQUERIDOS) if c in df.columns]
    if not cols:
        return df
    mask = df[cols].notna().all(axis=1)
    if not mask.any():
        logger.warning("IPP: ningún mes tiene %s; no se recorta la muestra", cols)
        return df
    primera = df.loc[mask, "fecha"].min()
    recortado = df[df["fecha"] >= primera].reset_index(drop=True)
    if len(recortado) < len(df):
        logger.info(
            "IPP: muestra recortada a drivers disponibles (%d -> %d meses, desde %s)",
            len(df), len(recortado), pd.to_datetime(primera).date(),
        )
    return recortado


# ---------------------------------------------------------------------------
# Test de cointegración
# ---------------------------------------------------------------------------

def johansen_cointegracion(df: pd.DataFrame, max_lags: int = 2) -> dict:
    """Aplica el test de Johansen al sistema {IPP, brent_cop}.

    Si brent_cop no está en df pero sí brent y trm, lo computa internamente.

    Returns
    -------
    dict con:
        n_cointegrating_vectors : número de vectores de cointegración detectados
        trace_stats             : estadísticos traza
        p_valor_aprox           : '< 0.05' o '>= 0.05' (Johansen no da p-valores exactos)
    """
    df = df.copy()
    if "brent_cop" not in df.columns:
        if "brent" in df.columns and "trm" in df.columns:
            df["brent_cop"] = df["brent"] * df["trm"]
        else:
            raise ValueError("Faltan columnas para Johansen: necesita brent_cop (o brent+trm) e ipp")

    required = ["ipp", "brent_cop"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para Johansen: {missing}")

    data = df[required].dropna()
    if len(data) < 20:
        return {"n_cointegrating_vectors": 0, "cointegran": False, "nota": "datos insuficientes (<20 obs)"}

    data_log = np.log(data.astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = coint_johansen(data_log.values, det_order=0, k_ar_diff=max_lags)

    trace_stat = res.lr1     # estadístico traza
    crit_95 = res.cvt[:, 1]  # valores criticos al 95%
    n_coint = int(np.sum(trace_stat > crit_95))

    return {
        "n_cointegrating_vectors": n_coint,
        "trace_stats": trace_stat.tolist(),
        "crit_95": crit_95.tolist(),
        "cointegran": n_coint > 0,
    }


# ---------------------------------------------------------------------------
# Modelo SARIMA (baseline univariado)
# ---------------------------------------------------------------------------

@dataclass
class SARIMABaselineIPP(PickleCompactoSARIMAX):
    """SARIMA univariado en diferencias log del IPP.

    Trabaja en log con d=1 (el IPP es I(1): ADF p=0.90 en log-nivel, p<0.001 en log-dif).

    ATENCIÓN — sin término de drift. El orden está fijo en (1,1,1) sin `trend`, lo que hace
    que el pronóstico converja a una recta plana: proyecta +0.4% a 12 meses contra una deriva
    histórica de +5.1%/año. Medido: RMSE a h=24 de 50.63 contra 29.70 de un random walk, es
    decir peor que no hacer nada. La Fase 2 del plan lo reemplaza por `ARIMADriftIPP` con
    selección de orden por AIC e `trend="c"` entre los candidatos.
    """
    _result: object = field(default=None, init=False, repr=False)
    _ipp_init: float = field(default=100.0, init=False, repr=False)  # ultimo nivel para reconstruir
    # Guardados para el pickle compacto (ver models/sarimax_pickle.py)
    _endog: object = field(default=None, init=False, repr=False)
    _exog: object = field(default=None, init=False, repr=False)
    _order: tuple = field(default=(1, 1, 1), init=False, repr=False)
    _seasonal_order: tuple = field(default=(0, 0, 0, 0), init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "SARIMABaselineIPP":
        ipp = df_train["ipp"].dropna()
        self._ipp_init = float(ipp.iloc[-1])
        log_ipp = np.log(ipp)
        # ARIMA(1,1,1) sin componente estacional (IPP mensual tiene ciclo debil)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(log_ipp, order=self._order, seasonal_order=self._seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            self._result = model.fit(disp=False)
        self._endog, self._exog = log_ipp, None
        logger.debug("SARIMA IPP ajustado. AIC=%.1f", self._result.aic)
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("Llamar fit() primero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self._result.get_forecast(steps=horizon)

        mu_log = pred.predicted_mean.values
        ci_log = pred.conf_int(alpha=0.10)

        # Reconstruir nivel: exp(log_ipp_T + cumsum(log_dif))
        # El modelo predice log(IPP_t), reconstruimos nivel
        pred_nivel = np.exp(mu_log)
        ci_lo = np.exp(ci_log.iloc[:, 0].values)
        ci_hi = np.exp(ci_log.iloc[:, 1].values)

        return pd.DataFrame({
            "pred": pred_nivel,
            "ci_lo90": ci_lo,
            "ci_hi90": ci_hi,
        })


# ---------------------------------------------------------------------------
# Modelo SARIMAX (con drivers macro)
# ---------------------------------------------------------------------------

@dataclass
class SARIMAXDriversIPP(PickleCompactoSARIMAX):
    """SARIMAX del IPP con drivers macro rezagados como variables exógenas.

    Las exógenas reales son `_EXOG_SARIMAX` (YoY de Brent y TRM, dummy El Niño y los
    términos trigonométricos del mes). `ppi_usa` NO se consume: se descartó por
    multicolinealidad, aunque el VIF se midió en niveles (ver plan, Fase 2).

    Dos limitaciones medidas, ambas atacadas en la Fase 2 del plan:
      - Sin drift, igual que `SARIMABaselineIPP`: a h=24 el RMSE es 30.06 contra 24.43 de la
        misma especificación con `trend="c"`.
      - Los drivers no aportan precisión out-of-sample: incluso con foresight perfecto de las
        exógenas no le gana a un ARIMA univariado en ningún horizonte. Su valor real es
        articular escenarios (mover TRM/Brent y ver la respuesta), no reducir el error.
    """
    _result: object = field(default=None, init=False, repr=False)
    _exog_cols: list = field(default_factory=list, init=False, repr=False)
    # Guardados para el pickle compacto (ver models/sarimax_pickle.py)
    _endog: object = field(default=None, init=False, repr=False)
    _exog: object = field(default=None, init=False, repr=False)
    _order: tuple = field(default=(1, 1, 1), init=False, repr=False)
    _seasonal_order: tuple = field(default=(0, 0, 0, 0), init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "SARIMAXDriversIPP":
        df = df_train.copy()
        ipp = df["ipp"].dropna()

        available_exog = [c for c in _EXOG_SARIMAX if c in df.columns]
        exog = df.loc[ipp.index, available_exog].ffill().fillna(0) if available_exog else None
        self._exog_cols = available_exog

        log_ipp = np.log(ipp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(log_ipp, exog=exog,
                            order=self._order, seasonal_order=self._seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            self._result = model.fit(disp=False)
        self._endog, self._exog = log_ipp, exog
        logger.debug("SARIMAX IPP ajustado. AIC=%.1f", self._result.aic)
        return self

    def forecast(self, horizon: int, exog_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("Llamar fit() primero")

        exog_f = None
        if exog_future is not None and self._exog_cols:
            avail = [c for c in self._exog_cols if c in exog_future.columns]
            exog_f = exog_future[avail].ffill().fillna(0) if avail else None

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


# ---------------------------------------------------------------------------
# Modelo VECM (si hay cointegración)
# ---------------------------------------------------------------------------

@dataclass
class VECMDriversIPP:
    """VECM del sistema {log IPP, log brent_cop}, activado solo si la cointegración se sostiene.

    La versión anterior decía "solo se usa si el test Johansen encuentra cointegracion" y hacía
    exactamente lo contrario: `n_coint = max(1, test["n_cointegrating_vectors"])` imponía rango 1
    aunque el test devolviera 0, y `self._available = True` se fijaba incondicionalmente. El
    resultado del test se calculaba y se tiraba. Encima el test corría con `k_ar_diff=2` y la
    estimación con `k_ar_diff=1`.

    Importa porque no es cosmético: sobre los datos reales la traza rechaza r=0 con k=1 y k=2
    pero NO con k=3. La cointegración es frágil y depende del rezago, así que activarla siempre
    le entregaba a este componente un peso alto (0.77 en la última corrida) apoyado en un test
    que a veces dice lo contrario.

    Condiciones para activar, todas obligatorias:
      1. k elegido por AIC del VAR en niveles (y se usa EL MISMO en test y estimación).
      2. La traza rechaza r=0 al 95% con ese k.
      3. También rechaza al 90% con k±1 (robustez al rezago).
      4. Al menos `min_obs` observaciones.
      5. El coeficiente de ajuste alpha de la ecuación del IPP es negativo y significativo:
         sin corrección de error con el signo correcto, un VECM no es un VECM.

    Si falla cualquiera, `_available=False` y `motivo` explica cuál, para que el dashboard y el
    resumen puedan decirlo en vez de mostrar un peso sin justificación.
    """
    k_ar_diff: int | str = "auto"
    min_obs: int = 80
    exigir_robustez_k: bool = True
    exigir_alpha_negativo: bool = True
    _model: object = field(default=None, init=False, repr=False)
    _n_coint: int = field(default=0, init=False, repr=False)
    _available: bool = field(default=False, init=False, repr=False)
    _k_usado: int = field(default=1, init=False, repr=False)
    _motivo: str = field(default="sin ajustar", init=False, repr=False)
    # Distingue "se pudo estimar" de "es valido para el pronostico oficial". Solo el modo
    # escenario usa un VECM estimable pero no validado.
    _estimable: bool = field(default=False, init=False, repr=False)

    @property
    def motivo(self) -> str:
        return self._motivo

    def _desactivar(self, motivo: str, *, estimar_igual: bool = False,
                    data_log=None, k: int = 1) -> "VECMDriversIPP":
        """Marca el componente como no válido para el pronóstico oficial.

        `estimar_igual=True` ajusta el modelo de todos modos y lo deja accesible **solo** para
        el modo escenario. Motivo: el VECM es el único componente con un canal real hacia los
        drivers (spread medido a 24 meses ante TRM/Brent ±20%: VECM +39.92, SARIMAX +0.87,
        ARIMA 0.00), así que apagarlo del todo deja los sliders del dashboard inertes. Se
        conserva para responder "cuánto se movería", no para decir "cuánto va a valer".
        """
        self._available = False
        self._n_coint = 0
        self._motivo = motivo
        if estimar_igual and data_log is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._model = VECM(data_log.values, k_ar_diff=k, coint_rank=1,
                                       deterministic="ci").fit()
                self._estimable = True
                self._k_usado = k
            except Exception as exc:
                logger.debug("VECM: tampoco se pudo estimar para escenarios (%s)", exc)
                self._model = None
                self._estimable = False
        else:
            self._model = None
            self._estimable = False
        logger.info("VECM IPP no validado para el pronostico: %s%s", motivo,
                    " (se conserva para escenarios)" if self._estimable else "")
        return self

    def fit(self, df_train: pd.DataFrame) -> "VECMDriversIPP":
        from proybolsa.models.ipp.diagnosticos import (
            decidir_cointegracion,
            johansen_reporte,
            seleccionar_k_ar_diff,
        )

        df = df_train.copy()
        if "brent_cop" not in df.columns:
            if "brent" in df.columns and "trm" in df.columns:
                df["brent_cop"] = df["brent"] * df["trm"]
            else:
                return self._desactivar("falta brent_cop (o brent+trm)")

        if "ipp" not in df.columns:
            return self._desactivar("falta la columna ipp")

        data = df[["ipp", "brent_cop"]].dropna()
        if len(data) < self.min_obs:
            return self._desactivar(f"muestra insuficiente ({len(data)} < {self.min_obs} obs)")

        data_log = np.log(data.astype(float))

        # 1. Rezago por AIC del VAR en niveles.
        if self.k_ar_diff == "auto":
            try:
                k = seleccionar_k_ar_diff(data_log)
            except Exception as exc:
                logger.warning("VECM: selección de k falló (%s); se usa k=1", exc)
                k = 1
        else:
            k = int(self.k_ar_diff)

        # 2-3. Johansen con k y sus vecinos, y la regla de decisión.
        grid = tuple(sorted({max(1, k - 1), k, k + 1})) if self.exigir_robustez_k else (k,)
        reporte = johansen_reporte(data, ("ipp", "brent_cop"), k_ar_diff_grid=grid)
        decision = decidir_cointegracion(reporte, k, len(data), min_obs=self.min_obs)
        if not decision["activar"]:
            return self._desactivar(decision["motivo"], estimar_igual=True,
                                    data_log=data_log, k=k)

        # 4. Estimar con EL MISMO k que se testeó.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = VECM(data_log.values, k_ar_diff=k,
                          coint_rank=decision["rango"], deterministic="ci")
            ajustado = modelo.fit()

        # 5. El término de corrección de error debe tirar del IPP hacia el equilibrio.
        if self.exigir_alpha_negativo:
            try:
                alpha = float(np.asarray(ajustado.alpha)[0, 0])
                p_alpha = float(np.asarray(ajustado.pvalues_alpha)[0, 0])
            except Exception:
                alpha, p_alpha = np.nan, np.nan
            if not (np.isfinite(alpha) and alpha < 0 and np.isfinite(p_alpha) and p_alpha < 0.10):
                return self._desactivar(
                    f"el ajuste al equilibrio no tiene el signo correcto o no es significativo "
                    f"(alpha={alpha:.4f}, p={p_alpha:.3f}); sin correccion de error no es un VECM",
                    estimar_igual=True, data_log=data_log, k=k,
                )

        self._model = ajustado
        self._n_coint = decision["rango"]
        self._k_usado = k
        self._available = True
        self._estimable = True
        self._motivo = f"activado: {decision['motivo']}"
        logger.info("VECM IPP activado (k=%d, rango=%d)", k, decision["rango"])
        return self

    def forecast(self, horizon: int, brent_cop_future=None, *,
                 permitir_no_validado: bool = False) -> pd.DataFrame | None:
        """Pronóstico del VECM, o None si el componente no es utilizable.

        `permitir_no_validado=True` lo usa **solo** el modo escenario: devuelve el forecast de
        un VECM que se pudo estimar pero cuya cointegración no pasó la regla de activación.
        Sirve para medir sensibilidad a los drivers, no para el pronóstico oficial.
        """
        utilizable = self._available or (permitir_no_validado and self._estimable)
        if not utilizable or self._model is None:
            return None

        if brent_cop_future is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = self._model.predict(steps=horizon)
            ipp_log_pred = fc[:, 0]
        else:
            ipp_log_pred = self._forecast_condicional(horizon, brent_cop_future)

        pred = np.exp(ipp_log_pred)
        resid_std = float(np.std(self._model.resid[:, 0]))
        horizon_std = resid_std * np.sqrt(np.arange(1, horizon + 1))
        ci_lo = np.exp(ipp_log_pred - 1.645 * horizon_std)
        ci_hi = np.exp(ipp_log_pred + 1.645 * horizon_std)
        return pd.DataFrame({"pred": pred, "ci_lo90": ci_lo, "ci_hi90": ci_hi})

    def _forecast_condicional(self, horizon: int, brent_cop_future) -> np.ndarray:
        """Itera la representacion VAR del VECM imponiendo el path de brent_cop.

        Usa var_rep (el mismo que usa predict() internamente) para garantizar
        coherencia numérica exacta con el forecast incondicional. En cada paso,
        tras computar y_new con la dinámica VAR completa, sobreescribe el
        componente de brent_cop antes de actualizar el estado.
        """
        brent_cop_log = np.log(np.asarray(brent_cop_future, dtype=float)[:horizon])

        var_rep = self._model.var_rep        # (k_ar, k, k): coefs VAR por lag
        k_ar = self._model.k_ar             # lags en representación VAR = k_ar_diff + 1

        # Término de tendencia constante (contribución del "ci" a cada paso)
        trend = np.zeros(var_rep.shape[1], dtype=float)
        if "ci" in self._model.deterministic:
            trend = self._model.alpha.dot(self._model.const_coint.T).T[0]

        # Ventana de k_ar observaciones más recientes (shape: k_ar × k)
        last_obs = self._model.y_all.T[-k_ar:].copy().astype(float)

        ipp_log_pred = np.zeros(horizon)

        for t in range(horizon):
            # y_new = sum(var_rep[lag] @ y_{t-lag-1}) + trend
            y_new = trend.copy()
            for lag in range(k_ar):
                y_new = y_new + var_rep[lag] @ last_obs[-(lag + 1)]

            # Imponer el path de brent_cop del escenario (índice 1)
            y_new[1] = brent_cop_log[t]

            ipp_log_pred[t] = y_new[0]

            # Deslizar la ventana: descartar la más antigua, añadir y_new
            last_obs = np.vstack([last_obs[1:], y_new])

        return ipp_log_pred


# ---------------------------------------------------------------------------
# Modelo LightGBM
# ---------------------------------------------------------------------------

@dataclass
class LGBDriversIPP:
    """LightGBM sobre features macro mensuales para IPP."""
    _model: lgb.Booster | None = field(default=None, init=False, repr=False)
    _feature_cols: list = field(default_factory=list, init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "LGBDriversIPP":
        df = df_train.dropna(subset=["ipp"]).copy()
        feature_cols = [c for c in _FEATS_LGB if c in df.columns]
        self._feature_cols = feature_cols

        X = df[feature_cols].fillna(-9999)
        y = np.log(df["ipp"])

        params = {
            "objective": "regression", "metric": "rmse",
            "n_estimators": 200, "learning_rate": 0.05,
            "num_leaves": 15, "min_child_samples": 5,
            "verbose": -1,
        }
        dtrain = lgb.Dataset(X, label=y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = lgb.train(
                params, dtrain, num_boost_round=200,
                valid_sets=[dtrain],
                callbacks=[lgb.log_evaluation(period=-1)],
            )
        return self

    def predict(self, df_pred: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Llamar fit() primero")
        avail = [c for c in self._feature_cols if c in df_pred.columns]
        X = df_pred[avail].fillna(-9999)
        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = -9999
        return np.exp(self._model.predict(X[self._feature_cols]))


# ---------------------------------------------------------------------------
# Ensemble IPP
# ---------------------------------------------------------------------------

HORIZONTE_IPP = Literal["corto", "medio", "largo"]


@dataclass
class EnsembleIPP:
    """Ensemble ARIMA-drift + SARIMAX + VECM + LGB para IPP mensual.

    Pesos calibrados por inverse-MSE sobre período de validación.
    Si VECM no está disponible (sin cointegración), usa solo ARIMA+SARIMAX+LGB.

    El componente univariado es `ARIMADriftIPP`, no el viejo `SARIMABaselineIPP`. El campo se
    sigue llamando `sarima` y su peso `w_sarima` por compatibilidad: esos nombres viajan en
    `resumen_ipp.json`, los lee el dashboard y los asume `resumen_modelo()`. Renombrarlos es
    un cambio aparte.

    Medido sobre 56 orígenes (RMSE, muestra 2015+, drivers congelados):
        SARIMABaselineIPP (1,1,1) sin drift   h6 8.16   h12 18.46   h24 49.72
        ARIMADriftIPP     (1,1,0) con drift   h6 7.08   h12 13.50   h24 24.11
    """
    sarima: ARIMADriftIPP = field(default_factory=ARIMADriftIPP)
    sarimax: SARIMAXDriversIPP = field(default_factory=SARIMAXDriversIPP)
    vecm: VECMDriversIPP = field(default_factory=VECMDriversIPP)
    lgb: LGBDriversIPP = field(default_factory=LGBDriversIPP)
    w_sarima: float = 0.25
    w_sarimax: float = 0.25
    w_vecm: float = 0.25
    w_lgb: float = 0.25
    sesgo_por_horizonte: dict = field(default_factory=dict)

    # Pesos por horizonte, calibrados desde el backtest rolling-origin.
    # {h: {"sarima": w, "sarimax": w, "vecm": w, "lgb": w}}
    pesos_por_horizonte: dict = field(default_factory=dict)
    # Encogimiento hacia pesos iguales: w = (1-a)*w_invmse + a*(1/4), con a = lambda/(n+lambda).
    # Con ~56 orígenes y h=24 hay apenas ~2 bloques independientes de 24 meses, así que el MSE
    # ahí es casi ruido y el inverse-MSE crudo produce pesos absurdos.
    lambda_shrink: float = 10.0
    piso_peso: float = 0.02

    # Bandas calibradas empíricamente. Viaja dentro del pickle, así que Streamlit Cloud usa la
    # calibración hecha en local sin tener que correr un backtest.
    calibrador: object = None
    # Claves del backtest de donde se calibra la banda, en orden de preferencia: primero el
    # propio ensemble, y si no está, el componente dominante.
    _CLAVES_CALIBRACION = ("ipp_ensemble", "ipp_arima_drift")

    _CLAVES_BACKTEST = {
        "sarima": "ipp_arima_drift",
        "sarimax": "ipp_sarimax_actual",
        "vecm": "ipp_vecm",
        "lgb": "ipp_lgb_actual",
    }

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame | None = None,
        df_full: pd.DataFrame | None = None,
        errores_backtest: pd.DataFrame | None = None,
    ) -> "EnsembleIPP":
        """Ajusta los 4 componentes y calibra pesos.

        Los componentes se ajustan sobre df_train y los pesos se calibran sobre df_val
        (validacion honesta). Si se pasa df_full, los componentes se RE-AJUSTAN sobre
        toda la serie para el pronostico de produccion: de lo contrario el forecast
        partiria del fin de df_train (stale) en vez del ultimo dato observado.
        """
        logger.info("Ajustando ARIMA-drift IPP...")
        self.sarima.fit(df_train)
        logger.info("Ajustando SARIMAX IPP...")
        self.sarimax.fit(df_train)
        logger.info("Ajustando VECM IPP...")
        self.vecm.fit(df_train)
        logger.info("Ajustando LGB IPP...")
        self.lgb.fit(df_train)

        # Preferencia de calibración: el backtest rolling-origin por encima de `_calibrar_pesos`.
        # El segundo mezcla horizontes y le da foresight perfecto al SARIMAX (ver el docstring
        # de calibrar_pesos_desde_backtest), así que solo se usa si no hay backtest disponible.
        if errores_backtest is not None and not errores_backtest.empty:
            self.calibrar_pesos_desde_backtest(errores_backtest)
            self.calibrar_intervalos_desde_backtest(errores_backtest)
        elif df_val is not None and len(df_val) >= 3:
            logger.warning(
                "Sin backtest disponible: se calibran los pesos con el metodo antiguo "
                "(mezcla horizontes y usa foresight perfecto). Corra "
                "scripts/ejecutar_backtest_ipp.py para calibrar sobre el rolling-origin."
            )
            self._calibrar_pesos(df_val)
        else:
            self._normalizar_pesos()

        # Re-ajuste sobre la serie completa (pesos ya calibrados se conservan).
        if df_full is not None and len(df_full) > len(df_train):
            logger.info("Re-ajustando componentes IPP sobre la serie completa (%d obs)...", len(df_full))
            self.sarima.fit(df_full)
            self.sarimax.fit(df_full)
            self.vecm.fit(df_full)
            self.lgb.fit(df_full)
        return self

    def _normalizar_pesos(self) -> None:
        if not self.vecm._available:
            self.w_sarima = 0.3
            self.w_sarimax = 0.4
            self.w_vecm = 0.0
            self.w_lgb = 0.3
        logger.info("Pesos IPP: SARIMA=%.2f SARIMAX=%.2f VECM=%.2f LGB=%.2f",
                    self.w_sarima, self.w_sarimax, self.w_vecm, self.w_lgb)

    def calibrar_pesos_desde_backtest(self, errores: pd.DataFrame,
                                      horizontes: tuple[int, ...] = (1, 3, 6, 12, 24)
                                      ) -> "EnsembleIPP":
        """Calibra pesos inverse-MSE **por horizonte** desde el backtest rolling-origin.

        Reemplaza a `_calibrar_pesos`, que era insalvable por tres motivos:

        1. Comparaba horizontes distintos. SARIMA/SARIMAX/VECM producían un forecast de ~21
           pasos desde el fin de train, mientras `self.lgb.predict(df_val)` predecía ~1 paso
           alimentado con los `ipp_lag1m/2m/3m` REALES de cada mes de validación. Los MSE no
           eran comparables.
        2. Le pasaba `exog_future=df_val` al SARIMAX: foresight perfecto de los drivers, que en
           producción no existe.
        3. Un solo origen y pesos constantes para todo h entre 1 y 24, cuando el mejor modelo
           a 1 mes y a 24 meses no tiene por qué ser el mismo.

        Aquí cada componente se evaluó por separado en multi-paso real con drivers congelados,
        sobre los mismos orígenes, y los pesos salen de esos errores por horizonte.
        """
        req = {"modelo", "horizonte", "error"}
        if errores is None or errores.empty or not req <= set(errores.columns):
            logger.warning("Backtest no utilizable para calibrar pesos (faltan %s)",
                           req - set(errores.columns if errores is not None else []))
            return self

        disponibles = set(errores["modelo"].unique())
        faltan = {k: v for k, v in self._CLAVES_BACKTEST.items() if v not in disponibles}
        if faltan:
            logger.warning("Sin backtest para %s; esos componentes van con peso igual",
                           sorted(faltan.values()))

        nuevos: dict[int, dict[str, float]] = {}
        for h in horizontes:
            eh = errores[errores["horizonte"] == h]
            if eh.empty:
                continue
            mses: dict[str, float] = {}
            for comp, clave in self._CLAVES_BACKTEST.items():
                e = eh.loc[eh["modelo"] == clave, "error"].dropna()
                if len(e) >= 5:
                    mses[comp] = float(np.mean(e.to_numpy() ** 2))
            if len(mses) < 2:
                continue

            n_orig = int(eh.groupby("modelo").size().max())
            inv = {k: 1.0 / (v + 1e-9) for k, v in mses.items()}
            total = sum(inv.values())
            w = {k: v / total for k, v in inv.items()}

            # Encogimiento hacia pesos iguales.
            a = self.lambda_shrink / (n_orig + self.lambda_shrink)
            igual = 1.0 / len(w)
            w = {k: (1 - a) * v + a * igual for k, v in w.items()}

            # Los componentes sin backtest entran con el piso, no con cero: no se midieron,
            # que es distinto de haberse medido mal.
            #
            # El piso se fija DESPUÉS de normalizar y el resto se reparte sobre lo que sobra.
            # Hacerlo al revés (max(v, piso) y luego dividir por la suma) deja a los
            # componentes del piso por debajo del piso, porque la suma pasa de 1.
            for comp in self._CLAVES_BACKTEST:
                w.setdefault(comp, 0.0)
            en_piso = {k for k, v in w.items() if v < self.piso_peso}
            libres = {k: v for k, v in w.items() if k not in en_piso}
            disponible = 1.0 - self.piso_peso * len(en_piso)
            s = sum(libres.values())

            final = {k: self.piso_peso for k in en_piso}
            if libres and s > 0:
                final.update({k: v / s * disponible for k, v in libres.items()})
            elif libres:
                final.update({k: disponible / len(libres) for k in libres})
            nuevos[int(h)] = final

        if not nuevos:
            logger.warning("El backtest no produjo pesos utilizables")
            return self

        self.pesos_por_horizonte = nuevos
        # Los escalares reflejan el horizonte de referencia (12m) para `resumen_modelo()`,
        # `resumen_ipp.json` y el dashboard, que esperan un único número por componente.
        ref = nuevos.get(12) or nuevos[sorted(nuevos)[len(nuevos) // 2]]
        self.w_sarima, self.w_sarimax = ref["sarima"], ref["sarimax"]
        self.w_vecm, self.w_lgb = ref["vecm"], ref["lgb"]
        logger.info("Pesos IPP por horizonte calibrados desde backtest: %s",
                    {h: {k: round(v, 3) for k, v in w.items()} for h, w in sorted(nuevos.items())})
        return self

    # Componentes con un canal real hacia los drivers macro. El ARIMA es univariado: su
    # respuesta a TRM/Brent es exactamente cero por construcción.
    _COMPONENTES_CON_DRIVERS = ("sarimax", "vecm", "lgb")

    def calibrar_intervalos_desde_backtest(self, errores: pd.DataFrame) -> "EnsembleIPP":
        """Calibra las bandas con los cuantiles empíricos del error del rolling-origin.

        Reemplaza el trasplante del ancho de un componente. Las bandas nominales del 90%
        cubrían entre 44% y 75% según horizonte: no eran intervalos del 90%.
        """
        from proybolsa.models.ipp.incertidumbre import CalibradorIntervalos

        disponibles = set(errores.get("modelo", pd.Series(dtype=str)).unique())
        for clave in self._CLAVES_CALIBRACION:
            if clave not in disponibles:
                continue
            cal = CalibradorIntervalos.desde_errores(errores, clave)
            if cal.disponible:
                self.calibrador = cal
                if clave != self._CLAVES_CALIBRACION[0]:
                    logger.info("Intervalos calibrados con %s (el ensemble no estaba en el "
                                "backtest); las bandas seran ligeramente optimistas", clave)
                return self
        logger.warning("No se pudo calibrar los intervalos: se usara el ancho nominal escalado")
        return self

    def pesos_escenario(self) -> dict[str, float]:
        """Pesos del modo escenario: reparto igual entre los componentes con drivers.

        Se usa reparto igual y no los pesos de precisión porque estos últimos dejan al VECM en
        el piso (0.019), y el VECM es el que aporta casi toda la sensibilidad. El objetivo de
        este modo no es minimizar error sino que el slider comunique una elasticidad.
        """
        vivos = [c for c in self._COMPONENTES_CON_DRIVERS
                 if c != "vecm" or self.vecm._estimable or self.vecm._available]
        if not vivos:
            return self.pesos(12)
        w = {c: 1.0 / len(vivos) for c in vivos}
        w.setdefault("sarima", 0.0)
        for c in ("sarimax", "vecm", "lgb"):
            w.setdefault(c, 0.0)
        return w

    def pesos(self, h: int) -> dict[str, float]:
        """Pesos vigentes para el horizonte `h`, interpolando entre los calibrados."""
        if not self.pesos_por_horizonte:
            return {"sarima": self.w_sarima, "sarimax": self.w_sarimax,
                    "vecm": self.w_vecm, "lgb": self.w_lgb}

        hs = sorted(self.pesos_por_horizonte)
        if h <= hs[0]:
            base = dict(self.pesos_por_horizonte[hs[0]])
        elif h >= hs[-1]:
            base = dict(self.pesos_por_horizonte[hs[-1]])
        else:
            hi = next(x for x in hs if x >= h)
            lo = max(x for x in hs if x <= h)
            if lo == hi:
                base = dict(self.pesos_por_horizonte[lo])
            else:
                t = (h - lo) / (hi - lo)
                a, b = self.pesos_por_horizonte[lo], self.pesos_por_horizonte[hi]
                base = {k: (1 - t) * a[k] + t * b[k] for k in a}

        # El VECM puede haberse desactivado después de calibrar: su peso se redistribuye.
        # Solo si ya se INTENTÓ ajustarlo: un ensemble recién construido tiene _available=False
        # porque nadie lo ha ajustado todavía, y ahí redistribuir sería inventar.
        ya_evaluado = getattr(self.vecm, "_motivo", "sin ajustar") != "sin ajustar"
        if ya_evaluado and not self.vecm._available and base.get("vecm", 0) > 0:
            base["vecm"] = 0.0
        s = sum(base.values())
        return {k: v / s for k, v in base.items()} if s > 0 else base

    def _calibrar_pesos(self, df_val: pd.DataFrame) -> None:
        y_real = df_val["ipp"].dropna().values
        n = len(y_real)

        preds = {}
        preds["sarima"]  = self.sarima.forecast(n)["pred"].values
        preds["sarimax"] = self.sarimax.forecast(n, exog_future=df_val)["pred"].values
        vecm_fc = self.vecm.forecast(n)
        preds["vecm"] = vecm_fc["pred"].values if vecm_fc is not None else preds["sarima"]
        preds["lgb"]  = self.lgb.predict(df_val)

        eps = 1e-6
        mses = {k: np.mean((y_real - v[:n]) ** 2) for k, v in preds.items()}
        weights_raw = {k: 1.0 / (v + eps) for k, v in mses.items()}
        if not self.vecm._available:
            weights_raw["vecm"] = 0.0
        total = sum(weights_raw.values())
        self.w_sarima  = weights_raw["sarima"]  / total
        self.w_sarimax = weights_raw["sarimax"] / total
        self.w_vecm    = weights_raw["vecm"]    / total
        self.w_lgb     = weights_raw["lgb"]     / total
        logger.info(
            "Pesos IPP calibrados: SARIMA=%.2f SARIMAX=%.2f VECM=%.2f LGB=%.2f",
            self.w_sarima, self.w_sarimax, self.w_vecm, self.w_lgb,
        )
        for k, mse in mses.items():
            logger.debug("  MSE %s: %.4f (RMSE=%.3f)", k, mse, np.sqrt(mse))

    def predict(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
        devolver_componentes: bool = False,
        modo: Literal["precision", "escenario"] = "precision",
    ) -> pd.DataFrame:
        """Pronostica `horizon` meses hacia adelante.

        modo="precision" (default): pesos calibrados por backtest. Es el pronóstico oficial y
            el que minimiza el error medido.
        modo="escenario": reparte el peso entre los componentes que SÍ tienen canal hacia los
            drivers, para que mover TRM/Brent produzca una respuesta informativa. **No es un
            pronóstico**: es un ejercicio de sensibilidad y la UI debe decirlo.

            Existe porque la sensibilidad a escenarios estaba concentrada en un solo
            componente. Spread medido a 24 meses ante TRM/Brent ±20%:
                ARIMA +0.00 (univariado) · SARIMAX +0.87 · VECM +39.92 · LGB +3.71
            Con los pesos honestos el VECM cae a 0.019 y el spread del ensemble pasa de ~29 a
            1.18 puntos. Separar los modos es lo que permite tener a la vez un pronóstico
            defendible y unos sliders que digan algo.
        """
        es_escenario = modo == "escenario"
        fc_sarima  = self.sarima.forecast(horizon)
        fc_sarimax = self.sarimax.forecast(horizon, exog_future=exog_future)
        brent_cop_path = (
            exog_future["brent_cop"].iloc[:horizon].values
            if exog_future is not None and "brent_cop" in exog_future.columns
            else None
        )
        fc_vecm = self.vecm.forecast(horizon, brent_cop_future=brent_cop_path,
                                     permitir_no_validado=es_escenario)
        pred_lgb   = self.lgb.predict(exog_future.iloc[:horizon]) if exog_future is not None and len(exog_future) >= horizon else fc_sarima["pred"].values

        # Los pesos dependen del horizonte: el mejor modelo a 1 mes no tiene por qué serlo a
        # 24. Si no hay calibración por horizonte, `pesos()` cae a los escalares de siempre.
        w = self.pesos_escenario() if es_escenario else self.pesos(horizon)
        pred = (
            w["sarima"]  * fc_sarima["pred"].values  +
            w["sarimax"] * fc_sarimax["pred"].values +
            (w["vecm"]   * fc_vecm["pred"].values if fc_vecm is not None else 0) +
            w["lgb"]     * pred_lgb
        )
        if fc_vecm is None and w.get("vecm", 0) > 0:
            # El VECM se desactivó tras calibrar: renormalizar para no perder masa.
            resto = 1.0 - w["vecm"]
            if resto > 0:
                pred = pred / resto

        # Correccion de sesgo (si fue actualizada)
        if self.sesgo_por_horizonte and horizon in self.sesgo_por_horizonte:
            pred = pred - self.sesgo_por_horizonte[horizon]

        # Ancho del CI: se toma del componente univariado (ARIMA con deriva), no del SARIMAX.
        #
        # El comentario original decía "el ancho del SARIMAX (mas conservador)", pero eso
        # nunca se verificó. Medido con Winkler-90 sobre 56 orígenes (penaliza ancho Y fallos
        # de cobertura; menor es mejor):
        #     h=6    h=12    h=24
        #     56.0   138.6   306.0   SARIMAX
        #     47.4   109.9   212.1   ARIMA-drift
        # El SARIMAX era peor en todos los horizontes: el modelo con el peor punto le estaba
        # dictando la incertidumbre a todo el ensemble.
        #
        # Sigue siendo un ancho heredado de un componente, no una calibración: las bandas
        # nominales al 90% cubren entre 44% y 75% según el horizonte. Eso lo arregla el
        # calibrador empírico de la Fase 2.4, no este cambio.
        fc_ancho = fc_sarima if "ci_lo90" in fc_sarima.columns else fc_sarimax
        nominal = (fc_ancho["ci_lo90"].values, fc_ancho["ci_hi90"].values)
        pasos = np.arange(1, horizon + 1)

        if self.calibrador is not None:
            # Banda de los cuantiles empíricos del error medido, no del intervalo teórico de
            # un componente. Si el calibrador no tiene datos suficientes, degrada solo al
            # ancho nominal ESCALADO por el factor de subcobertura medido.
            ci_lo, ci_hi = self.calibrador.aplicar(pred, pasos, ci_nominal=nominal)
        else:
            half_log = (np.log(nominal[1].clip(1)) - np.log(nominal[0].clip(1))) / 2
            log_pred = np.log(np.maximum(pred, 1))
            ci_lo = np.exp(log_pred - half_log)
            ci_hi = np.exp(log_pred + half_log)

        result = pd.DataFrame({"pred": pred, "ci_lo90": ci_lo, "ci_hi90": ci_hi})
        if devolver_componentes:
            result["pred_sarima"]  = fc_sarima["pred"].values
            result["pred_sarimax"] = fc_sarimax["pred"].values
            result["pred_vecm"]    = fc_vecm["pred"].values if fc_vecm is not None else np.nan
            result["pred_lgb"]     = pred_lgb
        return result

    def actualizar_sesgo(self, sesgo: dict) -> None:
        """Actualiza la corrección de sesgo desde el loop mensual. Ver EnsembleNivel."""
        self.sesgo_por_horizonte.update(sesgo)
        logger.info("IPP sesgos actualizados: %s", sesgo)


# ---------------------------------------------------------------------------
# Pronosticador IPP (interfaz principal)
# ---------------------------------------------------------------------------

@dataclass
class PronosticadorIPP:
    """Pronosticador multi-horizonte del IPP Colombia.

    Uso:
        p = PronosticadorIPP()
        p.fit(df_ipp_features)
        fc = p.pronosticar(horizonte_meses=12)
    """
    modelo: EnsembleIPP = field(default_factory=EnsembleIPP)
    _df_train: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _fecha_ultimo: object = field(default=None, init=False, repr=False)

    def fit(
        self,
        df_features: pd.DataFrame,
        val_fraccion: float = 0.15,
        muestra: Literal["drivers", "completa"] = "drivers",
        ruta_backtest: str | Path | None = _RUTA_BACKTEST_IPP,
    ) -> "PronosticadorIPP":
        """Ajusta el ensemble con la feature matrix mensual.

        Requiere que df_features tenga columna 'ipp'. Si no la tiene,
        solo ajusta los modelos univariados sobre los drivers.

        Parámetros
        ----------
        muestra : "drivers" recorta al tramo donde los drivers macro existen (2015-01+),
                  que es la muestra estimable por SARIMAX/VECM/LGB. "completa" usa toda la
                  historia del IPP (1999-06+) y solo tiene sentido para modelos univariados.
        """
        if "ipp" not in df_features.columns:
            raise ValueError(
                "df_features no tiene columna 'ipp'. "
                "Cargar datos IPP con load_ipp_local() y volver a correr construir_features.py"
            )
        df = df_features.sort_values("fecha").dropna(subset=["ipp"]).reset_index(drop=True)
        if muestra == "drivers":
            df = recortar_a_drivers(df)
        self._df_train = df
        self._fecha_ultimo = pd.to_datetime(df["fecha"].iloc[-1])

        n_val = max(2, int(len(df) * val_fraccion))
        df_train = df.iloc[:-n_val]
        df_val   = df.iloc[-n_val:]
        logger.info("IPP: %d meses train, %d meses val", len(df_train), len(df_val))
        self.modelo.fit(df_train, df_val=df_val, df_full=df,
                        errores_backtest=_cargar_errores_backtest(ruta_backtest))
        return self

    def pronosticar(
        self,
        horizonte_meses: int = 12,
        df_futuro: pd.DataFrame | None = None,
        devolver_componentes: bool = False,
        modo: Literal["precision", "escenario"] = "precision",
    ) -> pd.DataFrame:
        """Genera el pronóstico de IPP para los próximos `horizonte_meses` meses.

        Parámetros
        ----------
        horizonte_meses   : Meses a pronosticar (1-24)
        df_futuro         : DataFrame con drivers futuros (TRM, Brent, PPI USA asumidos).
                           Si None, propaga el último valor observado (naive macro).
        devolver_componentes: Incluir predicciones por modelo en el resultado

        Returns
        -------
        DataFrame con: fecha, pred, ci_lo90, ci_hi90 [+ componentes si solicitado]
        """
        if df_futuro is None and self._df_train is not None:
            df_futuro = self._construir_futuro_naive(horizonte_meses)

        pred_df = self.modelo.predict(
            horizonte_meses,
            exog_future=df_futuro,
            devolver_componentes=devolver_componentes,
            modo=modo,
        )
        fechas_fc = pd.date_range(
            self._fecha_ultimo + pd.DateOffset(months=1),
            periods=horizonte_meses,
            freq="MS",
        )
        pred_df.insert(0, "fecha", fechas_fc)
        return pred_df

    def _construir_futuro_naive(self, horizonte: int) -> pd.DataFrame:
        """Propaga los ultimos valores observados de drivers futuros."""
        ultima_fila = self._df_train.iloc[[-1]].copy()
        # Repetir la ultima fila h veces (propagar constante)
        df_futuro = pd.concat([ultima_fila] * horizonte, ignore_index=True)
        # Actualizar mes y trig
        fechas = pd.date_range(self._fecha_ultimo + pd.DateOffset(months=1),
                               periods=horizonte, freq="MS")
        df_futuro["fecha"] = fechas
        df_futuro["mes"] = fechas.month
        df_futuro["cos_mes"] = np.cos(2 * np.pi * fechas.month / 12)
        df_futuro["sin_mes"] = np.sin(2 * np.pi * fechas.month / 12)
        return df_futuro

    def construir_futuro_drivers(
        self,
        horizonte: int,
        trm_var_anual: float = 0.0,
        brent_var_anual: float = 0.0,
        oni: float | None = None,
    ) -> pd.DataFrame:
        """Construye df_futuro con los drivers ortogonales del IPP.

        Usado por el dashboard interactivo para escenarios "what-if" de IPP.
        Ver docs/estudio_drivers_ipp.md para el fundamento económico.

        Parametros
        ----------
        horizonte       : meses a proyectar
        trm_var_anual   : variación % ANUAL de TRM (0.05 = depreciación 5%/año)
        brent_var_anual : variación % ANUAL de Brent USD
        oni             : valor ONI asumido (constante). Si None, propaga el ultimo.

        Produce las columnas que consumen SARIMAX y LGB:
          brent_cop, brent_cop_lag{1,2,3}m,
          trm_yoy, trm_yoy_lag1m,
          brent_yoy, brent_yoy_lag1m,
          cos_mes, sin_mes, mes, oni_lag, enso_*
        """
        hist = self._df_train
        ultima = hist.iloc[-1]
        fechas = pd.date_range(self._fecha_ultimo + pd.DateOffset(months=1),
                               periods=horizonte, freq="MS")

        futuro = pd.DataFrame({"fecha": fechas})
        futuro["mes"] = fechas.month
        futuro["cos_mes"] = np.cos(2 * np.pi * fechas.month / 12)
        futuro["sin_mes"] = np.sin(2 * np.pi * fechas.month / 12)

        # Trayectorias de TRM y Brent (compuesta mensual)
        trm_base  = float(hist["trm"].iloc[-1])  if "trm"   in hist.columns else 4000.0
        brent_base = float(hist["brent"].iloc[-1]) if "brent" in hist.columns else 80.0
        trm_factor   = (1.0 + trm_var_anual)   ** (1.0 / 12.0)
        brent_factor = (1.0 + brent_var_anual) ** (1.0 / 12.0)
        futuro["trm"]   = [trm_base   * trm_factor   ** (t + 1) for t in range(horizonte)]
        futuro["brent"] = [brent_base * brent_factor ** (t + 1) for t in range(horizonte)]

        # brent_cop = Brent_USD × TRM: variable ortogonal principal
        futuro["brent_cop"] = futuro["brent"] * futuro["trm"]
        brent_cop_hist = (
            hist["brent_cop"] if "brent_cop" in hist.columns
            else hist["brent"] * hist["trm"]
        )
        brent_cop_full = pd.concat(
            [brent_cop_hist.reset_index(drop=True), futuro["brent_cop"].reset_index(drop=True)],
            ignore_index=True,
        )
        for lag in (1, 2, 3):
            futuro[f"brent_cop_lag{lag}m"] = brent_cop_full.shift(lag).iloc[-horizonte:].to_numpy()

        # trm_yoy y brent_yoy: el escenario asumido ES la variación anual (en %)
        # Lag1m: mes 1 usa el último valor histórico; meses 2+ usan el valor del escenario
        # (esto replica el comportamiento real: en el primer mes el lag todavía refleja historia)
        trm_yoy_hist   = float(hist["trm_yoy"].iloc[-1])   if "trm_yoy"   in hist.columns else trm_var_anual   * 100
        brent_yoy_hist = float(hist["brent_yoy"].iloc[-1]) if "brent_yoy" in hist.columns else brent_var_anual * 100
        futuro["trm_yoy"]         = trm_var_anual   * 100
        futuro["brent_yoy"]       = brent_var_anual * 100
        futuro["trm_yoy_lag1m"]   = [trm_yoy_hist]   + [trm_var_anual   * 100] * (horizonte - 1)
        futuro["brent_yoy_lag1m"] = [brent_yoy_hist] + [brent_var_anual * 100] * (horizonte - 1)

        # Lags de IPP: congelados en el último valor observado para todos los meses del
        # horizonte. En pronóstico multi-paso el valor real de ipp_lag1m en el mes 2
        # depende de la predicción del mes 1, pero actualizar recursivamente amplificaría
        # el error. SARIMAX y VECM manejan la dinámica de largo plazo; LGB usa estos lags
        # principalmente para anclar el nivel inicial.
        for lag_col in ("ipp_lag1m", "ipp_lag2m", "ipp_lag3m", "ipp_lag12m"):
            if lag_col in hist.columns:
                futuro[lag_col] = float(hist[lag_col].iloc[-1])

        # ENSO / ONI
        futuro["oni_lag"] = oni if oni is not None else float(ultima.get("oni_lag", 0.0))
        if oni is not None:
            futuro["enso_el_nino"] = int(oni >= 0.5)
            futuro["enso_la_nina"] = int(oni <= -0.5)
        else:
            futuro["enso_el_nino"] = int(ultima.get("enso_el_nino", 0))
            futuro["enso_la_nina"] = int(ultima.get("enso_la_nina", 0))

        return futuro

    def resumen_modelo(self) -> dict:
        m = self.modelo
        resumen = {
            "w_sarima":  round(m.w_sarima, 3),
            "w_sarimax": round(m.w_sarimax, 3),
            "w_vecm":    round(m.w_vecm, 3),
            "w_lgb":     round(m.w_lgb, 3),
            "vecm_disponible": m.vecm._available,
            # Por qué el VECM está o no está: antes solo se publicaba el booleano, así que un
            # peso de 0.77 llegaba al dashboard sin ninguna justificación visible.
            "vecm_motivo": getattr(m.vecm, "motivo", ""),
            "vecm_k_ar_diff": getattr(m.vecm, "_k_usado", None),
            "n_train": len(self._df_train) if self._df_train is not None else None,
            "pesos_calibrados_por_horizonte": bool(m.pesos_por_horizonte),
            "componente_univariado": type(m.sarima).__name__,
        }
        if m.pesos_por_horizonte:
            resumen["pesos_por_horizonte"] = {
                str(h): {k: round(v, 3) for k, v in w.items()}
                for h, w in sorted(m.pesos_por_horizonte.items())
            }
        diag = getattr(m.sarima, "diagnostico", None)
        if isinstance(diag, dict) and diag:
            resumen["arima_order"] = str(diag.get("order"))
            resumen["arima_trend"] = diag.get("trend")
            resumen["arima_aic"] = round(float(diag.get("aic", float("nan"))), 2)
        return resumen
