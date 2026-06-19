"""Modelo de IPP mensual (Indice de Precios al Productor, Colombia).

Pipeline:
  1. SARIMA univariado (baseline robusto)
  2. SARIMAX con drivers macro (TRM, Brent, PPI USA con rezagos)
  3. VECM si hay cointegración Johansen entre IPP y drivers
  4. LightGBM sobre features tabulares
  5. Ensemble por inverse-MSE sobre período de validación

El IPP colombiano tiene:
  - Estacionalidad débil (la manufactura no es tan estacional como el precio de bolsa)
  - Cointegración de largo plazo con TRM y PPI internacional (literatura BanRep)
  - Rezago típico de 1-3 meses (transmisión importaciones -> precios internos)

Nota: Requiere ipp > 0 (índice base=100, DANE dic-2014). El modelo trabaja en
diferencias logarítmicas (log_dif) para estacionariedad y el pronóstico se
reconstruye por integración.
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
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

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
class SARIMABaselineIPP:
    """SARIMA univariado en diferencias log del IPP.

    Estima el mejor orden (p,d,q)(P,D,Q,12) buscando entre modelos candidatos
    y seleccionando por AIC. Trabaja en log-dif para estacionariedad.
    """
    _result: object = field(default=None, init=False, repr=False)
    _ipp_init: float = field(default=100.0, init=False, repr=False)  # ultimo nivel para reconstruir

    def fit(self, df_train: pd.DataFrame) -> "SARIMABaselineIPP":
        ipp = df_train["ipp"].dropna()
        self._ipp_init = float(ipp.iloc[-1])
        log_ipp = np.log(ipp)
        # ARIMA(1,1,1) sin componente estacional (IPP mensual tiene ciclo debil)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(log_ipp, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                            enforce_stationarity=False, enforce_invertibility=False)
            self._result = model.fit(disp=False)
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
class SARIMAXDriversIPP:
    """SARIMAX con TRM, Brent y PPI USA rezagados como variables exógenas."""
    _result: object = field(default=None, init=False, repr=False)
    _exog_cols: list = field(default_factory=list, init=False, repr=False)

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
                            order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                            enforce_stationarity=False, enforce_invertibility=False)
            self._result = model.fit(disp=False)
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
    """VECM entre IPP, TRM y Brent. Solo se usa si el test Johansen encuentra cointegracion.

    Modela la relacion de largo plazo entre los tres indices (todos en log).
    El pronostico del IPP extrae la ecuacion correspondiente del sistema.
    """
    k_ar_diff: int = 1
    _model: object = field(default=None, init=False, repr=False)
    _n_coint: int = field(default=1, init=False, repr=False)
    _available: bool = field(default=False, init=False, repr=False)

    def fit(self, df_train: pd.DataFrame) -> "VECMDriversIPP":
        df = df_train.copy()
        if "brent_cop" not in df.columns:
            if "brent" in df.columns and "trm" in df.columns:
                df["brent_cop"] = df["brent"] * df["trm"]
            else:
                logger.warning("VECM no disponible: falta brent_cop (o brent+trm)")
                self._available = False
                return self

        if "ipp" not in df.columns or len(df) < 20:
            logger.warning("VECM no disponible: falta ipp o datos insuficientes")
            self._available = False
            return self

        data = df[["ipp", "brent_cop"]].dropna()
        if len(data) < 20:
            self._available = False
            return self

        data_log = np.log(data.astype(float))

        # Test de cointegración {IPP, brent_cop}
        test = johansen_cointegracion(df)
        n_coint = max(1, test["n_cointegrating_vectors"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VECM(data_log.values, k_ar_diff=self.k_ar_diff,
                         coint_rank=n_coint, deterministic="ci")
            self._model = model.fit()

        self._n_coint = n_coint
        self._available = True
        logger.debug("VECM IPP ajustado. Rango cointegración=%d", n_coint)
        return self

    def forecast(self, horizon: int, brent_cop_future=None) -> pd.DataFrame | None:
        if not self._available or self._model is None:
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
    """Ensemble SARIMA + SARIMAX + VECM + LGB para IPP mensual.

    Pesos calibrados por inverse-MSE sobre período de validación.
    Si VECM no está disponible (sin cointegración), usa solo SARIMA+SARIMAX+LGB.
    """
    sarima: SARIMABaselineIPP = field(default_factory=SARIMABaselineIPP)
    sarimax: SARIMAXDriversIPP = field(default_factory=SARIMAXDriversIPP)
    vecm: VECMDriversIPP = field(default_factory=VECMDriversIPP)
    lgb: LGBDriversIPP = field(default_factory=LGBDriversIPP)
    w_sarima: float = 0.25
    w_sarimax: float = 0.25
    w_vecm: float = 0.25
    w_lgb: float = 0.25
    sesgo_por_horizonte: dict = field(default_factory=dict)

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame | None = None,
        df_full: pd.DataFrame | None = None,
    ) -> "EnsembleIPP":
        """Ajusta los 4 componentes y calibra pesos.

        Los componentes se ajustan sobre df_train y los pesos se calibran sobre df_val
        (validacion honesta). Si se pasa df_full, los componentes se RE-AJUSTAN sobre
        toda la serie para el pronostico de produccion: de lo contrario el forecast
        partiria del fin de df_train (stale) en vez del ultimo dato observado.
        """
        logger.info("Ajustando SARIMA IPP...")
        self.sarima.fit(df_train)
        logger.info("Ajustando SARIMAX IPP...")
        self.sarimax.fit(df_train)
        logger.info("Ajustando VECM IPP...")
        self.vecm.fit(df_train)
        logger.info("Ajustando LGB IPP...")
        self.lgb.fit(df_train)

        if df_val is not None and len(df_val) >= 3:
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
            self.w_sarima = 0.3; self.w_sarimax = 0.4; self.w_vecm = 0.0; self.w_lgb = 0.3
        logger.info("Pesos IPP: SARIMA=%.2f SARIMAX=%.2f VECM=%.2f LGB=%.2f",
                    self.w_sarima, self.w_sarimax, self.w_vecm, self.w_lgb)

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
    ) -> pd.DataFrame:
        """Pronostica `horizon` meses hacia adelante."""
        fc_sarima  = self.sarima.forecast(horizon)
        fc_sarimax = self.sarimax.forecast(horizon, exog_future=exog_future)
        brent_cop_path = (
            exog_future["brent_cop"].iloc[:horizon].values
            if exog_future is not None and "brent_cop" in exog_future.columns
            else None
        )
        fc_vecm    = self.vecm.forecast(horizon, brent_cop_future=brent_cop_path)
        pred_lgb   = self.lgb.predict(exog_future.iloc[:horizon]) if exog_future is not None and len(exog_future) >= horizon else fc_sarima["pred"].values

        pred = (
            self.w_sarima  * fc_sarima["pred"].values  +
            self.w_sarimax * fc_sarimax["pred"].values +
            (self.w_vecm   * fc_vecm["pred"].values if fc_vecm is not None else 0) +
            self.w_lgb     * pred_lgb
        )

        # Correccion de sesgo (si fue actualizada)
        if self.sesgo_por_horizonte and horizon in self.sesgo_por_horizonte:
            pred = pred - self.sesgo_por_horizonte[horizon]

        # CI: tomar el ancho del SARIMAX (mas conservador)
        half_log = (np.log(fc_sarimax["ci_hi90"].values.clip(1)) -
                    np.log(fc_sarimax["ci_lo90"].values.clip(1))) / 2
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
    ) -> "PronosticadorIPP":
        """Ajusta el ensemble con la feature matrix mensual.

        Requiere que df_features tenga columna 'ipp'. Si no la tiene,
        solo ajusta los modelos univariados sobre los drivers.
        """
        if "ipp" not in df_features.columns:
            raise ValueError(
                "df_features no tiene columna 'ipp'. "
                "Cargar datos IPP con load_ipp_local() y volver a correr construir_features.py"
            )
        df = df_features.sort_values("fecha").dropna(subset=["ipp"]).reset_index(drop=True)
        self._df_train = df
        self._fecha_ultimo = pd.to_datetime(df["fecha"].iloc[-1])

        n_val = max(2, int(len(df) * val_fraccion))
        df_train = df.iloc[:-n_val]
        df_val   = df.iloc[-n_val:]
        logger.info("IPP: %d meses train, %d meses val", len(df_train), len(df_val))
        self.modelo.fit(df_train, df_val=df_val, df_full=df)
        return self

    def pronosticar(
        self,
        horizonte_meses: int = 12,
        df_futuro: pd.DataFrame | None = None,
        devolver_componentes: bool = False,
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
        return {
            "w_sarima":  round(self.modelo.w_sarima, 3),
            "w_sarimax": round(self.modelo.w_sarimax, 3),
            "w_vecm":    round(self.modelo.w_vecm, 3),
            "w_lgb":     round(self.modelo.w_lgb, 3),
            "vecm_disponible": self.modelo.vecm._available,
            "n_train": len(self._df_train) if self._df_train is not None else None,
        }
