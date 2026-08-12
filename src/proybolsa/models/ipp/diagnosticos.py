"""Diagnósticos estadísticos de la serie del IPP.

Funciones puras: reciben DataFrames, devuelven DataFrames o dicts serializables, no tocan
disco ni dibujan. `scripts/estudio_modelo_ipp.py` las orquesta para producir el estudio.

Contexto de por qué existen: el docstring de `SARIMABaselineIPP` afirmaba "estima el mejor
orden (p,d,q)(P,D,Q,12) buscando entre modelos candidatos y seleccionando por AIC", y el
código fijaba `(1,1,1)` sin búsqueda ni diagnóstico. `johansen_cointegracion` se llamaba pero
su resultado se descartaba con `max(1, n_coint)`. Estas funciones convierten esas
afirmaciones en mediciones.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _serie_log(df: pd.DataFrame, col: str = "ipp") -> pd.Series:
    return np.log(df[col].dropna().reset_index(drop=True))


def test_raices_unitarias(serie: pd.Series,
                          transformaciones: tuple[str, ...] = ("log", "log_dif")) -> pd.DataFrame:
    """ADF y KPSS por transformación.

    Se reportan juntos a propósito: sus hipótesis nulas son opuestas (ADF H0 = raíz unitaria,
    KPSS H0 = estacionariedad), así que coincidir es evidencia mucho más fuerte que cualquiera
    de los dos por separado.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    y = serie.dropna()
    datos = {"nivel": y, "log": np.log(y), "log_dif": np.log(y).diff().dropna()}

    filas = []
    for nombre in transformaciones:
        s = datos.get(nombre)
        if s is None or len(s) < 20:
            continue
        regresion = "ct" if nombre in ("nivel", "log") else "c"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = adfuller(s, regression=regresion, autolag="AIC")
            kp = kpss(s, regression=regresion, nlags="auto")
        filas.append({
            "transformacion": nombre,
            "n": len(s),
            "adf_stat": round(float(adf[0]), 3),
            "adf_p": round(float(adf[1]), 4),
            "adf_rechaza_raiz_unitaria": bool(adf[1] < 0.05),
            "kpss_stat": round(float(kp[0]), 3),
            "kpss_p": round(float(kp[1]), 4),
            "kpss_rechaza_estacionariedad": bool(kp[1] < 0.05),
        })
    return pd.DataFrame(filas)


def perfil_estacional(dlog: pd.Series, fechas: pd.Series) -> dict:
    """¿Hay estacionalidad mensual real en los cambios del IPP?

    El docstring del módulo de IPP afirma "estacionalidad débil". Esto lo mide: F-test de
    dummies de mes sobre dlog, más la ACF en el rezago 12. Si no se rechaza, un componente
    SARIMA (P,D,Q,12) gasta parámetros en un patrón que no está.
    """
    from statsmodels.tsa.stattools import acf

    d = pd.DataFrame({"dlog": dlog, "mes": pd.to_datetime(fechas).dt.month}).dropna()
    if len(d) < 30:
        return {"n": len(d), "nota": "muestra insuficiente"}

    media_global = d["dlog"].mean()
    por_mes = d.groupby("mes")["dlog"]
    k = por_mes.ngroups
    n = len(d)
    ss_entre = float(sum(g.size * (g.mean() - media_global) ** 2 for _, g in por_mes))
    ss_dentro = float(sum(((g - g.mean()) ** 2).sum() for _, g in por_mes))
    gl1, gl2 = k - 1, n - k
    f_stat = (ss_entre / gl1) / (ss_dentro / gl2) if ss_dentro > 0 and gl2 > 0 else np.nan

    from scipy import stats as _st
    p_valor = float(_st.f.sf(f_stat, gl1, gl2)) if np.isfinite(f_stat) else np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = acf(d["dlog"], nlags=min(13, len(d) // 3), fft=False)

    return {
        "n": n,
        "f_stat": round(float(f_stat), 4),
        "p_valor": round(p_valor, 4),
        "r2": round(float(ss_entre / (ss_entre + ss_dentro)), 4) if (ss_entre + ss_dentro) else np.nan,
        "hay_estacionalidad": bool(p_valor < 0.05) if np.isfinite(p_valor) else False,
        "acf_lag12": round(float(a[12]), 4) if len(a) > 12 else np.nan,
        "efecto_por_mes": {int(m): round(float(g.mean() - media_global), 5) for m, g in por_mes},
    }


def acf_pacf_tabla(serie: pd.Series, nlags: int = 24) -> pd.DataFrame:
    """ACF y PACF con las bandas de significancia +-1.96/sqrt(n)."""
    from statsmodels.tsa.stattools import acf, pacf

    s = serie.dropna()
    nlags = min(nlags, len(s) // 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = acf(s, nlags=nlags, fft=False)
        p = pacf(s, nlags=nlags)
    banda = 1.96 / np.sqrt(len(s))
    return pd.DataFrame({
        "lag": range(len(a)),
        "acf": np.round(a, 4),
        "pacf": np.round(np.pad(p, (0, len(a) - len(p)), constant_values=np.nan), 4),
        "banda": round(float(banda), 4),
        "acf_significativa": np.abs(a) > banda,
    })


def buscar_orden_arima(y_log: pd.Series, *, p_max: int = 3, q_max: int = 3,
                       drift: tuple[bool, ...] = (True, False),
                       exog: pd.DataFrame | None = None,
                       criterio: Literal["aic", "bic", "hqic"] = "aic") -> pd.DataFrame:
    """Rejilla de órdenes ARIMA(p,1,q) con y sin deriva, ordenada por criterio.

    Incluir `trend="c"` entre los candidatos es el punto entero de esta función: es lo que el
    modelo en producción nunca probó.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    filas = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            for con_drift in drift:
                trend = "c" if con_drift else None
                if (p, q) == (0, 0) and trend is None:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = SARIMAX(y_log, exog=exog, order=(p, 1, q),
                                    seasonal_order=(0, 0, 0, 0), trend=trend,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit(disp=False)
                    if not np.isfinite(r.aic):
                        continue
                    filas.append({
                        "p": p, "d": 1, "q": q, "drift": con_drift,
                        "aic": round(float(r.aic), 2),
                        "bic": round(float(r.bic), 2),
                        "hqic": round(float(r.hqic), 2),
                        "n_params": int(len(r.params)),
                        "convergio": True,
                    })
                except Exception:
                    filas.append({"p": p, "d": 1, "q": q, "drift": con_drift,
                                  "aic": np.nan, "bic": np.nan, "hqic": np.nan,
                                  "n_params": np.nan, "convergio": False})
    df = pd.DataFrame(filas)
    return df.sort_values(criterio, na_position="last").reset_index(drop=True)


def diagnostico_residuos(res) -> dict:
    """Ljung-Box, normalidad y heterocedasticidad sobre los residuos de un ajuste."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera

    resid = pd.Series(np.asarray(res.resid, dtype=float)).dropna()
    salida: dict = {"n": len(resid)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lag in (12, 24):
            if len(resid) > lag + 5:
                lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
                salida[f"ljungbox_p_lag{lag}"] = round(float(lb["lb_pvalue"].iloc[0]), 4)
        jb = jarque_bera(resid)
        salida["jarque_bera_p"] = round(float(jb[1]), 4)
        salida["skew"] = round(float(jb[2]), 3)
        salida["kurtosis"] = round(float(jb[3]), 3)
        # ARCH: Ljung-Box sobre los residuos al cuadrado.
        if len(resid) > 17:
            lb2 = acorr_ljungbox(resid ** 2, lags=[12], return_df=True)
            salida["arch_p_lag12"] = round(float(lb2["lb_pvalue"].iloc[0]), 4)
    salida["residuos_blancos"] = bool(salida.get("ljungbox_p_lag12", 0) > 0.05)
    return salida


def seleccionar_k_ar_diff(df_log: pd.DataFrame, max_lags: int = 6) -> int:
    """Rezago del VECM por AIC del VAR en niveles (k_ar_diff = p_var - 1).

    El código actual usa `k_ar_diff=2` en el test de Johansen y `k_ar_diff=1` en la
    estimación. Ese desacople importa: el resultado del test depende del rezago.
    """
    from statsmodels.tsa.api import VAR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = VAR(df_log.dropna().to_numpy()).select_order(maxlags=max_lags)
        p = int(sel.aic)
    return max(1, p - 1)


def johansen_reporte(df: pd.DataFrame, variables: tuple[str, ...] = ("ipp", "brent_cop"),
                     k_ar_diff_grid: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Traza de Johansen para varios rezagos, con los críticos al 90/95/99%.

    Reportar la rejilla y no un solo k es el punto: si el rango de cointegración cambia con el
    rezago, la evidencia es frágil y el VECM no debería activarse por defecto.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    cols = [c for c in variables if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    data = np.log(df[cols].dropna())
    if len(data) < 20:
        return pd.DataFrame()

    filas = []
    for k in k_ar_diff_grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = coint_johansen(data.to_numpy(), det_order=0, k_ar_diff=k)
            for r in range(len(res.lr1)):
                filas.append({
                    "k_ar_diff": k, "r": r, "n_obs": len(data),
                    "traza": round(float(res.lr1[r]), 3),
                    "crit_90": round(float(res.cvt[r, 0]), 3),
                    "crit_95": round(float(res.cvt[r, 1]), 3),
                    "crit_99": round(float(res.cvt[r, 2]), 3),
                    "rechaza_95": bool(res.lr1[r] > res.cvt[r, 1]),
                    "rechaza_90": bool(res.lr1[r] > res.cvt[r, 0]),
                })
        except Exception as exc:
            logger.warning("Johansen k=%d falló: %s", k, exc)
    return pd.DataFrame(filas)


def decidir_cointegracion(reporte: pd.DataFrame, k_elegido: int, n_obs: int,
                          *, min_obs: int = 80) -> dict:
    """Regla de activación del VECM, explícita y auditable.

    El código actual hace `n_coint = max(1, test["n_cointegrating_vectors"])` y luego fija
    `_available = True` incondicionalmente: el test se corre y su resultado NO se usa. Aquí se
    exige, además de que la traza rechace r=0 con el rezago elegido, que también rechace al
    90% con k±1. Esa condición de robustez es la que decide de verdad, porque en los datos
    reales el rango cambia entre k=2 y k=3.
    """
    motivo: list[str] = []
    if n_obs < min_obs:
        return {"activar": False, "rango": 0,
                "motivo": f"muestra insuficiente ({n_obs} < {min_obs} obs)"}
    if reporte.empty:
        return {"activar": False, "rango": 0, "motivo": "el test no pudo correr"}

    fila = reporte[(reporte["k_ar_diff"] == k_elegido) & (reporte["r"] == 0)]
    if fila.empty:
        return {"activar": False, "rango": 0, "motivo": f"sin resultado para k={k_elegido}"}
    if not bool(fila["rechaza_95"].iloc[0]):
        return {"activar": False, "rango": 0,
                "motivo": f"la traza no rechaza r=0 al 95% con k={k_elegido} "
                          f"({fila['traza'].iloc[0]} < {fila['crit_95'].iloc[0]})"}

    vecinos = reporte[(reporte["k_ar_diff"].isin([k_elegido - 1, k_elegido + 1])) & (reporte["r"] == 0)]
    if not vecinos.empty and not vecinos["rechaza_90"].all():
        fallan = vecinos.loc[~vecinos["rechaza_90"], "k_ar_diff"].tolist()
        return {"activar": False, "rango": 0,
                "motivo": f"cointegración frágil: no se sostiene al 90% con k={fallan}"}

    rango = int((reporte[(reporte["k_ar_diff"] == k_elegido)]["rechaza_95"]).sum())
    motivo.append(f"traza rechaza r=0 al 95% con k={k_elegido} y se sostiene con k±1")
    return {"activar": True, "rango": max(1, rango), "motivo": "; ".join(motivo)}


def quiebres_regimen(dlog: pd.Series, fechas: pd.Series) -> pd.DataFrame:
    """Media y desviación de dlog por año, anualizadas.

    El IPP colombiano tiene episodios inflacionarios discretos (2021-2022) separados de
    períodos casi planos. Es el argumento central contra confiar en un punto a 24 meses.
    """
    d = pd.DataFrame({"dlog": dlog, "anio": pd.to_datetime(fechas).dt.year}).dropna()
    g = d.groupby("anio")["dlog"]
    return pd.DataFrame({
        "anio": g.mean().index,
        "n_meses": g.size().to_numpy(),
        "dlog_medio_anualizado_pct": (g.mean() * 12 * 100).round(2).to_numpy(),
        "sd_mensual_pct": (g.std() * 100).round(3).to_numpy(),
    }).reset_index(drop=True)
