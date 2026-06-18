"""Tests del modelo IPP mensual.

Usa series sinteticas que replican las propiedades del IPP colombiano:
  - IPP ~ random walk con tendencia positiva (indice en crecimiento)
  - TRM correlacionada positivamente con IPP (importaciones)
  - Brent correlacionado positivamente
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from proybolsa.models.ipp import (
    EnsembleIPP,
    LGBDriversIPP,
    PronosticadorIPP,
    SARIMABaselineIPP,
    SARIMAXDriversIPP,
    VECMDriversIPP,
    johansen_cointegracion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _df_ipp_sintetico(n: int = 60) -> pd.DataFrame:
    """60 meses de datos mensuales con IPP y drivers ortogonales."""
    rng = np.random.default_rng(99)
    fechas = pd.date_range("2020-01-01", periods=n, freq="MS")

    # IPP: random walk con tendencia ~2% anual
    ipp = 150.0 + np.cumsum(rng.normal(0.3, 2.5, n))
    ipp = np.maximum(ipp, 80.0)

    # TRM: correlacionada con IPP + ruido
    trm = 3200 + 0.8 * (ipp - 150) * 10 + rng.normal(0, 100, n)
    trm = np.maximum(trm, 2000.0)

    # Brent
    brent = 70 + rng.normal(0, 10, n)
    brent = np.maximum(brent, 20.0)

    # PPI USA (mantenido en fixture para retrocompat pero no en _FEATS_LGB)
    ppi_usa = 250 + np.cumsum(rng.normal(0.2, 1.0, n))

    df = pd.DataFrame({
        "fecha": fechas,
        "ipp": ipp,
        "trm": trm,
        "brent": brent,
        "ppi_usa": ppi_usa,
        "mes": fechas.month,
        "cos_mes": np.cos(2 * np.pi * fechas.month / 12),
        "sin_mes": np.sin(2 * np.pi * fechas.month / 12),
        "enso_el_nino": rng.integers(0, 2, n),
        "enso_la_nina": rng.integers(0, 2, n),
        "oni_lag": rng.uniform(-1, 1, n),
    })

    # Driver ortogonal principal: brent_cop = Brent_USD × TRM
    df["brent_cop"] = df["brent"] * df["trm"]

    # Variaciones anuales (NaN para primeros 12 meses, se dropna al final)
    df["trm_yoy"]   = df["trm"].pct_change(12) * 100
    df["brent_yoy"] = df["brent"].pct_change(12) * 100

    # Lags de drivers originales
    for col in ("trm", "brent", "ppi_usa"):
        for lag in (1, 2, 3):
            df[f"{col}_lag{lag}m"] = df[col].shift(lag)

    # Lags de drivers ortogonales
    for lag in (1, 2, 3):
        df[f"brent_cop_lag{lag}m"] = df["brent_cop"].shift(lag)
    df["trm_yoy_lag1m"]   = df["trm_yoy"].shift(1)
    df["brent_yoy_lag1m"] = df["brent_yoy"].shift(1)

    # Lags del IPP (target)
    for lag in (1, 2, 3):
        df[f"ipp_lag{lag}m"] = df["ipp"].shift(lag)
    df["ipp_lag12m"] = df["ipp"].shift(12)

    return df.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tests SARIMA baseline
# ---------------------------------------------------------------------------

class TestSARIMABaselineIPP:
    def test_fit_sin_error(self):
        df = _df_ipp_sintetico(60)
        m = SARIMABaselineIPP()
        m.fit(df)

    def test_forecast_shape(self):
        df = _df_ipp_sintetico(60)
        m = SARIMABaselineIPP()
        m.fit(df)
        fc = m.forecast(6)
        assert len(fc) == 6
        assert "pred" in fc.columns

    def test_forecast_positivo(self):
        df = _df_ipp_sintetico(60)
        m = SARIMABaselineIPP()
        m.fit(df)
        fc = m.forecast(12)
        assert (fc["pred"] > 0).all()

    def test_ci_valido(self):
        df = _df_ipp_sintetico(60)
        m = SARIMABaselineIPP()
        m.fit(df)
        fc = m.forecast(6)
        assert (fc["ci_lo90"] < fc["pred"]).all()
        assert (fc["pred"] < fc["ci_hi90"]).all()


# ---------------------------------------------------------------------------
# Tests SARIMAX con drivers
# ---------------------------------------------------------------------------

class TestSARIMAXDriversIPP:
    def test_fit_y_forecast(self):
        df = _df_ipp_sintetico(60)
        m = SARIMAXDriversIPP()
        m.fit(df)
        fc = m.forecast(6, exog_future=df.tail(6))
        assert len(fc) == 6
        assert (fc["pred"] > 0).all()


# ---------------------------------------------------------------------------
# Tests VECM
# ---------------------------------------------------------------------------

class TestVECMDriversIPP:
    def test_johansen_devuelve_dict(self):
        df = _df_ipp_sintetico(60)
        result = johansen_cointegracion(df)
        assert "n_cointegrating_vectors" in result
        assert "cointegran" in result

    def test_fit_sin_error(self):
        df = _df_ipp_sintetico(60)
        m = VECMDriversIPP()
        m.fit(df)   # puede o no tener cointegración según datos sintéticos

    def test_forecast_si_disponible(self):
        df = _df_ipp_sintetico(60)
        m = VECMDriversIPP()
        m.fit(df)
        if m._available:
            fc = m.forecast(6)
            assert fc is not None
            assert len(fc) == 6
            assert (fc["pred"] > 0).all()


# ---------------------------------------------------------------------------
# Tests LGB IPP
# ---------------------------------------------------------------------------

class TestLGBDriversIPP:
    def test_fit_y_predict(self):
        df = _df_ipp_sintetico(60)
        m = LGBDriversIPP()
        m.fit(df)
        preds = m.predict(df.tail(5))
        assert len(preds) == 5
        assert (preds > 0).all()


# ---------------------------------------------------------------------------
# Tests EnsembleIPP
# ---------------------------------------------------------------------------

class TestEnsembleIPP:
    def test_pesos_suman_uno(self):
        df = _df_ipp_sintetico(60)
        e = EnsembleIPP()
        e.fit(df.head(50), df_val=df.tail(10))
        total = e.w_sarima + e.w_sarimax + e.w_vecm + e.w_lgb
        assert abs(total - 1.0) < 1e-9

    def test_predict_shape(self):
        df = _df_ipp_sintetico(60)
        e = EnsembleIPP()
        e.fit(df.head(50))
        fc = e.predict(6, exog_future=df.tail(6))
        assert len(fc) == 6
        assert "pred" in fc.columns

    def test_ci_valido(self):
        df = _df_ipp_sintetico(60)
        e = EnsembleIPP()
        e.fit(df.head(50))
        fc = e.predict(6, exog_future=df.tail(6))
        assert (fc["ci_lo90"] < fc["pred"]).all()
        assert (fc["pred"] < fc["ci_hi90"]).all()

    def test_componentes(self):
        df = _df_ipp_sintetico(60)
        e = EnsembleIPP()
        e.fit(df.head(50))
        fc = e.predict(4, exog_future=df.tail(4), devolver_componentes=True)
        assert "pred_sarima" in fc.columns
        assert "pred_sarimax" in fc.columns


# ---------------------------------------------------------------------------
# Tests PronosticadorIPP
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pronosticador_ipp():
    df = _df_ipp_sintetico(60)
    p = PronosticadorIPP()
    p.fit(df)
    return p


class TestPronosticadorIPP:
    def test_fit_completa(self, pronosticador_ipp):
        res = pronosticador_ipp.resumen_modelo()
        assert res["n_train"] > 0
        # resumen_modelo redondea a 3dp; tolerancia de 1 ULP de rounding (4 weights × 0.0005)
        assert abs(res["w_sarima"] + res["w_sarimax"] + res["w_vecm"] + res["w_lgb"] - 1.0) < 2e-3

    def test_pronostico_12_meses(self, pronosticador_ipp):
        fc = pronosticador_ipp.pronosticar(12)
        assert len(fc) == 12
        assert "fecha" in fc.columns
        assert "pred" in fc.columns

    def test_fechas_mensuales_consecutivas(self, pronosticador_ipp):
        fc = pronosticador_ipp.pronosticar(6)
        fechas = pd.to_datetime(fc["fecha"])
        diffs = fechas.diff().dropna().dt.days
        assert (diffs >= 28).all() and (diffs <= 31).all()

    def test_ci_valido(self, pronosticador_ipp):
        fc = pronosticador_ipp.pronosticar(6)
        assert (fc["ci_lo90"] < fc["pred"]).all()
        assert (fc["pred"] < fc["ci_hi90"]).all()

    def test_sin_ipp_levanta_error(self):
        df = _df_ipp_sintetico(60).drop(columns=["ipp"])
        with pytest.raises(ValueError, match="ipp"):
            PronosticadorIPP().fit(df)

    def test_componentes_reajustados_sobre_serie_completa(self):
        """El forecast de produccion debe usar componentes ajustados sobre TODA la
        serie, no solo sobre df_train (que excluye los meses recientes para calibrar
        pesos). Bug historico: SARIMA/SARIMAX se ajustaban hasta el fin de df_train,
        dejando el pronostico stale ~N meses y desanclado del ultimo dato real."""
        df = _df_ipp_sintetico(60)
        p = PronosticadorIPP()
        p.fit(df)
        n_full = len(df.dropna(subset=["ipp"]))
        assert int(p.modelo.sarima._result.nobs) == n_full
        assert int(p.modelo.sarimax._result.nobs) == n_full

    def test_construir_futuro_drivers(self, pronosticador_ipp):
        """El constructor de escenarios arma df_futuro con trayectorias de drivers."""
        fut = pronosticador_ipp.construir_futuro_drivers(
            12, trm_var_anual=0.10, brent_var_anual=-0.05
        )
        assert len(fut) == 12
        assert fut["trm"].iloc[-1] > fut["trm"].iloc[0]       # TRM sube 10%/ano
        assert fut["brent"].iloc[-1] < fut["brent"].iloc[0]   # Brent baja 5%/ano
        # brent_cop debe subir aunque brent baje, porque TRM sube más (10% > 5% efecto neto)
        assert fut["brent_cop"].iloc[-1] > fut["brent_cop"].iloc[0]
        # columnas que consumen SARIMAX y LGB (drivers ortogonales)
        for col in ("brent_cop_lag1m", "brent_cop_lag2m", "trm_yoy", "brent_yoy", "cos_mes"):
            assert col in fut.columns, f"falta {col}"

    def test_pronosticar_con_futuro_construido(self, pronosticador_ipp):
        """Se puede pronosticar con el df_futuro de un escenario."""
        fut = pronosticador_ipp.construir_futuro_drivers(12, trm_var_anual=0.05)
        fc = pronosticador_ipp.pronosticar(12, df_futuro=fut)
        assert len(fc) == 12
        assert (fc["pred"] > 0).all()

    def test_oni_override_en_futuro(self, pronosticador_ipp):
        """El ONI asumido se propaga al df_futuro."""
        fut = pronosticador_ipp.construir_futuro_drivers(6, oni=1.5)
        assert (fut["oni_lag"] == 1.5).all()
