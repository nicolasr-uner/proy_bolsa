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

    def test_vecm_condicional_con_path_incondicional_es_identico(self):
        """Pasar el path incondicional de brent_cop produce IPP idéntico al incondicional."""
        df = _df_ipp_sintetico(60)
        m = VECMDriversIPP()
        m.fit(df)
        if not m._available:
            pytest.skip("VECM no disponible")
        fc_incond = m.forecast(12)
        fc_full = m._model.predict(12)           # (12, 2) en log-espacio
        bc_path_incond = np.exp(fc_full[:, 1])   # path incondicional en nivel
        fc_cond = m.forecast(12, brent_cop_future=bc_path_incond)
        assert fc_cond is not None
        assert len(fc_cond) == 12
        np.testing.assert_allclose(fc_cond["pred"].values, fc_incond["pred"].values, rtol=1e-6)

    def test_vecm_condicional_brent_cop_afecta_ipp(self):
        """Paths de brent_cop distintos producen pronósticos de IPP distintos."""
        df = _df_ipp_sintetico(60)
        m = VECMDriversIPP()
        m.fit(df)
        if not m._available:
            pytest.skip("VECM no disponible")
        last_bc = float(df["brent_cop"].dropna().iloc[-1])
        fc_bajo = m.forecast(12, brent_cop_future=np.full(12, last_bc * 0.5))
        fc_alto = m.forecast(12, brent_cop_future=np.full(12, last_bc * 2.0))
        assert not np.allclose(fc_bajo["pred"].values, fc_alto["pred"].values, rtol=1e-3)

    def test_los_escenarios_mueven_el_pronostico_en_modo_escenario(self):
        """Mover los drivers debe producir una respuesta no trivial en el modo escenario.

        Este test exigía antes un spread > 5 puntos al pronóstico **por defecto**, y eso dejó
        de ser cierto al calibrar los pesos honestamente. Medida de dónde venía la
        sensibilidad (spread a 24m ante TRM/Brent ±20%, datos REALES):

            ARIMA +0.00 (univariado)  SARIMAX +0.87  VECM +39.92  LGB +3.71

        Casi toda venía del VECM, el componente cuya cointegración el test de Johansen no
        sostiene y que por eso ahora pesa 0.019 en el pronóstico oficial.

        El umbral de 5 puntos es una propiedad de la serie real, no un invariante: sobre el
        fixture sintético de 60 meses el sistema no está fuertemente cointegrado y el spread
        legítimamente es menor. La comprobación de magnitud vive en
        `scripts/serializar_modelos.py`, que corre sobre datos reales (medido: +14.83 pts).
        Aquí se fija lo que sí debe cumplirse siempre.
        """
        df = _df_ipp_sintetico(60)
        p = PronosticadorIPP()
        p.fit(df, ruta_backtest=None)
        fut_alto = p.construir_futuro_drivers(24, trm_var_anual=0.20, brent_var_anual=0.20)
        fut_bajo = p.construir_futuro_drivers(24, trm_var_anual=-0.20, brent_var_anual=-0.20)
        alto = float(p.pronosticar(24, df_futuro=fut_alto, modo="escenario")["pred"].iloc[-1])
        bajo = float(p.pronosticar(24, df_futuro=fut_bajo, modo="escenario")["pred"].iloc[-1])

        assert abs(alto - bajo) > 0.5, "los drivers no mueven nada: el canal está roto"
        assert alto > bajo, "TRM y Brent al alza deben empujar el IPP hacia arriba"

    def test_modo_escenario_ignora_el_componente_univariado(self):
        """La garantía estructural del modo escenario.

        No se puede prometer que el spread sea mayor que en modo precisión —eso depende de
        qué pesos haya dado la calibración—, pero sí que el ARIMA, que es univariado y tiene
        respuesta exactamente cero a los drivers, no diluya el ejercicio.
        """
        df = _df_ipp_sintetico(60)
        p = PronosticadorIPP()
        p.fit(df, ruta_backtest=None)

        w = p.modelo.pesos_escenario()
        assert w["sarima"] == 0.0, "el univariado no debe participar del modo escenario"
        assert sum(w.values()) == pytest.approx(1.0)
        assert all(w[c] > 0 for c in ("sarimax", "lgb")), "faltan componentes con drivers"

    def test_el_modo_por_defecto_es_precision(self):
        """Nadie debe obtener el modo sensibilidad sin pedirlo."""
        df = _df_ipp_sintetico(60)
        p = PronosticadorIPP()
        p.fit(df, ruta_backtest=None)
        fut = p.construir_futuro_drivers(12, trm_var_anual=0.20, brent_var_anual=0.20)
        por_defecto = p.pronosticar(12, df_futuro=fut)["pred"].to_numpy()
        explicito = p.pronosticar(12, df_futuro=fut, modo="precision")["pred"].to_numpy()
        np.testing.assert_allclose(por_defecto, explicito)


class TestPesosPorHorizonte:
    def test_pesos_varian_con_el_horizonte(self):
        e = EnsembleIPP()
        e.pesos_por_horizonte = {
            1: {"sarima": 0.4, "sarimax": 0.4, "vecm": 0.1, "lgb": 0.1},
            24: {"sarima": 0.7, "sarimax": 0.2, "vecm": 0.05, "lgb": 0.05},
        }
        assert e.pesos(1)["sarima"] == pytest.approx(0.4)
        assert e.pesos(24)["sarima"] == pytest.approx(0.7)

    def test_interpola_entre_horizontes_calibrados(self):
        e = EnsembleIPP()
        e.pesos_por_horizonte = {
            1: {"sarima": 0.4, "sarimax": 0.4, "vecm": 0.1, "lgb": 0.1},
            21: {"sarima": 0.8, "sarimax": 0.1, "vecm": 0.05, "lgb": 0.05},
        }
        w = e.pesos(11)  # punto medio
        assert w["sarima"] == pytest.approx(0.6, abs=0.01)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_extrapola_plano_fuera_del_rango(self):
        e = EnsembleIPP()
        e.pesos_por_horizonte = {6: {"sarima": 0.5, "sarimax": 0.3, "vecm": 0.1, "lgb": 0.1}}
        assert e.pesos(1) == e.pesos(6) == e.pesos(99)

    def test_sin_calibracion_cae_a_los_escalares(self):
        e = EnsembleIPP()
        w = e.pesos(12)
        assert w == {"sarima": 0.25, "sarimax": 0.25, "vecm": 0.25, "lgb": 0.25}

    def test_los_pesos_siempre_suman_uno(self):
        e = EnsembleIPP()
        e.pesos_por_horizonte = {
            1: {"sarima": 0.4, "sarimax": 0.4, "vecm": 0.1, "lgb": 0.1},
            12: {"sarima": 0.6, "sarimax": 0.3, "vecm": 0.05, "lgb": 0.05},
        }
        for h in (1, 3, 6, 12, 24, 36):
            assert sum(e.pesos(h).values()) == pytest.approx(1.0), f"h={h}"

    def test_calibrar_desde_backtest_favorece_al_de_menor_error(self):
        """El componente con menos error debe salir con más peso."""
        filas = []
        for origen in range(20):
            filas += [
                {"modelo": "ipp_arima_drift", "horizonte": 12, "error": 1.0, "fecha_corte": origen},
                {"modelo": "ipp_sarimax_actual", "horizonte": 12, "error": 5.0, "fecha_corte": origen},
            ]
        e = EnsembleIPP().calibrar_pesos_desde_backtest(pd.DataFrame(filas), horizontes=(12,))
        w = e.pesos(12)
        assert w["sarima"] > w["sarimax"], "el de menor error debe pesar más"
        assert sum(w.values()) == pytest.approx(1.0)

    def test_componente_sin_backtest_recibe_el_piso_no_cero(self):
        """No medido no es lo mismo que medido mal."""
        filas = [{"modelo": "ipp_arima_drift", "horizonte": 12, "error": 1.0, "fecha_corte": i}
                 for i in range(20)]
        filas += [{"modelo": "ipp_sarimax_actual", "horizonte": 12, "error": 2.0, "fecha_corte": i}
                  for i in range(20)]
        e = EnsembleIPP().calibrar_pesos_desde_backtest(pd.DataFrame(filas), horizontes=(12,))
        assert e.pesos(12)["vecm"] >= e.piso_peso

    def test_backtest_vacio_no_rompe_ni_calibra(self):
        e = EnsembleIPP()
        e.calibrar_pesos_desde_backtest(pd.DataFrame())
        assert not e.pesos_por_horizonte


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
