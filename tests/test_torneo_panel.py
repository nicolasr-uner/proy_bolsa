from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_backtest_ipp import df_ipp


def test_fabricas_ipp_es_publica_y_trae_el_panel():
    from proybolsa.backtest.ipp_mensual import fabricas_ipp
    fab = fabricas_ipp()
    esperados = {
        "ipp_bench_rw", "ipp_bench_drift", "ipp_bench_theta",
        "ipp_arima_drift", "ipp_sarimax_actual", "ipp_vecm",
        "ipp_lgb_actual", "ipp_ensemble",
    }
    assert esperados <= set(fab), f"faltan modelos: {esperados - set(fab)}"
    modelo = fab["ipp_arima_drift"]()
    assert hasattr(modelo, "fit")


def test_panel_una_fila_por_modelo_y_horizonte(df_ipp):
    from proybolsa.torneo.panel import pronosticar_panel
    origen = df_ipp["fecha"].iloc[-1]
    panel = pronosticar_panel(df_ipp, origen_fecha=origen, horizontes=(1, 6, 12))
    assert set(panel.columns) >= {"modelo", "horizonte", "fecha_objetivo",
                                  "y_pred", "ci_lo90", "ci_hi90"}
    ad = panel[panel["modelo"] == "ipp_arima_drift"]
    assert sorted(ad["horizonte"]) == [1, 6, 12]
    assert np.isfinite(ad["y_pred"]).all()
    assert (ad["fecha_objetivo"] > origen).all()


def test_panel_no_deja_pasar_el_futuro(df_ipp):
    """El pronostico del panel usa drivers CONGELADOS: cambiar el futuro real no lo altera."""
    from proybolsa.torneo.panel import pronosticar_panel
    origen = df_ipp["fecha"].iloc[-13]  # deja 12 meses de 'futuro' en el df
    p1 = pronosticar_panel(df_ipp, origen_fecha=origen, horizontes=(6,))
    df2 = df_ipp.copy()
    fut = df2["fecha"] > origen
    df2.loc[fut, ["brent_cop", "trm_yoy", "brent_yoy"]] *= 3.0
    p2 = pronosticar_panel(df2, origen_fecha=origen, horizontes=(6,))
    m = "ipp_sarimax_actual"
    v1 = p1[p1.modelo == m]["y_pred"].iloc[0]
    v2 = p2[p2.modelo == m]["y_pred"].iloc[0]
    assert v1 == pytest.approx(v2), "el SARIMAX vio el futuro: hay fuga"


def test_panel_aisla_fallos_por_modelo(df_ipp, monkeypatch):
    """Si un modelo del panel lanza excepcion, el panel sigue con los demas (NaN para el que fallo)."""
    from proybolsa.torneo import panel as panel_mod

    class _ModeloRoto:
        def fit(self, *a, **k):
            raise RuntimeError("modelo roto a proposito")

    def _fabricas_con_uno_roto():
        from proybolsa.backtest.ipp_mensual import DriftIPP
        return {"ipp_bench_drift": DriftIPP, "ipp_roto": _ModeloRoto}

    monkeypatch.setattr(panel_mod, "fabricas_ipp", _fabricas_con_uno_roto)
    origen = df_ipp["fecha"].iloc[-1]
    p = panel_mod.pronosticar_panel(df_ipp, origen_fecha=origen, horizontes=(1, 6))
    # el modelo bueno tiene predicciones finitas
    buenos = p[p["modelo"] == "ipp_bench_drift"]
    assert len(buenos) == 2 and np.isfinite(buenos["y_pred"]).all()
    # el modelo roto quedo registrado con NaN, no tumbo el panel
    rotos = p[p["modelo"] == "ipp_roto"]
    assert len(rotos) == 2 and rotos["y_pred"].isna().all()
