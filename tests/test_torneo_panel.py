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
