from __future__ import annotations


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
