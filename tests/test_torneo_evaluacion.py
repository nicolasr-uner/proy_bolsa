from __future__ import annotations
import numpy as np
import pandas as pd
from proybolsa.torneo.evaluacion import resolver, agregar_leaderboard

def _registro():
    return pd.DataFrame({
        "origen_fecha": pd.to_datetime(["2025-12-01", "2025-12-01",
                                        "2026-01-01", "2026-01-01"]),
        "modelo": ["ipp_arima_drift", "ipp_bench_drift"] * 2,
        "horizonte": [1, 1, 1, 1],
        "fecha_objetivo": pd.to_datetime(["2026-01-01", "2026-01-01",
                                          "2026-02-01", "2026-02-01"]),
        "y_pred": [100.0, 103.0, 101.0, 104.0],
        "ci_lo90": [np.nan] * 4, "ci_hi90": [np.nan] * 4,
        "fecha_run": ["x"] * 4,
    })

def _actuals():
    return pd.DataFrame({"fecha": pd.to_datetime(["2026-01-01"]), "ipp": [101.0]})

def test_resolver_solo_puntea_lo_vencido():
    res = resolver(_registro(), _actuals())
    assert len(res) == 2
    assert set(res["fecha_objetivo"].dt.month) == {1}
    fila = res[res["modelo"] == "ipp_arima_drift"].iloc[0]
    assert fila["y_real"] == 101.0
    assert fila["error"] == 100.0 - 101.0

def test_leaderboard_skill_vs_naive():
    res = resolver(_registro(), _actuals())
    lb = agregar_leaderboard(res, naive="ipp_bench_drift")
    fila = lb[(lb["modelo"] == "ipp_arima_drift") & (lb["horizonte"] == 1)].iloc[0]
    assert fila["rmse"] == 1.0
    assert fila["skill_vs_naive"] == 0.5
    assert fila["n_resueltos"] == 1
