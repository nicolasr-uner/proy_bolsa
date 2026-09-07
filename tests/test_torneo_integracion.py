from __future__ import annotations

import numpy as np
import pandas as pd

from tests.test_backtest_ipp import df_ipp


def test_ciclo_registra_puntea_y_lidera_el_mejor(df_ipp, tmp_path):
    """Corriendo el ciclo mes a mes, el registro crece, el scorer resuelve al llegar los
    reales, y el leaderboard termina con evidencia (n_resueltos>0) para arima_drift."""
    from proybolsa.torneo.ciclo import correr_ciclo_torneo
    from proybolsa.torneo.evaluacion import agregar_leaderboard
    import proybolsa.torneo.ciclo as ciclo

    dir_out = tmp_path
    corte0 = len(df_ipp) - 8
    for i in range(6):
        hist = df_ipp.iloc[: corte0 + i].reset_index(drop=True)
        correr_ciclo_torneo(hist, dir_salida=dir_out, fecha_run=f"2026{i:02d}01",
                            horizontes=(1, 3))
    reg = pd.read_parquet(dir_out / "registro_ipp.parquet")
    res = pd.read_parquet(dir_out / "resueltos_ipp.parquet")
    lb = pd.read_parquet(dir_out / "leaderboard_ipp.parquet")
    assert len(reg) > 0 and len(res) > 0
    ad = lb[(lb.modelo == "ipp_arima_drift") & (lb.horizonte == 1)]
    assert not ad.empty and ad["n_resueltos"].iloc[0] > 0

def test_ciclo_es_idempotente(df_ipp, tmp_path):
    from proybolsa.torneo.ciclo import correr_ciclo_torneo
    hist = df_ipp.iloc[:-4].reset_index(drop=True)
    correr_ciclo_torneo(hist, dir_salida=tmp_path, fecha_run="20260101", horizontes=(1,))
    n1 = len(pd.read_parquet(tmp_path / "registro_ipp.parquet"))
    correr_ciclo_torneo(hist, dir_salida=tmp_path, fecha_run="20260101", horizontes=(1,))
    n2 = len(pd.read_parquet(tmp_path / "registro_ipp.parquet"))
    assert n1 == n2, "re-correr el mismo ciclo no debe duplicar el registro"
