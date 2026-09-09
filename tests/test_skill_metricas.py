"""Skill-vs-naive por horizonte desde las metricas del backtest (para los resumen_*.json)."""
from __future__ import annotations

import pandas as pd

from proybolsa.backtest.rolling_origin import skill_vs_naive_por_horizonte


def _met():
    return pd.DataFrame({
        "modelo": ["ensemble", "naive", "ensemble", "naive"],
        "horizonte": [1, 1, 30, 30],
        "rmse": [90.0, 100.0, 200.0, 100.0],
    })


def test_skill_positivo_y_negativo_por_horizonte():
    sk = skill_vs_naive_por_horizonte(_met(), modelo="ensemble", naive="naive")
    assert sk[1] == 0.1    # 1 - 90/100 = 0.10 -> le gana al naive
    assert sk[30] == -1.0  # 1 - 200/100 = -1.0 -> pierde
    assert set(sk) == {1, 30}


def test_skill_vacio_no_crashea():
    assert skill_vs_naive_por_horizonte(pd.DataFrame(), modelo="x", naive="y") == {}


def test_skill_ignora_horizonte_sin_naive():
    m = pd.DataFrame({"modelo": ["ensemble"], "horizonte": [7], "rmse": [50.0]})
    assert skill_vs_naive_por_horizonte(m, modelo="ensemble", naive="naive") == {}
