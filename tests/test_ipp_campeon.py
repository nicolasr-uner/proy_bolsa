"""Estrategia de pesos 'campeon': el pronostico oficial del IPP es el mejor componente
por horizonte (pick-the-winner), no una mezcla que pierde contra su mejor parte.

Decision 2026-09-07: medido en backtest, el ensemble pierde contra arima_drift en todo
horizonte y reponderar (cualquier lambda_shrink) no lo salva. Se despliega el campeon;
el torneo mide el panel en vivo por si otro lo supera.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from proybolsa.models.ipp.modelo_ipp import EnsembleIPP
from tests.test_backtest_ipp import df_ipp  # fixture compartido (mensual con drivers)


def _errores_sinteticos(ganador: str = "ipp_arima_drift") -> pd.DataFrame:
    """Backtest sintetico donde `ganador` tiene el menor error en todos los horizontes."""
    rng = np.random.default_rng(0)
    escala = {"ipp_arima_drift": 1.0, "ipp_sarimax_actual": 5.0,
              "ipp_vecm": 6.0, "ipp_lgb_actual": 8.0}
    escala[ganador] = 0.5  # el ganador, mucho mas chico
    filas = []
    for h in (1, 3, 6, 12, 24):
        for modelo, s in escala.items():
            for _ in range(20):
                filas.append({"modelo": modelo, "horizonte": h,
                              "error": float(rng.normal(0, s))})
    return pd.DataFrame(filas)


def test_fijar_pesos_campeon_pone_todo_en_el_mejor_por_horizonte():
    err = _errores_sinteticos(ganador="ipp_arima_drift")
    ens = EnsembleIPP()
    ens.fijar_pesos_campeon(err, horizontes=(1, 3, 6, 12, 24))

    for h in (1, 3, 6, 12, 24):
        w = ens.pesos_por_horizonte[h]
        assert w["sarima"] == 1.0, f"h={h}: el campeon (arima_drift=sarima) debe pesar 1.0"
        assert w["sarimax"] == 0.0 and w["vecm"] == 0.0 and w["lgb"] == 0.0


def test_fijar_pesos_campeon_sigue_al_ganador_real():
    """Si otro componente es el mejor, el campeon cambia (self-adapta al backtest)."""
    err = _errores_sinteticos(ganador="ipp_sarimax_actual")
    ens = EnsembleIPP()
    ens.fijar_pesos_campeon(err, horizontes=(12,))
    w = ens.pesos_por_horizonte[12]
    assert w["sarimax"] == 1.0 and w["sarima"] == 0.0


def test_fit_con_estrategia_campeon_despliega_el_campeon(df_ipp):
    """fit(estrategia_pesos='campeon') deja el peso vigente concentrado en el ganador."""
    err = _errores_sinteticos(ganador="ipp_arima_drift")
    ens = EnsembleIPP()
    ens.fit(df_ipp.iloc[:120], errores_backtest=err, estrategia_pesos="campeon")
    assert ens.pesos(12)["sarima"] == 1.0
