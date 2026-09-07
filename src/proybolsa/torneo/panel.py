"""Pronostico del panel completo de modelos en un origen dado, con drivers CONGELADOS.

Reusa la fabrica del backtest (`fabricas_ipp`) y `congelar_drivers_futuros`, de modo que el
panel del torneo mide exactamente lo mismo que el backtest: multi-paso real sin foresight. El
ensemble entra con sus pesos de produccion (fit con errores_backtest); el resto con fit(train).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from proybolsa.backtest.ipp_mensual import (
    _llamar_forecast,
    congelar_drivers_futuros,
    fabricas_ipp,
)

HORIZONTES = (1, 3, 6, 12, 24)


def _marco_futuro(train: pd.DataFrame, origen_fecha: pd.Timestamp, h_max: int) -> pd.DataFrame:
    """Filas futuras con el calendario real y los drivers copiados del ultimo train
    (congelar_drivers_futuros los re-congela; el calendario SI es conocido)."""
    fechas = pd.date_range(origen_fecha, periods=h_max + 1, freq="MS")[1:]
    fut = pd.DataFrame({"fecha": fechas})
    fut["mes"] = fut["fecha"].dt.month
    fut["cos_mes"] = np.cos(2 * np.pi * fut["mes"] / 12)
    fut["sin_mes"] = np.sin(2 * np.pi * fut["mes"] / 12)
    ultimo = train.iloc[-1]
    for col in train.columns:
        if col not in fut.columns and col not in ("fecha", "ipp"):
            fut[col] = ultimo[col]
    return fut


def pronosticar_panel(df: pd.DataFrame, origen_fecha, horizontes=HORIZONTES,
                      errores_backtest: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.sort_values("fecha").reset_index(drop=True)
    origen_fecha = pd.Timestamp(origen_fecha)
    train = df[df["fecha"] <= origen_fecha].reset_index(drop=True)
    h_max = max(horizontes)
    fut = _marco_futuro(train, origen_fecha, h_max)
    exog = congelar_drivers_futuros(fut, train, modo="congelado")
    fechas_obj = fut["fecha"].to_numpy()

    filas = []
    for nombre, fabrica in fabricas_ipp().items():
        modelo = fabrica()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if nombre == "ipp_ensemble":
                modelo.fit(train, errores_backtest=errores_backtest)
            else:
                modelo.fit(train)
            fc = _llamar_forecast(modelo, h_max, exog)
        for h in horizontes:
            filas.append({
                "modelo": nombre,
                "horizonte": h,
                "fecha_objetivo": pd.Timestamp(fechas_obj[h - 1]),
                "y_pred": float(fc["pred"].iloc[h - 1]),
                "ci_lo90": float(fc["ci_lo90"].iloc[h - 1]),
                "ci_hi90": float(fc["ci_hi90"].iloc[h - 1]),
            })
    return pd.DataFrame(filas)
