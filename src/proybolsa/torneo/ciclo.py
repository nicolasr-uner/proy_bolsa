"""Orquesta un ciclo del torneo: puntear lo vencido, luego registrar el panel del origen.

El origen es el ultimo mes con IPP real en `df`. Los 'actuals' para puntear son la propia
serie de IPP de `df` (todos los meses observados). Diseñado con dir_salida parametrizable
para testear sin tocar outputs/ reales.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from proybolsa.torneo.evaluacion import agregar_leaderboard, resolver
from proybolsa.torneo.panel import HORIZONTES, pronosticar_panel
from proybolsa.torneo.registro import cargar_registro, registrar_panel

logger = logging.getLogger(__name__)


def correr_ciclo_torneo(df: pd.DataFrame, dir_salida: str | Path, fecha_run: str,
                        horizontes=HORIZONTES, errores_backtest: pd.DataFrame | None = None
                        ) -> None:
    dir_salida = Path(dir_salida)
    r_reg = dir_salida / "registro_ipp.parquet"
    r_res = dir_salida / "resueltos_ipp.parquet"
    r_lb = dir_salida / "leaderboard_ipp.parquet"

    df = df.sort_values("fecha").dropna(subset=["ipp"]).reset_index(drop=True)
    if df.empty:
        logger.warning("Torneo IPP: sin IPP real disponible; se omite el ciclo")
        return
    origen = df["fecha"].iloc[-1]

    # 1. PUNTEAR: resolver el registro previo contra los IPP reales conocidos hoy.
    reg_prev = cargar_registro(r_reg)
    if not reg_prev.empty:
        res = resolver(reg_prev, df[["fecha", "ipp"]])
        if not res.empty:
            dir_salida.mkdir(parents=True, exist_ok=True)
            res.to_parquet(r_res, index=False)
            agregar_leaderboard(res).to_parquet(r_lb, index=False)

    # 2. REGISTRAR: el panel del origen actual (drivers congelados).
    panel = pronosticar_panel(df, origen_fecha=origen, horizontes=horizontes,
                              errores_backtest=errores_backtest)
    registrar_panel(panel, origen_fecha=origen, fecha_run=fecha_run, ruta=r_reg)
