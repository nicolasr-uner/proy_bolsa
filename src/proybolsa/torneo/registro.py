"""Registro append-only de los pronosticos publicados por el panel de modelos.

Una fila por (origen_fecha, modelo, horizonte). Idempotente: registrar dos veces el mismo
origen no duplica ni pisa lo ya escrito, para que re-correr un ciclo mensual sea seguro.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNAS = [
    "origen_fecha", "modelo", "horizonte", "fecha_objetivo",
    "y_pred", "ci_lo90", "ci_hi90", "fecha_run",
]
_CLAVE = ["origen_fecha", "modelo", "horizonte"]


def registrar_panel(panel_df: pd.DataFrame, origen_fecha, fecha_run: str,
                    ruta: str | Path) -> pd.DataFrame:
    """Agrega el panel de un origen al registro. No duplica (origen, modelo, horizonte)."""
    nuevo = panel_df.copy()
    nuevo["origen_fecha"] = pd.Timestamp(origen_fecha)
    nuevo["fecha_run"] = str(fecha_run)
    nuevo = nuevo[COLUMNAS]

    ruta = Path(ruta)
    if ruta.exists():
        prev = pd.read_parquet(ruta)
        ya = set(map(tuple, prev[_CLAVE].astype(str).to_numpy()))
        mask = [tuple(r) not in ya for r in nuevo[_CLAVE].astype(str).to_numpy()]
        combinado = pd.concat([prev, nuevo[mask]], ignore_index=True)
    else:
        combinado = nuevo

    ruta.parent.mkdir(parents=True, exist_ok=True)
    combinado.to_parquet(ruta, index=False)
    return combinado


def cargar_registro(ruta: str | Path) -> pd.DataFrame:
    ruta = Path(ruta)
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS)
    return pd.read_parquet(ruta)
