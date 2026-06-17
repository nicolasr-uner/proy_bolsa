"""Ingesta de datos del mercado electrico colombiano desde la API de XM.

Usa el paquete oficial `pydataxm`. La API entrega las metricas horarias en
formato ancho (columnas ``Values_Hour01..Values_Hour24`` + ``Date``); aqui se
ofrece la utilidad para pasarlas a serie horaria larga (``timestamp``, valor),
que es como las consumen los modelos.

Limite de la API: 31 dias por consulta en metricas horarias/diarias, por eso
``fetch_metric_range`` parte el rango en bloques.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterator

import pandas as pd

MAX_DIAS_POR_CONSULTA = 31


def _get_api():
    """Crea un cliente de la API de XM (import perezoso para no exigir red en tests)."""
    from pydataxm.pydataxm import ReadDB

    return ReadDB()


def horas_anchas_a_largo(df: pd.DataFrame, value_name: str = "valor") -> pd.DataFrame:
    """Convierte el formato ancho de 24 horas de XM a serie horaria larga.

    Devuelve las columnas identificadoras originales mas ``hora`` (0..23),
    ``timestamp`` (Date + hora) y ``value_name``.
    """
    hour_cols = [c for c in df.columns if c.lower().startswith("values_hour")]
    if not hour_cols:
        raise ValueError("El DataFrame no tiene columnas Values_HourNN")
    id_cols = [c for c in df.columns if c not in hour_cols]
    long = df.melt(
        id_vars=id_cols, value_vars=hour_cols, var_name="hora", value_name=value_name
    )
    # Values_Hour01 -> hora 0 (primera hora del dia).
    long["hora"] = long["hora"].str.extract(r"(\d+)").astype(int) - 1
    date_col = "Date" if "Date" in long.columns else id_cols[-1]
    long["timestamp"] = pd.to_datetime(long[date_col]) + pd.to_timedelta(
        long["hora"], unit="h"
    )
    return long.sort_values("timestamp").reset_index(drop=True)


def _chunks(start: dt.date, end: dt.date, paso: int) -> Iterator[tuple[dt.date, dt.date]]:
    cur = start
    while cur <= end:
        fin = min(cur + dt.timedelta(days=paso - 1), end)
        yield cur, fin
        cur = fin + dt.timedelta(days=1)


def fetch_metric(metric_id: str, entity: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Descarga una metrica de XM para un rango <= 31 dias."""
    return _get_api().request_data(metric_id, entity, start, end)


def fetch_metric_range(
    metric_id: str, entity: str, start: dt.date, end: dt.date
) -> pd.DataFrame:
    """Descarga una metrica para un rango arbitrario, partiendo en bloques de 31 dias."""
    partes = [
        fetch_metric(metric_id, entity, ini, fin)
        for ini, fin in _chunks(start, end, MAX_DIAS_POR_CONSULTA)
    ]
    partes = [p for p in partes if p is not None and not p.empty]
    if not partes:
        return pd.DataFrame()
    return pd.concat(partes, ignore_index=True)


def fetch_precio_bolsa_horario(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Serie horaria larga del Precio de Bolsa Nacional (COP/kWh)."""
    wide = fetch_metric_range("PrecBolsNaci", "Sistema", start, end)
    if wide.empty:
        return wide
    return horas_anchas_a_largo(wide, value_name="precio_bolsa")[["timestamp", "precio_bolsa"]]


def get_catalogo() -> pd.DataFrame:
    """Catalogo completo de metricas disponibles en la API de XM."""
    return _get_api().get_collections()
