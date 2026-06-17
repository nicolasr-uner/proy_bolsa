"""Ingesta de datos del mercado electrico colombiano desde la API de XM.

Usa el paquete oficial `pydataxm`. La API entrega las metricas horarias en
formato ancho (columnas Values_Hour01..Values_Hour24 + Date); aqui se ofrece
la utilidad para pasarlas a serie horaria larga (timestamp, valor).

Limite de la API: 31 dias por consulta en metricas horarias/diarias, por eso
`fetch_metric_range` parte el rango en bloques.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Iterator, Literal

import pandas as pd

logger = logging.getLogger(__name__)

MAX_DIAS_POR_CONSULTA = 31


def _get_api():
    """Crea un cliente de la API de XM (import perezoso para no exigir red en tests)."""
    from pydataxm.pydataxm import ReadDB

    return ReadDB()


# ---------------------------------------------------------------------------
# Transformaciones de formato
# ---------------------------------------------------------------------------

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
    # Values_Hour01 -> hora 0 (primera hora del dia = medianoche).
    long["hora"] = long["hora"].str.extract(r"(\d+)").astype(int) - 1
    date_col = "Date" if "Date" in long.columns else id_cols[-1]
    long["timestamp"] = pd.to_datetime(long[date_col]) + pd.to_timedelta(
        long["hora"], unit="h"
    )
    return long.sort_values("timestamp").reset_index(drop=True)


def _a_diario(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Normaliza el output de pydataxm a serie diaria con columnas [fecha, col_name].

    Maneja tres formatos que devuelve la API:
    - 24 columnas Values_Hour01..24 (metrica horaria; se promedia a diario)
    - 1 columna Values_* (metrica diaria estilo antiguo)
    - Columna 'Value' (metrica diaria estilo pydataxm >= 0.3.x — 'Id','Value','Date')
    """
    if df.empty:
        return pd.DataFrame(columns=["fecha", col_name])

    date_col = next((c for c in df.columns if c.lower() == "date"), df.columns[0])
    hour_cols = [c for c in df.columns if c.lower().startswith("values_hour")]
    val_cols = [c for c in df.columns if c.lower().startswith("values_")]
    # Columna 'Value' (singular, sin prefijo) — formato de metricas diarias en pydataxm
    plain_val = next((c for c in df.columns if c == "Value"), None)

    df = df.copy()
    df["fecha"] = pd.to_datetime(df[date_col]).dt.date

    if len(hour_cols) == 24:
        df[col_name] = df[hour_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    elif plain_val is not None:
        df[col_name] = pd.to_numeric(df[plain_val], errors="coerce")
    elif len(val_cols) == 1:
        df[col_name] = pd.to_numeric(df[val_cols[0]], errors="coerce")
    else:
        raise ValueError(
            f"Formato inesperado para metrica diaria '{col_name}': "
            f"columnas={list(df.columns)}"
        )

    return (
        df[["fecha", col_name]]
        .dropna(subset=[col_name])
        .sort_values("fecha")
        .drop_duplicates("fecha", keep="last")
        .reset_index(drop=True)
    )


def _suma_por_dia(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Agrega multiples filas por dia (ej: metricas por Embalse/Recurso) sumando.

    Maneja columna 'Value' (estilo pydataxm diario) y columnas 'Values_*' (estilo antiguo).
    """
    if df.empty:
        return pd.DataFrame(columns=["fecha", col_name])

    date_col = next((c for c in df.columns if c.lower() == "date"), df.columns[0])
    val_cols = [c for c in df.columns if c.lower().startswith("values_")]
    plain_val = next((c for c in df.columns if c == "Value"), None)

    if plain_val is None and not val_cols:
        raise ValueError(f"Sin columna de valor para '{col_name}': {list(df.columns)}")
    val_col = plain_val if plain_val is not None else val_cols[0]

    df = df.copy()
    df["fecha"] = pd.to_datetime(df[date_col]).dt.date
    df[col_name] = pd.to_numeric(df[val_col], errors="coerce")
    result = df.groupby("fecha")[col_name].sum().reset_index()
    return result.sort_values("fecha").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Descarga basica
# ---------------------------------------------------------------------------

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
    partes = []
    chunks = list(_chunks(start, end, MAX_DIAS_POR_CONSULTA))
    for i, (ini, fin) in enumerate(chunks, 1):
        logger.debug(
            "  %s [%s] chunk %d/%d: %s — %s", metric_id, entity, i, len(chunks), ini, fin
        )
        parte = fetch_metric(metric_id, entity, ini, fin)
        if parte is not None and not parte.empty:
            partes.append(parte)
    if not partes:
        return pd.DataFrame()
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------
# Precio de bolsa
# ---------------------------------------------------------------------------

def fetch_precio_bolsa_horario(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Serie horaria larga del Precio de Bolsa Nacional (COP/kWh).
    Returns: timestamp (datetime64), precio_bolsa (float).
    """
    wide = fetch_metric_range("PrecBolsNaci", "Sistema", start, end)
    if wide.empty:
        return pd.DataFrame(columns=["timestamp", "precio_bolsa"])
    return horas_anchas_a_largo(wide, value_name="precio_bolsa")[["timestamp", "precio_bolsa"]]


def fetch_precio_escasez(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Precio de escasez diario (COP/kWh). Techo regulatorio del precio de bolsa.
    Returns: fecha (date), precio_escasez (float).
    """
    wide = fetch_metric_range("PrecEsca", "Sistema", start, end)
    if wide.empty:
        return pd.DataFrame(columns=["fecha", "precio_escasez"])
    return _a_diario(wide, "precio_escasez")


# ---------------------------------------------------------------------------
# Hidrologia
# ---------------------------------------------------------------------------

def fetch_aportes_diarios(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Aportes hidricos diarios del sistema.
    Returns: fecha (date), aportes_pct (%), aportes_energia (kWh).
    """
    dfs = []
    for metric_id, col in [("PorcApor", "aportes_pct"), ("AporEner", "aportes_energia")]:
        wide = fetch_metric_range(metric_id, "Sistema", start, end)
        if not wide.empty:
            dfs.append(_a_diario(wide, col))

    if not dfs:
        return pd.DataFrame(columns=["fecha", "aportes_pct", "aportes_energia"])
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="fecha", how="outer")
    return result.sort_values("fecha").reset_index(drop=True)


def fetch_embalses_diarios(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Estado de embalses diario del sistema.
    Returns: fecha, volumen_util_pct (%), volumen_util_energia (kWh), capacidad_util_energia (kWh).
    """
    specs = [
        ("PorcVoluUtilDiar", "volumen_util_pct"),
        ("VoluUtilDiarEner", "volumen_util_energia"),
        ("CapaUtilDiarEner", "capacidad_util_energia"),
    ]
    dfs = []
    for metric_id, col in specs:
        wide = fetch_metric_range(metric_id, "Sistema", start, end)
        if not wide.empty:
            dfs.append(_a_diario(wide, col))

    if not dfs:
        cols = ["fecha"] + [c for _, c in specs]
        return pd.DataFrame(columns=cols)
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on="fecha", how="outer")
    return result.sort_values("fecha").reset_index(drop=True)


def fetch_vertimientos_diarios(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Vertimientos diarios (energia no turbinada), suma de todos los embalses.
    Returns: fecha (date), vertimientos (kWh).
    """
    wide = fetch_metric_range("VertEner", "Embalse", start, end)
    if wide.empty:
        return pd.DataFrame(columns=["fecha", "vertimientos"])
    return _suma_por_dia(wide, "vertimientos")


# ---------------------------------------------------------------------------
# Demanda
# ---------------------------------------------------------------------------

def fetch_demanda_diaria(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Demanda SIN diaria del sistema (kWh).
    Returns: fecha (date), demanda_sin (kWh).
    """
    wide = fetch_metric_range("DemaSIN", "Sistema", start, end)
    if wide.empty:
        return pd.DataFrame(columns=["fecha", "demanda_sin"])
    return _a_diario(wide, "demanda_sin")


def fetch_demanda_horaria(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Demanda real horaria del sistema (kWh).
    Returns: timestamp (datetime64), demanda_real (kWh).
    """
    wide = fetch_metric_range("DemaReal", "Sistema", start, end)
    if wide.empty:
        return pd.DataFrame(columns=["timestamp", "demanda_real"])
    return horas_anchas_a_largo(wide, value_name="demanda_real")[["timestamp", "demanda_real"]]


# ---------------------------------------------------------------------------
# Proyecciones UPME (exogenas para horizonte largo / PPAs)
# ---------------------------------------------------------------------------

def fetch_demanda_upme(
    start: dt.date,
    end: dt.date,
    scenario: Literal["Alto", "Medio", "Bajo"] = "Medio",
) -> pd.DataFrame:
    """Proyeccion oficial UPME de demanda (mensual, kWh).
    Returns: fecha (date), demanda_upme (kWh).
    """
    metric_id = f"EscDemUPME{scenario}"
    wide = fetch_metric_range(metric_id, "Sistema", start, end)
    if wide.empty:
        return pd.DataFrame(columns=["fecha", "demanda_upme"])
    return _a_diario(wide, "demanda_upme")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def get_catalogo() -> pd.DataFrame:
    """Catalogo completo de metricas disponibles en la API de XM (193 metricas)."""
    return _get_api().get_collections()
