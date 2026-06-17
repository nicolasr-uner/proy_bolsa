"""Ingesta de variables macro: TRM, Brent, PPI USA, ONI, IPP Colombia.

Fuentes confirmadas (jun 2026):
  TRM     — datos.gov.co Socrata, dataset 32sa-8pi3
  Brent   — FRED CSV, serie DCOILBRENTEU (diaria, USD/bbl)
  PPI USA — FRED CSV, serie PPIACO (mensual)
  ONI     — NOAA CPC, archivo ASCII fijo

Fuente pendiente:
  IPP Colombia — se intenta BanRep SDMX y datos.gov.co.
  Ejecutar scripts/validar_ipp.py para confirmar el endpoint antes de usar fetch_ipp().
"""

from __future__ import annotations

import datetime as dt
import io
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_TRM_URL = "https://www.datos.gov.co/resource/32sa-8pi3.json"
_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
_BANREP_SDMX = "https://suameca.banrep.gov.co/estadisticas-banrep/rest"

# Temporadas ONI -> mes central de la ventana de 3 meses
_SEAS_MES: dict[str, int] = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


# ---------------------------------------------------------------------------
# TRM
# ---------------------------------------------------------------------------

def fetch_trm(start: dt.date, end: dt.date) -> pd.DataFrame:
    """TRM diaria (COP/USD) desde datos.gov.co Socrata (32sa-8pi3).

    El portal publica rangos de vigencia (vigenciadesde/vigenciahasta) en vez de
    una fila por dia; se forward-fill a frecuencia diaria.

    Returns
    -------
    DataFrame con columnas: fecha (date), trm (float).
    """
    params = {
        "$limit": 10000,
        "$where": (
            f"vigenciadesde >= '{start.isoformat()}'"
            f" AND vigenciadesde <= '{end.isoformat()}'"
        ),
        "$order": "vigenciadesde ASC",
    }
    resp = requests.get(_TRM_URL, params=params, timeout=30)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        logger.warning("TRM: sin datos para %s — %s", start, end)
        return pd.DataFrame(columns=["fecha", "trm"])

    df = pd.DataFrame(rows)
    df["fecha"] = pd.to_datetime(df["vigenciadesde"]).dt.date
    df["trm"] = pd.to_numeric(df["valor"], errors="coerce")
    df = (
        df[["fecha", "trm"]]
        .dropna()
        .sort_values("fecha")
        .drop_duplicates("fecha", keep="last")
    )
    daily = (
        df.set_index(pd.to_datetime(df["fecha"]))["trm"]
        .resample("D")
        .last()
        .ffill()
    )
    mask = (daily.index.date >= start) & (daily.index.date <= end)
    return pd.DataFrame(
        {"fecha": daily.index[mask].date, "trm": daily[mask].values}
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# FRED (generico)
# ---------------------------------------------------------------------------

def _fred_serie(series_id: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Descarga CSV de FRED; devuelve DataFrame con 'fecha' (date) y el series_id."""
    url = _FRED_BASE.format(series=series_id)
    try:
        raw = pd.read_csv(url)
    except Exception as exc:
        raise RuntimeError(f"FRED {series_id}: {exc}") from exc
    raw.columns = ["fecha", "valor"]
    raw["fecha"] = pd.to_datetime(raw["fecha"]).dt.date
    raw["valor"] = pd.to_numeric(raw["valor"], errors="coerce")
    raw = raw.dropna(subset=["valor"])
    raw = raw[(raw["fecha"] >= start) & (raw["fecha"] <= end)]
    return raw.rename(columns={"valor": series_id}).reset_index(drop=True)


def fetch_brent(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Precio Brent diario (USD/bbl) desde FRED (DCOILBRENTEU).
    Returns: fecha (date), brent (float).
    """
    return _fred_serie("DCOILBRENTEU", start, end).rename(columns={"DCOILBRENTEU": "brent"})


def fetch_ppi_usa(start: dt.date, end: dt.date) -> pd.DataFrame:
    """PPI EE.UU. mensual (all commodities) desde FRED (PPIACO).
    Returns: fecha (date), ppi_usa (float).
    """
    return _fred_serie("PPIACO", start, end).rename(columns={"PPIACO": "ppi_usa"})


# ---------------------------------------------------------------------------
# ONI (ENSO)
# ---------------------------------------------------------------------------

def fetch_oni(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Indice ONI mensual (ENSO) desde NOAA CPC.

    La fecha corresponde al primer dia del mes central de la ventana de 3 meses.
    Ejemplo: DJF 2020 -> 2020-01-01, NDJ 2020 -> 2020-12-01.

    Returns: fecha (date), oni (float, anomalia de TSM Nino 3.4).
    """
    resp = requests.get(_ONI_URL, timeout=30)
    resp.raise_for_status()

    records = []
    for line in resp.text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] not in _SEAS_MES:
            continue
        try:
            records.append({"seas": parts[0], "anio": int(parts[1]), "oni": float(parts[3])})
        except ValueError:
            continue

    if not records:
        return pd.DataFrame(columns=["fecha", "oni"])

    df = pd.DataFrame(records)
    df["mes"] = df["seas"].map(_SEAS_MES)
    df["fecha"] = pd.to_datetime(
        {"year": df["anio"], "month": df["mes"], "day": 1}
    ).dt.date
    df = df[["fecha", "oni"]].sort_values("fecha")
    df = df[(df["fecha"] >= start) & (df["fecha"] <= end)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# IPP Colombia
# ---------------------------------------------------------------------------

def fetch_ipp(start: dt.date, end: dt.date) -> pd.DataFrame:
    """IPP Colombia mensual, Oferta Interna (base dic-2014 = 100).

    Intenta en orden: BanRep SDMX, datos.gov.co Socrata.
    Si ninguna responde, lanza RuntimeError con instrucciones para carga manual.

    Returns: fecha (date, 1er dia del mes), ipp (float, indice).
    """
    errores: list[str] = []

    for fn, nombre in [(_fetch_ipp_banrep_sdmx, "BanRep SDMX"), (_fetch_ipp_socrata, "datos.gov.co")]:
        try:
            df = fn(start, end)
            if not df.empty:
                logger.info("IPP obtenido desde %s (%d filas)", nombre, len(df))
                return df
        except NotImplementedError as exc:
            errores.append(f"{nombre}: {exc}")
        except Exception as exc:
            logger.debug("IPP %s fallo: %s", nombre, exc)
            errores.append(f"{nombre}: {exc}")

    raise RuntimeError(
        "IPP no disponible en ninguna fuente automatica.\n"
        "1. Ejecuta scripts/validar_ipp.py para encontrar el endpoint correcto.\n"
        "2. O descarga manualmente desde dane.gov.co y carga con:\n"
        "   pd.read_csv('data/raw/macro/ipp_manual.csv', parse_dates=['fecha'])\n"
        f"Errores acumulados: {'; '.join(errores)}"
    )


def _fetch_ipp_banrep_sdmx(start: dt.date, end: dt.date) -> pd.DataFrame:
    """IPP via BanRep SDMX. Los candidatos de URL se validan en scripts/validar_ipp.py."""
    url_candidates = [
        f"{_BANREP_SDMX}/data/1_1_IPP_OFIN_CO_M/",
        f"{_BANREP_SDMX}/data/1_1_IPP/M.IPP_OFIN../",
        f"{_BANREP_SDMX}/data/1_2_1_IPP_OFIN/",
    ]
    for url in url_candidates:
        try:
            params = {
                "startPeriod": start.strftime("%Y-%m"),
                "endPeriod": end.strftime("%Y-%m"),
                "format": "csvdata",
            }
            resp = requests.get(url, params=params, timeout=15)
            if not resp.ok:
                continue
            df = pd.read_csv(io.StringIO(resp.text))
            if "OBS_VALUE" in df.columns and "TIME_PERIOD" in df.columns:
                df["fecha"] = pd.to_datetime(df["TIME_PERIOD"].str[:7] + "-01").dt.date
                df["ipp"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
                result = df[["fecha", "ipp"]].dropna().sort_values("fecha").reset_index(drop=True)
                if not result.empty:
                    return result
        except Exception:
            continue
    raise RuntimeError("Ninguna URL candidata de BanRep SDMX respondio con datos IPP validos")


def _fetch_ipp_socrata(start: dt.date, end: dt.date) -> pd.DataFrame:
    """IPP desde datos.gov.co Socrata (dataset_id a confirmar con scripts/validar_ipp.py)."""
    IPP_DATASET_ID: str | None = None  # TODO: rellenar con el ID confirmado
    if IPP_DATASET_ID is None:
        raise NotImplementedError(
            "Dataset ID de IPP en datos.gov.co no confirmado. "
            "Corre scripts/validar_ipp.py."
        )
    raise NotImplementedError  # placeholder hasta confirmar ID


# ---------------------------------------------------------------------------
# IPP carga manual (fallback definitivo)
# ---------------------------------------------------------------------------

def load_ipp_local(
    path: str = "data/raw/macro/ipp_manual.csv",
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Carga el IPP desde un CSV local descargado manualmente de DANE.

    Formato esperado del CSV: dos columnas, fecha (YYYY-MM-DD o YYYY-MM) e ipp (float).
    Las filas sin encabezado valido se omiten. La primera fila debe ser el header.

    Ejemplo de descarga manual:
        1. Ir a https://www.dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/
           indice-de-precios-del-productor-ipp
        2. Descargar la serie "IPP Total Oferta Interna" como Excel o CSV.
        3. Conservar solo las columnas de fecha e indice, guardar como
           data/raw/macro/ipp_manual.csv con header: fecha,ipp

    Returns: fecha (date, 1er dia del mes), ipp (float).
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Archivo IPP no encontrado: {p.resolve()}\n"
            "Descarga la serie desde dane.gov.co y guarda como:\n"
            f"  {p.resolve()}\n"
            "Con formato CSV: fecha,ipp (una fila por mes, base dic-2014=100)"
        )
    df = pd.read_csv(p, header=0)
    # Normalizar nombres de columna (case-insensitive)
    df.columns = [c.strip().lower() for c in df.columns]

    # Encontrar columna de fecha (acepta 'fecha', 'date', 'periodo', 'mes', 'anio_mes')
    fecha_col = next(
        (c for c in df.columns if c in ("fecha", "date", "periodo", "mes", "anio_mes")),
        df.columns[0],
    )
    # Encontrar columna de valor (acepta 'ipp', 'valor', 'indice', 'ipp_ofin', 'oferta_interna')
    valor_col = next(
        (c for c in df.columns if c in ("ipp", "valor", "indice", "ipp_ofin", "oferta_interna")),
        df.columns[1],
    )
    df["fecha"] = pd.to_datetime(df[fecha_col], errors="coerce")
    # Normalizar a primer dia del mes
    df["fecha"] = df["fecha"].dt.to_period("M").dt.to_timestamp().dt.date
    df["ipp"] = pd.to_numeric(df[valor_col], errors="coerce")
    df = df[["fecha", "ipp"]].dropna().sort_values("fecha").reset_index(drop=True)

    if start is not None:
        df = df[df["fecha"] >= start]
    if end is not None:
        df = df[df["fecha"] <= end]
    return df.reset_index(drop=True)
