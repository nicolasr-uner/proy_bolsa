"""Descarga y guarda el historico completo de datos para los modelos.

Descarga >= 3 anos de datos desde las fuentes confirmadas y los guarda en Parquet
bajo data/raw/. Se puede re-ejecutar; si el archivo ya existe no se sobreescribe
a menos que se pase --forzar.

Metricas que se descargan
--------------------------
XM (precio, hidrologia, demanda):
  - precio_bolsa_horario   (PrecBolsNaci, horario)
  - precio_escasez         (PrecEsca, diario)
  - aportes_diarios        (PorcApor + AporEner)
  - embalses_diarios       (PorcVoluUtilDiar + VoluUtilDiarEner + CapaUtilDiarEner)
  - vertimientos           (VertEner, Embalse, suma diaria)
  - demanda_diaria         (DemaSIN)
  - demanda_upme_medio     (EscDemUPMEMedio, para horizonte largo)

Macro:
  - trm_diaria             (datos.gov.co)
  - brent_diario           (FRED DCOILBRENTEU)
  - ppi_usa_mensual        (FRED PPIACO)
  - oni_mensual            (NOAA CPC)
  - ipp_mensual            (BanRep SDMX / manual) — puede fallar; ver validar_ipp.py

Uso
---
    .venv/Scripts/python scripts/descarga_historico.py
    .venv/Scripts/python scripts/descarga_historico.py --forzar
    .venv/Scripts/python scripts/descarga_historico.py --inicio 2021-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

# ---- setup de logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_XM = DATA_RAW / "xm"
DATA_MACRO = DATA_RAW / "macro"


def _guardar_parquet(df: pd.DataFrame, ruta: Path, nombre: str) -> None:
    if df.empty:
        logger.warning("  %s: DataFrame vacio, no se guarda", nombre)
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    logger.info("  Guardado: %s (%d filas)", ruta.relative_to(ROOT), len(df))


def _ya_existe(ruta: Path, forzar: bool) -> bool:
    if ruta.exists() and not forzar:
        logger.info("  Existe (--forzar para re-descargar): %s", ruta.relative_to(ROOT))
        return True
    return False


def descargar_xm(inicio: dt.date, fin: dt.date, forzar: bool) -> None:
    logger.info("=== XM (%s — %s) ===", inicio, fin)
    from proybolsa.ingest.xm import (
        fetch_aportes_diarios,
        fetch_demanda_diaria,
        fetch_demanda_upme,
        fetch_embalses_diarios,
        fetch_precio_bolsa_horario,
        fetch_precio_escasez,
        fetch_vertimientos_diarios,
    )

    tareas = [
        (
            DATA_XM / "precio_bolsa_horario.parquet",
            "precio_bolsa_horario",
            lambda: fetch_precio_bolsa_horario(inicio, fin),
        ),
        (
            DATA_XM / "precio_escasez.parquet",
            "precio_escasez",
            lambda: fetch_precio_escasez(inicio, fin),
        ),
        (
            DATA_XM / "aportes_diarios.parquet",
            "aportes_diarios",
            lambda: fetch_aportes_diarios(inicio, fin),
        ),
        (
            DATA_XM / "embalses_diarios.parquet",
            "embalses_diarios",
            lambda: fetch_embalses_diarios(inicio, fin),
        ),
        (
            DATA_XM / "vertimientos_diarios.parquet",
            "vertimientos_diarios",
            lambda: fetch_vertimientos_diarios(inicio, fin),
        ),
        (
            DATA_XM / "demanda_diaria.parquet",
            "demanda_diaria",
            lambda: fetch_demanda_diaria(inicio, fin),
        ),
        (
            DATA_XM / "demanda_upme_medio.parquet",
            "demanda_upme_medio",
            lambda: fetch_demanda_upme(inicio, fin, scenario="Medio"),
        ),
    ]

    for ruta, nombre, fn in tareas:
        if _ya_existe(ruta, forzar):
            continue
        logger.info("  Descargando %s ...", nombre)
        try:
            df = fn()
            _guardar_parquet(df, ruta, nombre)
        except Exception as exc:
            logger.error("  ERROR en %s: %s", nombre, exc)


def descargar_macro(inicio: dt.date, fin: dt.date, forzar: bool) -> None:
    logger.info("=== Macro (%s — %s) ===", inicio, fin)
    from proybolsa.ingest.macros import (
        fetch_brent,
        fetch_ipp,
        fetch_oni,
        fetch_ppi_usa,
        fetch_trm,
    )

    tareas: list[tuple] = [
        (DATA_MACRO / "trm_diaria.parquet", "trm_diaria", lambda: fetch_trm(inicio, fin)),
        (DATA_MACRO / "brent_diario.parquet", "brent_diario", lambda: fetch_brent(inicio, fin)),
        (
            DATA_MACRO / "ppi_usa_mensual.parquet",
            "ppi_usa_mensual",
            lambda: fetch_ppi_usa(inicio, fin),
        ),
        (DATA_MACRO / "oni_mensual.parquet", "oni_mensual", lambda: fetch_oni(inicio, fin)),
    ]

    for ruta, nombre, fn in tareas:
        if _ya_existe(ruta, forzar):
            continue
        logger.info("  Descargando %s ...", nombre)
        try:
            df = fn()
            _guardar_parquet(df, ruta, nombre)
        except Exception as exc:
            logger.error("  ERROR en %s: %s", nombre, exc)

    # IPP: puede fallar; registrar instrucciones si falla
    ruta_ipp = DATA_MACRO / "ipp_mensual.parquet"
    if not _ya_existe(ruta_ipp, forzar):
        logger.info("  Descargando ipp_mensual ...")
        try:
            df_ipp = fetch_ipp(inicio, fin)
            _guardar_parquet(df_ipp, ruta_ipp, "ipp_mensual")
        except RuntimeError as exc:
            logger.warning(
                "  ipp_mensual no disponible automaticamente.\n"
                "  Ejecuta scripts/validar_ipp.py para encontrar el endpoint,\n"
                "  o carga manualmente y guarda en: %s\n"
                "  Detalle: %s",
                ruta_ipp,
                exc,
            )


def resumen(inicio: dt.date, fin: dt.date) -> None:
    logger.info("=== Resumen de archivos descargados ===")
    total = 0
    for f in sorted((DATA_XM.glob("*.parquet"), DATA_MACRO.glob("*.parquet")), key=str):
        pass
    for directorio, label in [(DATA_XM, "XM"), (DATA_MACRO, "Macro")]:
        archivos = sorted(directorio.glob("*.parquet"))
        if not archivos:
            logger.info("  [%s] (vacio)", label)
            continue
        for f in archivos:
            df = pd.read_parquet(f)
            logger.info("  [%s] %s: %d filas", label, f.name, len(df))
            total += len(df)
    logger.info("  Total filas: %d", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga historico de datos para modelos.")
    parser.add_argument(
        "--inicio",
        default=str((dt.date.today() - dt.timedelta(days=3 * 365)),),
        help="Fecha de inicio (YYYY-MM-DD). Default: hace 3 anos.",
    )
    parser.add_argument(
        "--fin",
        default=str(dt.date.today() - dt.timedelta(days=1)),
        help="Fecha de fin (YYYY-MM-DD). Default: ayer.",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Re-descargar aunque el archivo ya exista.",
    )
    parser.add_argument(
        "--solo-macro",
        action="store_true",
        help="Descargar solo variables macro (no XM).",
    )
    parser.add_argument(
        "--solo-xm",
        action="store_true",
        help="Descargar solo metricas XM (no macro).",
    )
    args = parser.parse_args()

    inicio = dt.date.fromisoformat(args.inicio)
    fin = dt.date.fromisoformat(args.fin)

    logger.info("Descarga historica: %s — %s", inicio, fin)
    logger.info("Directorio datos: %s", DATA_RAW)

    if not args.solo_macro:
        descargar_xm(inicio, fin, args.forzar)

    if not args.solo_xm:
        descargar_macro(inicio, fin, args.forzar)

    resumen(inicio, fin)
    logger.info("Listo.")


if __name__ == "__main__":
    main()
