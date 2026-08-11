"""Actualiza la serie del IPP desde el anexo del DANE, sin intervención manual.

Reemplaza el paso manual del checklist mensual (bajar el .xlsx del portal a mano y correr
`parsear_ipp_dane.py`). El anexo del DANE se sirve en una URL estable, así que todo el
encadenamiento es automatizable:

    resolver anexo -> ¿cambió? -> descargar -> parsear -> diff -> validar -> escribir

Uso:
    .venv\\Scripts\\python scripts/actualizar_ipp.py
    .venv\\Scripts\\python scripts/actualizar_ipp.py --forzar        # ignora el skip por meta
    .venv\\Scripts\\python scripts/actualizar_ipp.py --dry-run       # no escribe nada
    .venv\\Scripts\\python scripts/actualizar_ipp.py --anexo 2026 7  # un anexo concreto

Salidas:
    data/raw/macro/ipp_manual.csv        serie [fecha,ipp]  (la que consume load_ipp_local)
    data/raw/macro/ipp_anexo_meta.json   huella del anexo, para el skip de la próxima corrida
    data/raw/macro/ipp_revisiones.csv    rastro acumulado de revisiones del DANE
    data/raw/macro/anexos/*.xlsx         el insumo descargado (NO se versiona)

Exit codes: 0 OK o sin cambios · 1 fallo de red/resolución · 2 la validación rechazó la serie.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from proybolsa.ingest.dane_ipp import (  # noqa: E402
    AnexoIPP,
    ValidacionIPP,
    descargar_anexo,
    diff_series_ipp,
    parsear_excel_ipp,
    resolver_anexo_mas_reciente,
    sha256,
    url_anexo,
    validar_serie_ipp,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("actualizar_ipp")

RAW_MACRO = ROOT / "data" / "raw" / "macro"
CSV_IPP = RAW_MACRO / "ipp_manual.csv"
META = RAW_MACRO / "ipp_anexo_meta.json"
REVISIONES = RAW_MACRO / "ipp_revisiones.csv"
DIR_ANEXOS = RAW_MACRO / "anexos"


def _escribir_atomico(df: pd.DataFrame, destino: Path) -> None:
    """Escritura atómica: un proceso muerto a mitad de camino no deja un CSV truncado.

    Sin esto, un fallo durante la escritura en CI deja la serie corrupta Y commiteada.
    """
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, destino)


def _leer_meta() -> dict:
    if not META.exists():
        return {}
    try:
        return json.loads(META.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("No se pudo leer %s (%s); se procede como si no existiera", META.name, exc)
        return {}


def _sin_cambios(anexo: AnexoIPP, meta: dict) -> bool:
    """¿El anexo remoto es el mismo que ya procesamos?

    Se compara la terna (url, last_modified, content_length). Si coincide y el CSV existe, no
    hay nada que hacer: ni descargar. Es lo que hace barato correr esto semanalmente.
    """
    if not meta or not CSV_IPP.exists():
        return False
    return (
        meta.get("url") == anexo.url
        and meta.get("last_modified") == anexo.last_modified
        and meta.get("content_length") == anexo.content_length
    )


def _registrar_revisiones(revisadas: pd.DataFrame, periodo: str) -> None:
    """Acumula las revisiones del DANE en un CSV, con el anexo que las trajo."""
    if revisadas.empty:
        return
    filas = revisadas.copy()
    filas.insert(0, "anexo", periodo)
    filas.insert(0, "detectado", dt.date.today().isoformat())
    if REVISIONES.exists():
        previo = pd.read_csv(REVISIONES)
        filas = pd.concat([previo, filas], ignore_index=True)
    _escribir_atomico(filas, REVISIONES)
    logger.info("Revisiones registradas en %s", REVISIONES.name)


def main() -> int:
    p = argparse.ArgumentParser(description="Actualiza el IPP desde el anexo del DANE.")
    p.add_argument("--forzar", action="store_true",
                   help="Descarga y reprocesa aunque la huella del anexo no haya cambiado.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resuelve, descarga, valida y reporta, pero no escribe el CSV.")
    p.add_argument("--anexo", nargs=2, type=int, metavar=("ANIO", "MES"),
                   help="Usar un anexo concreto en vez de resolver el más reciente.")
    p.add_argument("--serie", default="oferta_interna", choices=["oferta_interna", "prod_nacional"],
                   help="Agregado del IPP (default: oferta_interna, el target del modelo).")
    p.add_argument("--max-retroceso", type=int, default=6,
                   help="Meses hacia atrás a probar al resolver el anexo (default 6).")
    args = p.parse_args()

    # --- 1. Resolver el anexo ---
    try:
        if args.anexo:
            anio, mes = args.anexo
            from proybolsa.ingest.http import head
            url = url_anexo(anio, mes)
            cab = head(url)
            if not cab.ok:
                logger.error("El anexo %d-%02d no existe (HTTP %d): %s", anio, mes, cab.status, url)
                return 1
            anexo = AnexoIPP(url=url, anio=anio, mes=mes,
                             last_modified=cab.last_modified, content_length=cab.content_length)
        else:
            anexo = resolver_anexo_mas_reciente(max_retroceso=args.max_retroceso)
    except Exception as exc:
        logger.error("No se pudo resolver el anexo del IPP: %s", exc)
        return 1

    meta_previa = _leer_meta()
    if _sin_cambios(anexo, meta_previa) and not args.forzar:
        logger.info("Sin cambios: el anexo %s ya fue procesado (%s meses hasta %s). "
                    "No se descarga nada. Use --forzar para reprocesar.",
                    anexo.periodo, meta_previa.get("n_meses"), meta_previa.get("ultimo_mes"))
        return 0

    # --- 2. Descargar y parsear ---
    destino = DIR_ANEXOS / f"anex-IPP-historicos-{anexo.periodo}.xlsx"
    try:
        descargar_anexo(anexo, destino)
        nuevo = parsear_excel_ipp(destino, serie=args.serie)
    except Exception as exc:
        logger.error("Fallo al descargar/parsear el anexo %s: %s", anexo.periodo, exc)
        return 1

    # --- 3. Diff contra la serie vigente ---
    viejo = None
    if CSV_IPP.exists():
        viejo = pd.read_csv(CSV_IPP, parse_dates=["fecha"])
    nuevas, revisadas = diff_series_ipp(viejo, nuevo)

    if viejo is None:
        logger.info("Primera carga: %d meses", len(nuevo))
    else:
        logger.info("Meses nuevos: %d | meses revisados por el DANE: %d",
                    len(nuevas), len(revisadas))
        for _, r in nuevas.iterrows():
            logger.info("  NUEVO   %s: %.2f", r["fecha"].date(), r["ipp"])
        for _, r in revisadas.iterrows():
            logger.warning("  REVISION %s: %.2f -> %.2f (%+.2f%%)",
                           r["fecha"].date(), r["ipp_viejo"], r["ipp_nuevo"], r["dif_pct"])
        if nuevas.empty and revisadas.empty:
            logger.info("El anexo cambió de huella pero la serie es idéntica.")

    # --- 4. Validar ANTES de escribir ---
    try:
        validar_serie_ipp(nuevo, viejo=viejo)
    except ValidacionIPP as exc:
        logger.error("%s", exc)
        logger.error("La serie vigente (%s) queda intacta.", CSV_IPP.name)
        return 2

    if args.dry_run:
        logger.info("--dry-run: validación OK, no se escribió nada.")
        return 0

    # --- 5. Escribir serie + rastros ---
    salida = nuevo.copy()
    salida["fecha"] = pd.to_datetime(salida["fecha"]).dt.strftime("%Y-%m-%d")
    _escribir_atomico(salida[["fecha", "ipp"]], CSV_IPP)
    logger.info("Escrito %s (%d meses, %d bytes)", CSV_IPP.name, len(salida), CSV_IPP.stat().st_size)

    _registrar_revisiones(revisadas, anexo.periodo)

    META.write_text(json.dumps({
        "url": anexo.url,
        "periodo": anexo.periodo,
        "anio": anexo.anio,
        "mes": anexo.mes,
        "last_modified": anexo.last_modified,
        "content_length": anexo.content_length,
        "sha256": sha256(destino),
        "fecha_descarga": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "serie": args.serie,
        "n_meses": int(len(salida)),
        "ultimo_mes": str(pd.to_datetime(nuevo["fecha"]).max().date()),
        "n_nuevas": int(len(nuevas)),
        "n_revisadas": int(len(revisadas)),
    }, indent=2), encoding="utf-8")
    logger.info("Huella guardada en %s", META.name)

    logger.info("Siguiente paso: .venv\\Scripts\\python scripts/construir_features.py --solo-ipp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
