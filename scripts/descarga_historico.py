"""Descarga incremental de las series crudas hacia data/raw/.

QUÉ CAMBIÓ Y POR QUÉ
--------------------
La versión anterior tenía dos funciones que juntas hacían imposible actualizar datos:

    _ya_existe(ruta, forzar)   ->  si el parquet existe y no hay --forzar, no hagas nada
    _guardar_parquet(df, ruta) ->  to_parquet() del archivo completo

O sea: o se re-bajaban 3 años enteros con `--forzar`, o no se bajaba nada. Y como
`run_monthly_update.py` invocaba este script **sin** `--forzar`, el paso de descarga era un
no-op que registraba "OK" mientras las series envejecían. Así quedaron congeladas desde el
17-jun-2026: Brent 30% desviado del real, TRM 10%, y `brent_cop` —el driver principal del
IPP— 37%.

Ahora cada serie declara su ventana de revisión en `proybolsa.ingest.series` y se pide
`(última_fecha − ventana, ayer)`. El upsert fusiona, deduplica dejando ganar al dato nuevo, y
reporta cuántas filas son nuevas y cuántas son revisiones de valores ya publicados.

Uso
---
    .venv/Scripts/python scripts/descarga_historico.py
    .venv/Scripts/python scripts/descarga_historico.py --modo completo
    .venv/Scripts/python scripts/descarga_historico.py --series macro --resync-dias 90
    .venv/Scripts/python scripts/descarga_historico.py --fail-si-atrasado

El IPP no se descarga aquí: su adquisición no es un rango de fechas sino "el anexo publicado
más reciente" (que trae la historia completa). Lo hace `scripts/actualizar_ipp.py`.

Exit codes: 0 OK · 1 alguna serie falló · 2 hay series atrasadas y se pidió --fail-si-atrasado.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from proybolsa.ingest.series import SerieSpec, resolver  # noqa: E402
from proybolsa.ingest.store import (  # noqa: E402
    ResultadoUpsert,
    estado_series,
    rango_a_descargar,
    upsert_parquet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("descarga")

DATA_RAW = ROOT / "data" / "raw"
ESTADO = ROOT / "outputs" / "estado_datos.json"


def descargar_serie(spec: SerieSpec, *, modo: str, resync_dias: int,
                    hoy: dt.date) -> ResultadoUpsert | None:
    """Descarga y fusiona una serie. Devuelve None si no había nada por hacer."""
    rango = rango_a_descargar(spec, DATA_RAW, hoy, modo=modo, resync_dias=resync_dias)
    if rango is None:
        logger.info("  %s: al día, nada por pedir", spec.nombre)
        return None

    desde, hasta = rango
    logger.info("  %s: pidiendo %s -> %s", spec.nombre, desde, hasta)
    df = spec.fetcher()(desde, hasta, **spec.kwargs_fetcher)

    if df is None or df.empty:
        # No se traga el vacío en silencio: es la diferencia entre "no hay dato nuevo" y
        # "la fuente respondió mal", y el llamador necesita poder distinguirlo.
        raise RuntimeError(
            f"{spec.nombre}: la fuente devolvió 0 filas para {desde} -> {hasta}"
        )
    return upsert_parquet(df, spec, DATA_RAW)


def main() -> int:
    p = argparse.ArgumentParser(description="Descarga incremental de series crudas.")
    p.add_argument("--modo", choices=["incremental", "completo"], default="incremental",
                   help="incremental (default) re-baja solo la cola de revisión de cada serie.")
    p.add_argument("--forzar", action="store_true",
                   help="Alias histórico de --modo completo.")
    p.add_argument("--resync-dias", type=int, default=0,
                   help="Amplía la cola re-descargada, para recuperar un atraso puntual.")
    p.add_argument("--series", default=None,
                   help="Selección: 'xm', 'macro' o una lista 'trm_diaria,brent_diario'.")
    p.add_argument("--solo-macro", action="store_true", help="Atajo de --series macro.")
    p.add_argument("--solo-xm", action="store_true", help="Atajo de --series xm.")
    p.add_argument("--fail-si-atrasado", action="store_true",
                   help="Exit 2 si al terminar alguna serie sigue fuera de su tolerancia de "
                        "frescura. Es lo que convierte una congelación silenciosa en un fallo.")
    p.add_argument("--hoy", default=None, help="Fecha de referencia (YYYY-MM-DD), para pruebas.")
    args = p.parse_args()

    modo = "completo" if args.forzar else args.modo
    seleccion = args.series
    if args.solo_macro:
        seleccion = "macro"
    elif args.solo_xm:
        seleccion = "xm"

    hoy = dt.date.fromisoformat(args.hoy) if args.hoy else dt.date.today()

    try:
        specs = resolver(seleccion)
    except KeyError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Descarga %s de %d series (referencia: %s)", modo, len(specs), hoy)

    resultados: dict[str, ResultadoUpsert | None] = {}
    fallidas: dict[str, str] = {}

    for spec in specs:
        try:
            resultados[spec.nombre] = descargar_serie(
                spec, modo=modo, resync_dias=args.resync_dias, hoy=hoy
            )
        except Exception as exc:
            # Una fuente caída no debe abortar las otras 10: se registra y se sigue, y el
            # exit code al final refleja que hubo fallos.
            logger.error("  %s: FALLO -> %s", spec.nombre, exc)
            fallidas[spec.nombre] = str(exc)

    # --- Resumen ---
    logger.info("=== Resumen ===")
    nuevas_tot = revisadas_tot = 0
    for nombre, res in resultados.items():
        if res is None:
            continue
        nuevas_tot += res.filas_nuevas
        revisadas_tot += res.filas_revisadas
        if res.filas_revisadas:
            logger.warning("  %s: %d valores ya publicados cambiaron de valor",
                           nombre, res.filas_revisadas)
    logger.info("  %d filas nuevas, %d revisadas, %d series con fallo",
                nuevas_tot, revisadas_tot, len(fallidas))

    # --- Estado de frescura ---
    estado = estado_series(DATA_RAW, hoy)
    ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ESTADO.write_text(json.dumps({
        "generado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "referencia": hoy.isoformat(),
        "modo": modo,
        "series": estado.to_dict(orient="records"),
        "fallidas": fallidas,
    }, indent=2, default=str), encoding="utf-8")
    logger.info("  Estado escrito en %s", ESTADO.relative_to(ROOT))

    atrasadas = estado[~estado["frescura_ok"]]
    if not atrasadas.empty:
        for _, r in atrasadas.iterrows():
            logger.warning("  ATRASADA %s: ultimo=%s (%s dias, tolerancia %s)",
                           r["serie"], r["ultima_fecha"], r["dias_atraso"], r["tolerancia_dias"])

    if fallidas:
        return 1
    if args.fail_si_atrasado and not atrasadas.empty:
        logger.error("%d series fuera de tolerancia de frescura.", len(atrasadas))
        return 2
    logger.info("Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
