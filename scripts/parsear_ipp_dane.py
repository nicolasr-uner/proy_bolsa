"""Convierte el Excel DANE de IPP historico al CSV requerido por el modelo.

Uso:
    .venv\\Scripts\\python scripts\\parsear_ipp_dane.py anex-IPP-historicos-may2026.xlsx
    .venv\\Scripts\\python scripts\\parsear_ipp_dane.py anex-IPP-historicos-jun2026.xlsx --serie oferta_interna

El Excel de DANE tiene el formato:
  - Filas 4-5: cabeceras (grupos y sub-sectores). El nivel "Produccion Nacional"
    esta en una columna y "Oferta Interna" en otra.
  - Fila 6+: datos, col A=anio (solo primera fila del anio), col B=mes (nombre ES),
    y una columna por agregado/sector.
  - Base: Diciembre 2014 = 100

El target del modelo es **Oferta Interna** (lo que se consume internamente: producido +
importado; anclado a TRM y precios internacionales), tal como documenta
config/variables_ipp.yaml y el esquema pandera. La columna se detecta por NOMBRE de
cabecera para no depender de un indice fijo (DANE puede reordenar columnas).

Salida: data/raw/macro/ipp_manual.csv  (fecha,ipp)
"""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

import openpyxl
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

MESES_ES = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12,
}

# Series disponibles en el Excel DANE. `objetivo` es el texto de cabecera normalizado
# (sin acentos, sin marcas de nota al pie); `fallback_idx` es el indice de columna
# conocido (0-based) si la deteccion por cabecera falla.
SERIES = {
    "oferta_interna": {"objetivo": "oferta interna",      "fallback_idx": 6, "etiqueta": "Oferta Interna"},
    "prod_nacional":  {"objetivo": "produccion nacional", "fallback_idx": 2, "etiqueta": "Produccion Nacional"},
}


def _norm(valor: object) -> str:
    """Normaliza texto de cabecera: sin acentos, casefold, sin notas al pie ni simbolos."""
    if valor is None:
        return ""
    s = unicodedata.normalize("NFKD", str(valor))
    s = "".join(c for c in s if not unicodedata.combining(c))      # quitar acentos
    s = s.casefold()
    s = "".join(c if (c.isalpha() or c.isspace()) else " " for c in s)  # quitar digitos/simbolos de nota
    return " ".join(s.split())


def _detectar_columna(rows: list[tuple], serie: str) -> int:
    """Encuentra el indice de columna del agregado `serie` por nombre de cabecera.

    Escanea las primeras 8 filas (cabeceras). Devuelve el fallback conocido si no
    encuentra una coincidencia exacta.
    """
    objetivo = SERIES[serie]["objetivo"]
    for row in rows[:8]:
        for idx, cell in enumerate(row):
            if _norm(cell) == objetivo:
                logger.info("Cabecera '%s' detectada en columna indice %d", SERIES[serie]["etiqueta"], idx)
                return idx
    fallback = SERIES[serie]["fallback_idx"]
    logger.warning(
        "No se detecto la cabecera '%s'; usando indice de fallback %d",
        SERIES[serie]["etiqueta"], fallback,
    )
    return fallback


def parsear_excel_ipp(
    xlsx_path: Path,
    hoja: str = "IPP Histórico",
    serie: str = "oferta_interna",
) -> pd.DataFrame:
    if serie not in SERIES:
        raise ValueError(f"serie invalida: {serie!r}. Opciones: {list(SERIES)}")

    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)

    # Encontrar la hoja correcta (puede tener nombre con encoding roto)
    sheet = None
    for name in wb.sheetnames:
        if "IPP" in name or "Hist" in name:
            sheet = wb[name]
            break
    if sheet is None:
        raise ValueError(f"No se encontro hoja IPP en {xlsx_path}. Hojas: {wb.sheetnames}")

    # Materializar (la hoja es chica) para evitar problemas de doble iteracion en read_only
    rows = list(sheet.iter_rows(values_only=True))

    col_idx = _detectar_columna(rows, serie)
    logger.info("Serie objetivo: %s", SERIES[serie]["etiqueta"])

    records = []
    anio_actual = None

    # Los datos arrancan en la fila 6 (1-based) = indice 5
    for row in rows[5:]:
        col_anio = row[0] if len(row) > 0 else None
        col_mes = row[1] if len(row) > 1 else None
        col_val = row[col_idx] if len(row) > col_idx else None

        if col_anio and str(col_anio).strip().isdigit():
            anio_actual = int(str(col_anio).strip())

        if anio_actual is None or col_mes is None or col_val is None:
            continue

        mes_str = str(col_mes).strip().split()[0]
        mes_num = MESES_ES.get(mes_str)
        if mes_num is None:
            continue

        try:
            ipp_val = float(col_val)
        except (ValueError, TypeError):
            continue

        records.append({"fecha": pd.Timestamp(anio_actual, mes_num, 1), "ipp": ipp_val})

    if not records:
        raise ValueError(
            f"No se extrajo ningun dato de la columna {col_idx}. "
            "Revisar la estructura del Excel o el flag --serie."
        )

    df = pd.DataFrame(records).drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)

    # Verificar base dic-2014=100
    dic_2014 = df[df["fecha"] == "2014-12-01"]["ipp"]
    if not dic_2014.empty:
        logger.info("Dic-2014 = %.2f (esperado 100.0)", dic_2014.values[0])
    else:
        logger.warning("Diciembre 2014 no encontrado en el Excel")

    logger.info("Total meses: %d  Rango: %s -- %s", len(df), df["fecha"].min().date(), df["fecha"].max().date())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Parsea Excel DANE IPP -> CSV")
    parser.add_argument("xlsx", help="Ruta al Excel DANE (anex-IPP-historicos-*.xlsx)")
    parser.add_argument("--hoja", default="IPP Histórico", help="Nombre de la hoja")
    parser.add_argument(
        "--serie",
        choices=list(SERIES),
        default="oferta_interna",
        help="Agregado del IPP a extraer (default: oferta_interna, el target del modelo)",
    )
    parser.add_argument("--salida", default=str(ROOT / "data" / "raw" / "macro" / "ipp_manual.csv"))
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        logger.error("No se encontro el archivo: %s", xlsx_path)
        sys.exit(1)

    df = parsear_excel_ipp(xlsx_path, args.hoja, args.serie)
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    df["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")
    df.to_csv(out, index=False)
    logger.info("Guardado: %s (%d bytes)", out, out.stat().st_size)
    logger.info("Proximos pasos:")
    logger.info("  .venv\\Scripts\\python scripts/construir_features.py")
    logger.info("  .venv\\Scripts\\python run_monthly_update.py --sin-descarga --solo-ipp")


if __name__ == "__main__":
    main()
