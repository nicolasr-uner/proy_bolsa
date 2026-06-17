"""Convierte el Excel DANE de IPP historico al CSV requerido por el modelo.

Uso:
    .venv\\Scripts\\python scripts\\parsear_ipp_dane.py anex-IPP-historicos-may2026.xlsx
    .venv\\Scripts\\python scripts\\parsear_ipp_dane.py anex-IPP-historicos-jun2026.xlsx --hoja "IPP Histórico"

El Excel de DANE tiene el formato:
  - Fila 5: cabeceras (Produccion Nacional, sectores)
  - Fila 6+: datos, col A=anio (solo primera fila del anio), col B=mes (nombre ES), col C=IPP total
  - Base: Diciembre 2014 = 100

Salida: data/raw/macro/ipp_manual.csv  (fecha,ipp)
"""

from __future__ import annotations

import argparse
import logging
import sys
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


def parsear_excel_ipp(xlsx_path: Path, hoja: str = "IPP Histórico") -> pd.DataFrame:
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)

    # Encontrar la hoja correcta (puede tener nombre con encoding roto)
    sheet = None
    for name in wb.sheetnames:
        if "IPP" in name or "Hist" in name:
            sheet = wb[name]
            break
    if sheet is None:
        raise ValueError(f"No se encontro hoja IPP en {xlsx_path}. Hojas: {wb.sheetnames}")

    records = []
    anio_actual = None

    for row in sheet.iter_rows(min_row=6, values_only=True):
        col_a, col_b, col_c = (row[i] if len(row) > i else None for i in range(3))

        if col_a and str(col_a).strip().isdigit():
            anio_actual = int(str(col_a).strip())

        if anio_actual is None or col_b is None or col_c is None:
            continue

        mes_str = str(col_b).strip().split()[0]
        mes_num = MESES_ES.get(mes_str)
        if mes_num is None:
            continue

        try:
            ipp_val = float(col_c)
        except (ValueError, TypeError):
            continue

        records.append({"fecha": pd.Timestamp(anio_actual, mes_num, 1), "ipp": ipp_val})

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
    parser.add_argument("--salida", default=str(ROOT / "data" / "raw" / "macro" / "ipp_manual.csv"))
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        logger.error("No se encontro el archivo: %s", xlsx_path)
        sys.exit(1)

    df = parsear_excel_ipp(xlsx_path, args.hoja)
    out = Path(args.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    df["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")
    df.to_csv(out, index=False)
    logger.info("Guardado: %s (%d bytes)", out, out.stat().st_size)
    logger.info("Proximos pasos:")
    logger.info("  .venv\\Scripts\\python run_monthly_update.py --sin-descarga --solo-ipp")


if __name__ == "__main__":
    main()
