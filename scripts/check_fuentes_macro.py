"""Validacion de fuentes macro para el IPP (Fase 0/2).

- FRED: Brent (DCOILBRENTEU) y PPI EE.UU. (PPIACO) via pandas-datareader.
- TRM (USD/COP): portal de datos abiertos de Colombia (Socrata, dataset 32sa-8pi3).

El IPP oficial (DANE) y las series del Banco de la Republica via SDMX se
abordan al inicio de la Fase 2 (requieren mas exploracion de endpoints).
"""

import datetime as dt

start, end = dt.date(2024, 1, 1), dt.date(2026, 6, 1)

print("=== FRED (descarga CSV directa, sin dependencias extra) ===")
import pandas as pd

for code, nombre in [
    ("DCOILBRENTEU", "Brent (USD/barril)"),
    ("PPIACO", "PPI EE.UU. all commodities"),
]:
    try:
        df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}")
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[(df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))]
        serie = pd.to_numeric(df[code], errors="coerce").dropna()
        print(
            f"  OK {code} [{nombre}]: filas={len(serie)} | "
            f"ultimo={serie.iloc[-1]:.2f} @ {df[date_col].iloc[-1].date()}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  FALLO {code}: {exc!r}")

print("\n=== TRM via datos.gov.co (Socrata 32sa-8pi3) ===")
try:
    import requests

    resp = requests.get(
        "https://www.datos.gov.co/resource/32sa-8pi3.json",
        params={"$limit": 3, "$order": "vigenciadesde DESC"},
        timeout=30,
    )
    print("  HTTP", resp.status_code)
    if resp.ok:
        data = resp.json()
        print("  campos:", list(data[0].keys()) if data else "(vacio)")
        for row in data[:2]:
            print("   ", row)
except Exception as exc:  # noqa: BLE001
    print("  FALLO:", repr(exc))
