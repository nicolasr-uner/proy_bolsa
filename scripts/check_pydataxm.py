"""Validacion de acceso a la API de XM via pydataxm (Fase 0).

Confirma que podemos: (1) descargar el precio de bolsa horario, (2) descargar el
precio de escasez (techo regulatorio) y (3) transformar el formato ancho de 24
columnas a una serie horaria larga (timestamp, precio), que es como la
consumiran los modelos.
"""

import datetime as dt

import pandas as pd
from pydataxm.pydataxm import ReadDB

print("pandas:", pd.__version__)
api = ReadDB()


def fetch(metric: str, entity: str, start: dt.date, end: dt.date):
    print(f"\n=== {metric} / {entity}  [{start} -> {end}] ===")
    try:
        df = api.request_data(metric, entity, start, end)
    except Exception as exc:  # noqa: BLE001
        print("  FALLO:", repr(exc))
        return None
    print("  shape:", df.shape)
    print("  columnas:", list(df.columns))
    print(df.head(2).to_string())
    return df


start = dt.date(2026, 5, 1)
end = dt.date(2026, 5, 7)

precio_bolsa = fetch("PrecBolsNaci", "Sistema", start, end)
fetch("PrecEsca", "Sistema", start, end)

# Transformar el formato ancho (Values_Hour01..24) a serie horaria larga.
if precio_bolsa is not None:
    hour_cols = [c for c in precio_bolsa.columns if c.lower().startswith("values_hour")]
    id_cols = [c for c in precio_bolsa.columns if c not in hour_cols]
    long = precio_bolsa.melt(
        id_vars=id_cols, value_vars=hour_cols, var_name="hora", value_name="precio"
    )
    long["hora"] = long["hora"].str.extract(r"(\d+)").astype(int) - 1
    date_col = "Date" if "Date" in long.columns else id_cols[-1]
    long["timestamp"] = pd.to_datetime(long[date_col]) + pd.to_timedelta(
        long["hora"], unit="h"
    )
    long = long.sort_values("timestamp").reset_index(drop=True)

    print("\n=== Serie horaria larga (precio de bolsa) ===")
    print("  filas:", len(long))
    print("  rango:", long["timestamp"].min(), "->", long["timestamp"].max())
    print(long[["timestamp", "precio"]].head(6).to_string(index=False))
    print("\n  estadisticas precio (COP/kWh):")
    print(long["precio"].describe().to_string())
