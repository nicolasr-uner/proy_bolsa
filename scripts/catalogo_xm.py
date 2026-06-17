"""Mapea las metricas de XM relevantes para el modelo (Fase 0/2).

Descarga el catalogo completo de la API, lo guarda a CSV (docs/catalogo_xm.csv)
y lista los candidatos por categoria para confirmar los metric_id reales que
iran al diccionario de variables.
"""

from pathlib import Path

from pydataxm.pydataxm import ReadDB

api = ReadDB()
cat = api.get_collections()

out = Path("docs/catalogo_xm.csv")
cat.to_csv(out, index=False, encoding="utf-8")
print("Catalogo guardado:", out, "shape:", cat.shape)

categorias = {
    "aportes_hidrologia": r"Aporte|Aport|Hidr|Caudal|Rio",
    "embalses_volumen": r"Volu|Embalse|Util|Capacidad|Reserv",
    "demanda": r"Dema",
    "generacion": r"Gene",
}

cols = ["MetricId", "MetricName", "Entity", "MaxDays", "Type", "MetricUnits"]
for nombre, patron in categorias.items():
    mask = cat["MetricName"].str.contains(patron, case=False, na=False) | cat[
        "MetricId"
    ].str.contains(patron, case=False, na=False)
    sub = cat.loc[mask, cols]
    print(f"\n=== {nombre} ({int(mask.sum())} metricas) ===")
    print(sub.to_string(index=False))
