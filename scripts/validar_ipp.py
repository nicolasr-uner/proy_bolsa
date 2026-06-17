"""Buscar y validar la fuente del IPP Colombia (cierre de Fase 0).

Prueba sistematicamente los posibles endpoints del Banco de la Republica (SDMX)
y de datos.gov.co (Socrata) hasta encontrar el que devuelve datos de IPP.

Ejecutar:
    .venv/Scripts/python scripts/validar_ipp.py

El script imprime el resultado de cada intento y, si encuentra datos validos,
imprime la cadena exacta que hay que copiar a src/proybolsa/ingest/macros.py.
"""

import io
import sys

import pandas as pd
import requests

# =============================================================================
# Candidatos BanRep SDMX
# Los dataflows de BanRep siguen el patron {version}_{grupo}_{nombre}
# Para IPP Oferta Interna se prueban varias convenciones de nombre.
# =============================================================================

BANREP_BASE = "https://suameca.banrep.gov.co/estadisticas-banrep/rest"

BANREP_CANDIDATES = [
    # Formato: (url_template, descripcion)
    (f"{BANREP_BASE}/data/1_1_IPP_OFIN_CO_M/", "BanRep: 1_1_IPP_OFIN_CO_M (clasico)"),
    (f"{BANREP_BASE}/data/1_1_IPP/M.IPP_OFIN../", "BanRep: 1_1_IPP key M.IPP_OFIN"),
    (f"{BANREP_BASE}/data/1_2_1_IPP_OFIN/", "BanRep: 1_2_1_IPP_OFIN"),
    (f"{BANREP_BASE}/data/IPP_OFIN/", "BanRep: IPP_OFIN (sin version)"),
    (f"{BANREP_BASE}/data/1_1_IPP_TOTA/", "BanRep: 1_1_IPP_TOTA (total)"),
    (f"{BANREP_BASE}/data/PR_IPP/", "BanRep: PR_IPP"),
    # Intentar la raiz del SDMX para listar dataflows disponibles
    (f"{BANREP_BASE}/dataflow/", "BanRep: listar dataflows (discovery)"),
    (f"{BANREP_BASE}/datastructure/", "BanRep: listar estructuras (discovery)"),
]

SDMX_PARAMS_CSV = {
    "startPeriod": "2020-01",
    "endPeriod": "2026-05",
    "format": "csvdata",
}

SDMX_PARAMS_JSON = {
    "startPeriod": "2020-01",
    "endPeriod": "2026-05",
    "format": "jsondata",
}

# =============================================================================
# Candidatos datos.gov.co Socrata
# =============================================================================

SOCRATA_BASE = "https://www.datos.gov.co/resource"

SOCRATA_CANDIDATES = [
    # Busqueda generica (API de catalogo)
    ("https://www.datos.gov.co/api/catalog/v1?q=IPP+productor", "datos.gov.co catalogo: IPP"),
    ("https://www.datos.gov.co/api/catalog/v1?q=indice+precios+productor", "datos.gov.co catalogo: IPP largo"),
    # IDs hipoteticos (si se conocen de antemano)
    (f"{SOCRATA_BASE}/hipotetico-ipp.json?$limit=3", "Socrata: ID hipotetico"),
]


def probar_banrep_sdmx() -> None:
    print("\n" + "=" * 60)
    print("PRUEBA: BanRep SDMX")
    print("=" * 60)
    for url, desc in BANREP_CANDIDATES:
        print(f"\n  [{desc}]")
        print(f"  URL: {url}")
        for label, params in [("CSV", SDMX_PARAMS_CSV), ("JSON", SDMX_PARAMS_JSON)]:
            try:
                resp = requests.get(url, params=params, timeout=15)
                print(f"    HTTP {resp.status_code} ({label}), content-type: {resp.headers.get('content-type', '?')}")
                if resp.ok and len(resp.text) > 50:
                    preview = resp.text[:300].replace("\n", " ")
                    print(f"    Preview: {preview}")
                    # Intentar parsear como CSV SDMX
                    if label == "CSV":
                        try:
                            df = pd.read_csv(io.StringIO(resp.text))
                            print(f"    Columnas: {list(df.columns)}")
                            if "OBS_VALUE" in df.columns:
                                print(f"    *** EXITO: {len(df)} filas con OBS_VALUE ***")
                                print(f"\n    >> ACCION: usa esta URL en macros.py/_fetch_ipp_banrep_sdmx:")
                                print(f"       url = '{url}'")
                        except Exception as parse_exc:
                            print(f"    Error parseando CSV: {parse_exc}")
            except requests.exceptions.Timeout:
                print(f"    Timeout ({label})")
            except requests.exceptions.ConnectionError as exc:
                print(f"    ConnectionError ({label}): {exc}")
            except Exception as exc:
                print(f"    Error ({label}): {exc}")


def probar_socrata() -> None:
    print("\n" + "=" * 60)
    print("PRUEBA: datos.gov.co Socrata")
    print("=" * 60)
    for url, desc in SOCRATA_CANDIDATES:
        print(f"\n  [{desc}]")
        print(f"  URL: {url}")
        try:
            resp = requests.get(url, timeout=15)
            print(f"  HTTP {resp.status_code}")
            if resp.ok:
                try:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        print(f"  Primeros campos: {list(data[0].keys())}")
                        print(f"  Primera fila: {data[0]}")
                    elif isinstance(data, dict):
                        # Puede ser un catalogo con 'results'
                        results = data.get("results", data.get("dataset", []))
                        print(f"  Resultados del catalogo: {len(results)}")
                        for r in results[:3]:
                            name = r.get("name", r.get("title", "?"))
                            rid = r.get("id", r.get("resource", {}).get("id", "?"))
                            print(f"    - {name} (id: {rid})")
                            if "ipp" in str(name).lower() or "productor" in str(name).lower():
                                print(f"      *** CANDIDATO IPP: dataset id = {rid} ***")
                except Exception as parse_exc:
                    print(f"  Error parseando: {parse_exc}")
                    print(f"  Preview: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            print("  Timeout")
        except requests.exceptions.ConnectionError as exc:
            print(f"  ConnectionError: {exc}")


def probar_dane_directo() -> None:
    """Intenta acceder al API de DANE (experimental)."""
    print("\n" + "=" * 60)
    print("PRUEBA: DANE API directo (experimental)")
    print("=" * 60)
    candidates = [
        ("https://www.dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/indice-de-precios-del-productor-ipp", "DANE portal IPP"),
        ("https://portalgee.dane.gov.co/api/v2/rest/catalog/", "DANE portal GEE catalog"),
    ]
    for url, desc in candidates:
        print(f"\n  [{desc}]")
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            print(f"  HTTP {resp.status_code}")
            if resp.ok:
                print(f"  Longitud respuesta: {len(resp.text)} chars")
                if resp.headers.get("content-type", "").startswith("application/json"):
                    print(f"  JSON: {resp.text[:300]}")
                else:
                    # Ver si hay links a archivos CSV/Excel con IPP
                    text = resp.text.lower()
                    if ".csv" in text or ".xlsx" in text:
                        print("  Hay referencias a CSV/Excel en la pagina")
        except Exception as exc:
            print(f"  Error: {exc}")


if __name__ == "__main__":
    print("Validacion de fuente IPP Colombia")
    print("Fecha: 2026-06-17")
    print("Busca el endpoint que devuelve IPP mensual, oferta interna, base dic-2014=100")

    probar_banrep_sdmx()
    probar_socrata()
    probar_dane_directo()

    print("\n" + "=" * 60)
    print("SIGUIENTE PASO:")
    print("  Si algun endpoint mostro '*** EXITO ***' o '*** CANDIDATO IPP ***',")
    print("  copiar la URL a src/proybolsa/ingest/macros.py en la funcion")
    print("  _fetch_ipp_banrep_sdmx o _fetch_ipp_socrata segun corresponda.")
    print("  Si ninguno funciono, cargar manualmente desde:")
    print("  https://www.dane.gov.co -> IPP -> Archivos para descarga")
    print("  y guardar como data/raw/macro/ipp_manual.csv con columnas fecha,ipp")
    print("=" * 60)
    sys.exit(0)
