# Estudio de fuentes confiables (Fase 0)

Estado: Fase 0 completa salvo IPP Colombia (se valida con scripts/validar_ipp.py al inicio de Fase 2).

## Precio de bolsa (mercado electrico)

| Fuente | Qué aporta | Acceso | Estado |
|---|---|---|---|
| **XM API / `pydataxm`** | Precio de bolsa horario, escasez, despacho ideal, aportes, embalses, demanda, generación | Python `pydataxm`, sin autenticación (límite 31 días/consulta) | **Validado** |
| XM SINERGOX | Validación visual de series | Portal web | Referencia |
| CREG | Reglas de formación de precio, precio de escasez | Gestor normativo | Referencia regulatoria |
| **IDEAM / NOAA CPC** | Índice ONI (ENSO) — régimen hidrológico | `noaa.gov/data/indices/oni.ascii.txt` (sin autenticación) | **Validado** |
| Combustibles (gas, carbón) | Costo marginal térmico | Sin API gratuita confiable | Carga semi-manual |

Métricas XM confirmadas en el catálogo (193 en total):

| Grupo | metric_id | Entidad | Granularidad | Unidad |
|---|---|---|---|---|
| Precio | `PrecBolsNaci` | Sistema | Horaria | COP/kWh |
| Precio | `PrecEsca` / `PrecEscaInf` / `PrecEscaSup` | Sistema | Diaria | COP/kWh |
| Precio | `PrecOferIdeal` | Recurso | Diaria | COP/kWh |
| Hidrología | `PorcApor` | Sistema | Diaria | % media hist. |
| Hidrología | `AporEner` / `AporEnerMediHist` | Sistema | Diaria | kWh |
| Embalses | `PorcVoluUtilDiar` | Sistema | Diaria | % capacidad |
| Embalses | `VoluUtilDiarEner` / `CapaUtilDiarEner` | Sistema | Diaria | kWh |
| Embalses | `VertEner` | Embalse | Diaria | kWh |
| Demanda | `DemaReal` / `DemaCome` | Sistema | Horaria | kWh |
| Demanda | `DemaSIN` | Sistema | Diaria | kWh |
| Demanda | `EscDemUPMEAlto/Medio/Bajo` | Sistema | Mensual | kWh |
| Generación | `Gene` / `GeneIdea` | Recurso/Sistema | Horaria | kWh |

## IPP (macro)

| Fuente | Qué aporta | Acceso | Estado |
|---|---|---|---|
| **DANE** | IPP oficial mensual (base dic-2014), desagregado por destino y CIIU | Banco de datos / boletines (sin API REST limpia) | **Por validar** — ejecutar `scripts/validar_ipp.py` |
| **TRM** | Tasa de cambio USD/COP diaria | `datos.gov.co` Socrata `32sa-8pi3` | **Validado** (3427 COP/USD jun-2026) |
| **FRED — Brent** | Petróleo Brent diario (USD/bbl) | CSV directo `fredgraph.csv?id=DCOILBRENTEU` | **Validado** (98.29 USD/bbl jun-2026) |
| **FRED — PPI EE.UU.** | PPI all commodities mensual | CSV directo `fredgraph.csv?id=PPIACO` | **Validado** (292.50 may-2026) |
| **NOAA CPC — ONI** | Índice ONI mensual (ENSO El Niño/La Niña) | `noaa.gov/data/indices/oni.ascii.txt` | **Validado** |

### Nota sobre IPP DANE — VEREDICTO: carga manual (validado jun 2026)

`scripts/validar_ipp.py` probó todos los endpoints posibles. Resultado:
- **BanRep SDMX** (`suameca.banrep.gov.co/estadisticas-banrep/rest/`) → 404 en todos los paths intentados
- **BanRep web** (`banrep.gov.co/es/estadisticas/`) → bloqueado por bot-protection (CAPTCHA Radware)
- **datos.gov.co Socrata** → candidatos o son 403 o son datos de GLP (no IPP DANE)

**Solución definitiva: descarga mensual manual.**

Pasos (una vez al mes, cuando DANE publica el nuevo IPP):
1. Ir a `dane.gov.co → Precios y costos → IPP → Archivos para descarga`
2. Descargar la serie histórica "Total Oferta Interna" (Excel o CSV)
3. Formatear con columnas `fecha,ipp` (fecha = primer día del mes, ipp = índice base dic-2014=100)
4. Guardar como `data/raw/macro/ipp_manual.csv` (sobrescribir con la versión actualizada)
5. La plantilla de formato está en `data/raw/macro/ipp_manual_PLANTILLA.csv`

En código: `load_ipp_local("data/raw/macro/ipp_manual.csv")` en `src/proybolsa/ingest/macros.py`.

DANE publica el IPP **provisional** y lo revisa al mes siguiente; el loop mensual
debe re-incorporar la versión definitiva como nueva información.

## Hallazgos técnicos de la validación (Fase 0/1)

- `pydataxm 0.3.17` **rompe con pandas ≥ 3.0** (usa `freq='M'`, eliminado en pandas 3).
  Se fija **pandas 2.2.3** y **numpy 2.1.3** (último compatible con numba 0.65 de statsforecast).
- Entorno fijado en **Python 3.12.13** vía `uv` (Python 3.14 del sistema no tiene
  wheels para numba/statsforecast).
- Stack de modelado validado end-to-end: AutoARIMA (statsforecast), SARIMAX (statsmodels), LightGBM.
- `pandas-datareader` descartado: rompe en Python 3.12 (`distutils` eliminado del stdlib).
  Todas las fuentes externas se consumen con `requests` + `pd.read_csv` directo.
- **XM formato ancho → largo**: `Values_Hour01..Hour24` → `timestamp` horario via `horas_anchas_a_largo()`.
  Métricas diarias pueden volver con 1 columna `Values_Code01` o con 24 columnas idénticas;
  `_a_diario()` maneja ambos casos.
