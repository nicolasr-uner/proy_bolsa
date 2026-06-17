# Estudio de fuentes confiables (Fase 0)

Estado: en construccion. Las fuentes de XM ya se validaron con consultas reales
(junio 2026); el resto se valida en la Fase 2.

## Precio de bolsa (mercado electrico)

| Fuente | Qué aporta | Acceso | Estado |
|---|---|---|---|
| **XM API / `pydataxm`** | Precio de bolsa horario, escasez, despacho ideal, aportes, embalses, demanda, generación | Python `pydataxm`, sin autenticación (límite 31 días/consulta) | **Validado** |
| XM SINERGOX | Validación visual de series | Portal web | Referencia |
| CREG | Reglas de formación de precio, precio de escasez | Gestor normativo | Referencia regulatoria |
| IDEAM / NOAA CPC | Hidrología, índice ONI (ENSO) | Portales / descarga | Por validar |
| Combustibles (gas, carbón) | Costo marginal térmico | Sin API gratuita confiable | Carga semi-manual |

Métricas XM confirmadas en el catálogo (193 en total): `PrecBolsNaci` (horario),
`PrecEsca`/`PrecEscaInf`/`PrecEscaSup`, `PrecOferIdeal`, `MaxPrecOferNal`,
`PPPrecBolsNaci`, `CERE`, `CEE`. Aportes, embalses, demanda y generación se
mapean en la Fase 2 contra el catálogo (`scripts/check_pydataxm.py`).

## IPP (macro)

| Fuente | Qué aporta | Acceso | Estado |
|---|---|---|---|
| **DANE** | IPP oficial mensual (base dic-2014), desagregado por destino y CIIU | Banco de datos / boletines | Por validar |
| **Banco de la República (SUAMECA)** | IPP, TRM y series macro | SDMX / Excel | Por validar |
| FRED / EIA | Brent, PPI EE.UU. | `pandas-datareader` / API | Por validar |

Nota: DANE publica el IPP **provisional** y lo revisa al mes siguiente; el loop
mensual debe re-incorporar la versión definitiva como nueva información.

## Hallazgos técnicos de la validación (Fase 0/1)

- `pydataxm 0.3.17` **rompe con pandas ≥ 3.0** (usa `freq='M'`, eliminado en pandas 3).
  Se fija **pandas 2.2.3** y **numpy 2.1.3** (último compatible con numba 0.65 de statsforecast).
- Entorno fijado en **Python 3.12.13** vía `uv` (Python 3.14 del sistema no tiene
  wheels para numba/statsforecast).
- Stack de modelado validado end-to-end: AutoARIMA (statsforecast), SARIMAX
  (statsmodels), LightGBM.
