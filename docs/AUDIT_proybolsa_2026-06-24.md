# Auditoría Técnica y Master Action Plan — `proybolsa`

**Fecha:** 2026-06-24
**Auditor:** Arquitecto de Software Principal (Claude Code · Opus 4.8)
**Alcance:** auditoría técnica profunda en **solo lectura** (Pasos 0–3) + plan de finalización a producción.
**Repositorio:** [nicolasr-uner/proy_bolsa](https://github.com/nicolasr-uner/proy_bolsa) · rama `main` · árbol limpio · último commit `7b360af`.

> **Nota de método:** los hallazgos de calidad están **verificados ejecutando** `pytest` y `ruff` en el
> `.venv` real (no inferidos). Durante la auditoría se detectaron y **corrigieron 3 afirmaciones falsas**
> producidas por análisis automático desactualizado (ver §2.7).

---

## 1. Resumen Ejecutivo

### ¿Qué es el proyecto? (según lo encontrado, no supuesto)

`proybolsa` (v0.1.0) es un **sistema de pronóstico para el mercado eléctrico colombiano** con dos modelos:

1. **Precio de bolsa horario** (multi-horizonte 7d / 30d / 365d) — ensemble `SARIMAX + LightGBM` sobre
   nivel diario, expandido a perfil horario.
2. **IPP mensual** (12–24 meses) — ensemble `SARIMA + SARIMAX + VECM + LightGBM` con cointegración de Johansen.

Incluye **auto-recalibración mensual** (`run_monthly_update.py`), un **dashboard Streamlit** de 5 vistas
desplegado en **Streamlit Community Cloud**, y backtesting walk-forward (rolling-origin + Diebold-Mariano).

### Clasificación (taxonomía del encargo)

**Tipo 1 híbrido (tablero / app pública liviana) con fuerte naturaleza de Tipo 5 (paquete + CLI) por debajo.**
La cara pública es el dashboard Streamlit Cloud; el motor es un paquete Python instalable (`src/proybolsa`)
operado por un CLI de ciclo mensual. **No es Tipo 2**: los datos son públicos (XM, DANE, FRED, BanRep, NOAA),
sin PII ni secretos.

### Semáforo global

# 🟢 Avanzando — cerca de ✅ listo para producción

El sistema **compila, corre, pasa sus 111 tests y está desplegado** produciendo pronósticos mensuales reales.
Lo que falta es **hardening de cierre** (CI, docs al día, limpieza), no funcionalidad central.

### Completitud estimada: **~88%**

**Metodología de la estimación** (factores ponderados):

| Factor | Evidencia | Aporte |
|---|---|---|
| Features implementadas vs. previstas | 9/9 fases del plan implementadas (ingest→modelos→backtest→loop→dashboard→deploy) | Alto ✅ |
| ¿Corre? | `pytest` → **111 passed in 25.72s**; runs mensuales en `outputs/runs/2026-06/` | Alto ✅ |
| ¿Hay deploy? | Desplegado en Streamlit Community Cloud (URL pública activa) | Alto ✅ |
| Densidad de TODOs | **1** TODO real en todo el código (con fallback funcional) | Alto ✅ |
| Cobertura de tests | 111 tests, pero **sin medición de cobertura** (no hay `pytest-cov`) | Medio 🟡 |
| Calidad / CI | `ruff` con 46 hallazgos de estilo; **sin CI/CD ni pre-commit** | Medio 🟡 |
| Documentación | README "Estado" **obsoleto**; deuda doc vs. realidad | Bajo 🔴 |

El **~12% restante** es: CI/CD, medición de cobertura, README al día, limpieza de huérfanos, alineación de
pins/Dockerfile, y formalización de la fuente IPP.

### ¿Qué falta esencialmente para producción?

> **Nada bloqueante de funcionalidad.** Falta el cierre de calidad: CI que ejecute los tests automáticamente,
> documentación reconciliada con la realidad, y limpieza de deuda menor (huérfanos, pins, Dockerfile).

---

## 2. Auditoría Técnica

### 2.1 Stack y Arquitectura (real)

| Capa | Tecnología | Notas |
|---|---|---|
| Lenguaje | **Python 3.12** (`requires-python = ">=3.12,<3.13"`) | Gestionado con `uv` + `.venv` |
| Datos / I/O | **Parquet** (`pyarrow`), `pandas==2.2.3`, `numpy==2.1.3` | pandas fijado: `pydataxm` rompe con pandas ≥3.0 |
| Validación | `pandera==0.31.1` | Esquemas en `src/proybolsa/validate/schemas.py` |
| Modelos bolsa | `statsmodels` (SARIMAX) + `lightgbm` | `EnsembleNivel`, pesos inverse-MSE |
| Modelos IPP | `statsmodels` (SARIMA/SARIMAX/VECM) + `lightgbm` | Johansen condicional → degradación elegante |
| Forecasting | `statsforecast`, `mlforecast` | numpy fijado por `numba 0.65` |
| Ingesta | `pydataxm==0.3.17`, `requests` | XM, FRED, Socrata, BanRep SDMX, NOAA |
| Dashboard | `streamlit`, `plotly`, `openpyxl` | 5 tabs, escenarios en vivo, export Excel |

**Arquitectura:** pipeline por lotes basado en archivos. **No hay base de datos, ORM, migraciones ni API REST**
(confirmado por búsqueda exhaustiva). Flujo:

```
APIs (XM/FRED/NOAA/DANE) → data/raw/*.parquet → construir_features.py
  → data/processed/*.parquet → PronosticadorBolsa / PronosticadorIPP.fit()
  → outputs/runs/{YYYY-MM}/*.parquet + resumen_*.json → dashboard/app.py
```

El dashboard carga `outputs/models/{bolsa,ipp}.pkl` pre-ajustados (Cloud no reentrena por límite de CPU);
en local cae a ajuste en vivo si no encuentra los `.pkl`.

### 2.2 Estado de Calidad (resultados reales ejecutados)

| Herramienta | Comando | Resultado real |
|---|---|---|
| **Tests** | `pytest tests/ -q` | ✅ **111 passed in 25.72s** — 0 fallos, 0 errores |
| **Lint** | `ruff check .` | 🟡 **46 hallazgos** (todos de estilo, ninguno de correctitud) |

**Desglose de `ruff` por regla:**

| Regla | Nº | Descripción | Auto-fix |
|---|:--:|---|:--:|
| `E402` | 17 | import no al inicio del archivo (por `warnings.filterwarnings()` previo en tests) | ❌ |
| `F401` | 12 | import sin usar | ✅ |
| `F841` | 7 | variable asignada y no usada | ❌ |
| `E702` | 6 | múltiples sentencias en una línea (`;`) — p. ej. `_normalizar_pesos` en IPP | ❌ |
| `F541` | 3 | f-string sin placeholders | ✅ |
| `E401` | 1 | múltiples imports en una línea | ✅ |

**16 de 46 son auto-corregibles** con `ruff check . --fix`. Concentrados en `tests/` y algún punto de
`models/ipp/`. **Ninguno afecta el comportamiento** del sistema.

### 2.3 Deuda Técnica

| # | Deuda | Ubicación | Severidad |
|---|---|---|:--:|
| D1 | Paquetes **vacíos** (placeholders) — la lógica vive en otro sitio (`recalibrate` → `run_monthly_update.py`; `report` → JSON/parquet) | `src/proybolsa/recalibrate/__init__.py`, `src/proybolsa/report/__init__.py` | 🟡 |
| D2 | **Artefactos huérfanos** en raíz | `Proyecciones_Bolsa_Energia.rmd` (R viejo), `Estimación precio de Bolsa-…/` (dir vacío), `anex-IPP-historicos-may2026.xlsx`, `__pycache__/` | 🟡 |
| D3 | **Dockerfile incoherente**: base `python:3.11-slim` mientras `pyproject` exige `>=3.12,<3.13` | `Dockerfile` | 🟡 |
| D4 | **Sin lockfile** (`uv.lock`): reproducibilidad solo por pins en TOML, no de transitivos | raíz | 🟢 |
| D5 | 46 hallazgos de estilo de `ruff` | `tests/`, `models/ipp/` | 🟢 |
| D6 | **Sin medición de cobertura** (no hay `pytest-cov` ni `.coveragerc`) ni `conftest.py` con fixtures | `pyproject.toml`, `tests/` | 🟡 |

### 2.4 Seguridad — ✅ Sin riesgos críticos

| Control | Resultado |
|---|---|
| Secretos hardcodeados | ✅ **Ninguno.** Las únicas coincidencias son URLs públicas (FRED) y llamadas a APIs sin autenticación (Socrata, BanRep SDMX, NOAA). |
| Archivos `.env` commiteados | ✅ No existen. `.gitignore` cubre `.env`, `*.log`, `dashboard/auth.yaml`. |
| `.streamlit/secrets.toml` | ✅ No existe (ni commiteado ni en disco). |
| Variables de entorno | ✅ El código **no usa** `os.environ`/`getenv`. Toda la config viene de `config/*.yaml`. |
| Datos versionados | 🟡 ~24 MB intencionales para Cloud (`data/processed/*.parquet`, `outputs/runs/**`, `bolsa.pkl` 22 MB, `ipp.pkl` 1.1 MB). Correcto por diseño, pero contradice la nota del README. |

> No hay credenciales expuestas. El repo es público y **seguro** de mantener así. El único matiz es que el
> `.pkl` de 22 MB infla el repo; aceptable para el tier gratuito de Streamlit Cloud.

### 2.5 Inventario de Pendientes (TODOs / FIXMEs por módulo)

| Marcador | Archivo:línea | Texto | Estado |
|---|---|---|---|
| `TODO` + `NotImplementedError` | `src/proybolsa/ingest/macros.py:222-230` | `_fetch_ipp_socrata()` — dataset ID de datos.gov.co sin confirmar | 🟡 Mitigado por fallback |
| Comentario | `src/proybolsa/ingest/macros.py:9` | "Fuente pendiente: IPP Colombia" | 🟡 Doc del mismo punto |

> **Solo 1 pendiente funcional real en ~5.4K líneas.** No hay bloques de código muerto comentado.
> Los 4 `except` tolerantes detectados **todos loguean** (no hay `pass` silencioso) → aceptable.

### 2.6 Features Incompletas

| Feature | Detalle | Mitigación existente |
|---|---|---|
| **Fuente IPP automática (Socrata)** | `_fetch_ipp_socrata` lanza `NotImplementedError` (falta `IPP_DATASET_ID`) | ✅ **Fallback funcional** `load_ipp_local()` (CSV manual DANE, vía `scripts/parsear_ipp_dane.py`) + intento BanRep SDMX. Es el flujo documentado y operativo. |
| **VECM en IPP** | Condicional a cointegración de Johansen | ✅ **Por diseño**: si no hay cointegración o faltan datos, `_available=False` y el ensemble redistribuye pesos. No es un defecto. |

### 2.7 ⚠️ Correcciones a hallazgos automáticos (verificadas)

Durante la auditoría, el análisis automático produjo 3 afirmaciones **falsas** que se verificaron y descartaron:

1. **"`scikit-learn==1.9.0` y `pytest==9.1.0` son versiones futuras/rotas (riesgo ALTO)"** → **FALSO.**
   Están **instaladas y operativas** en `.venv/Lib/site-packages` (`scikit_learn-1.9.0.dist-info`,
   `pytest-9.1.0.dist-info`). Eran versiones reales de 2026, solo posteriores al conocimiento del analizador.
   **Los 111 tests pasan con ellas.** Riesgo descartado.
2. **"`diebold_mariano` no está implementada"** → **FALSO.** Existe en `src/proybolsa/backtest/rolling_origin.py`,
   se exporta en `__init__.py`, se usa en `scripts/ejecutar_backtest.py` y se cubre en `tests/test_backtest.py`.
3. **"El proyecto está en Fase 1-2 (en desarrollo)"** → **FALSO** (lo dice el README, no el código).
   La realidad: las 9 fases están implementadas, desplegadas y corriendo. **El README está desactualizado.**

---

## 3. Master Action Plan — Ruta Crítica hacia Producción

> Fases estrictas y secuenciales. Cada tarea está redactada para convertirse en un **prompt ONESHOT
> ejecutable** de Claude Code (autocontenida, con el archivo exacto). MoSCoW = Must / Should / Could / Won't.
> Esfuerzo: **S** ≤30 min · **M** ≤2 h · **L** ≤medio día.

### Fase 1 — Bloqueantes (lo que impide compilar / correr)

> **Buenas noticias: no hay bloqueantes activos.** El sistema corre y pasa sus tests. Esta fase solo
> formaliza la verificación y cierra una incoherencia de despliegue Docker.

| ID | Tarea (ONESHOT) | Archivo/Módulo | MoSCoW | Esfuerzo | Depende de |
|---|---|---|:--:|:--:|:--:|
| F1.1 | "Ejecuta `\.venv\Scripts\python -m pytest tests/ -q` y confirma que los 111 tests siguen verdes; deja el resultado registrado. No cambies código." (Verificación de regresión — ya verificado el 2026-06-24: 111 passed.) | `tests/` | Must | S | — |
| F1.2 | "En `Dockerfile`, cambia la imagen base de `python:3.11-slim` a `python:3.12-slim` para que coincida con `requires-python = '>=3.12,<3.13'` de `pyproject.toml`. Reconstruye mentalmente que `libgomp1` y `build-essential` siguen disponibles en 3.12-slim. No toques el resto del Dockerfile." | `Dockerfile` | Should | S | — |

### Fase 2 — Features Faltantes (mínimo para el alcance del producto)

| ID | Tarea (ONESHOT) | Archivo/Módulo | MoSCoW | Esfuerzo | Depende de |
|---|---|---|:--:|:--:|:--:|
| F2.1 | "Resuelve el estado de `_fetch_ipp_socrata` en `src/proybolsa/ingest/macros.py:222`. Opción A: si `scripts/validar_ipp.py` confirma un `dataset_id` de datos.gov.co válido, impleméntalo. Opción B (recomendada por la realidad operativa): elimina la función stub y su `NotImplementedError`, y deja documentado en el docstring del módulo que la fuente IPP oficial es `load_ipp_local()` (CSV manual DANE), eliminando la 'fuente pendiente' del comentario línea 9." | `src/proybolsa/ingest/macros.py` | Should | M | — |
| F2.2 | "Decide el destino de los paquetes vacíos `src/proybolsa/recalibrate/` y `src/proybolsa/report/`. Opción A: muévele la lógica de recalibración (`actualizar_sesgo`, evaluación de error de ciclo) desde `run_monthly_update.py` a `recalibrate/`, y la escritura de resúmenes a `report/`. Opción B (más simple): elimínalos del paquete si no se van a poblar. Actualiza los imports afectados." | `src/proybolsa/recalibrate/`, `src/proybolsa/report/`, `run_monthly_update.py` | Could | M | — |

### Fase 3 — Refactorización y Pruebas

| ID | Tarea (ONESHOT) | Archivo/Módulo | MoSCoW | Esfuerzo | Depende de |
|---|---|---|:--:|:--:|:--:|
| F3.1 | "Aplica los auto-fixes seguros de ruff: ejecuta `\.venv\Scripts\python -m ruff check . --fix` (resuelve los 16 F401/F541/E401). Luego revisa manualmente los E402 de los tests: muévelos a un patrón aceptado o añade `# noqa: E402` justo donde `warnings.filterwarnings()` obliga a importar después. Vuelve a correr pytest para confirmar que nada se rompió." | `tests/`, `src/proybolsa/models/ipp/` | Should | M | F1.1 |
| F3.2 | "Añade medición de cobertura: agrega `pytest-cov` a `[project.optional-dependencies].dev` en `pyproject.toml` y configura `[tool.pytest.ini_options] addopts = '-q --cov=proybolsa --cov-report=term-missing'`. Ejecuta y reporta el % de cobertura actual como línea base." | `pyproject.toml` | Should | S | F1.1 |
| F3.3 | "Crea `tests/conftest.py` con fixtures compartidas para los DataFrames sintéticos repetidos en `test_modelo_bolsa.py`, `test_modelo_ipp.py` y `test_features.py` (series de precio diario, features IPP mensuales). Refactoriza esos tests para consumir las fixtures sin cambiar las aserciones." | `tests/conftest.py`, `tests/test_modelo_*.py` | Could | M | F3.1 |
| F3.4 | "Limpia los artefactos huérfanos de la raíz tras confirmar que no son fuente activa (grep de referencias en `scripts/` y `src/`): elimina `__pycache__/` de la raíz, el directorio vacío `Estimación precio de Bolsa-20260617T142118Z-3-001/`, y mueve `Proyecciones_Bolsa_Energia.rmd` y `anex-IPP-historicos-may2026.xlsx` a una carpeta `legacy/` (o elimínalos si ya están respaldados). No borres nada referenciado por código." | raíz del proyecto | Should | S | — |

### Fase 4 — Seguridad y Hardening

| ID | Tarea (ONESHOT) | Archivo/Módulo | MoSCoW | Esfuerzo | Depende de |
|---|---|---|:--:|:--:|:--:|
| F4.1 | "Alinea el pin de Plotly: en `requirements.txt` cambia `plotly==6.6.0` a `plotly==6.8.0` para que coincida con `pyproject.toml` y con el `.venv` local (evita divergencia entre desarrollo y Streamlit Cloud). Verifica que ningún otro pin diverja entre ambos archivos." | `requirements.txt` | Should | S | — |
| F4.2 | "Genera un lockfile reproducible: ejecuta `py -m uv pip compile pyproject.toml -o uv.lock` (o `uv lock`) y commitéalo, para fijar también las versiones transitivas. Documenta en el README cómo regenerarlo." | `uv.lock`, `README.md` | Could | S | — |
| F4.3 | "Escaneo de vulnerabilidades de dependencias: ejecuta `pip-audit` (instalándolo en un entorno aislado) sobre el conjunto de `requirements.txt` y reporta cualquier CVE. Como las APIs son públicas y no hay secretos, esto es preventivo." | `requirements.txt` | Could | S | F4.1 |

### Fase 5 — Despliegue / Lanzamiento

| ID | Tarea (ONESHOT) | Archivo/Módulo | MoSCoW | Esfuerzo | Depende de |
|---|---|---|:--:|:--:|:--:|
| F5.1 | "Actualiza el README: reemplaza la sección 'Estado' (que dice 'Fase 1-2 / Fase 3-9 pendientes') por el estado real (9 fases implementadas, desplegado en Streamlit Cloud, 111 tests verdes). Reconcilia la nota 'data/ y outputs/ no versionado' explicando la excepción de `.gitignore` para Streamlit Cloud (parquets + .pkl SÍ se versionan)." | `README.md` | Must | S | — |
| F5.2 | "Crea `.github/workflows/ci.yml` que en cada push y PR a `main`: instale Python 3.12, instale `.[dev]`, y ejecute `ruff check .` y `pytest tests/ -q`. Usa caché de pip. Falla el job si ruff o pytest fallan." | `.github/workflows/ci.yml` | Should | M | F3.1 |
| F5.3 | "Formaliza el paso de re-serialización post-ciclo: añade a `run_monthly_update.py` una opción `--serializar` (o un paso final) que invoque `scripts/serializar_modelos.py` automáticamente al cerrar el ciclo, para que `outputs/models/*.pkl` nunca queden desfasados respecto a los parquets. Documenta en CLAUDE.md el checklist actualizado." | `run_monthly_update.py`, `CLAUDE.md` | Should | M | — |
| F5.4 | "Verifica el deploy en Streamlit Cloud tras el próximo push: confirma que la app pública carga los 5 tabs sin `ModuleNotFoundError` y que los pronósticos coinciden con el último run en `outputs/runs/`. Documenta el resultado." | (deploy) | Must | S | F5.1, F5.2 |

---

## 4. Definition of Done (Definición de "Hecho")

El proyecto se considera **listo para producción** cuando se cumplan, de forma verificable:

- [ ] **Tests verdes en CI**: `pytest tests/ -q` pasa los 111 tests **dentro de GitHub Actions**, no solo en local. *(F5.2)*
- [ ] **Lint limpio**: `ruff check .` retorna 0 errores (o solo `# noqa` justificados). *(F3.1)*
- [ ] **Cobertura medida** y publicada como línea base, con umbral mínimo acordado. *(F3.2)*
- [ ] **Dockerfile coherente** con Python 3.12; la imagen construye sin error. *(F1.2)*
- [ ] **Pins alineados** entre `pyproject.toml` y `requirements.txt`; existe `uv.lock`. *(F4.1, F4.2)*
- [ ] **README al día**: estado real, sin contradicciones sobre versionado de datos. *(F5.1)*
- [ ] **Sin artefactos huérfanos** en la raíz; sin paquetes vacíos sin propósito. *(F3.4, F2.2)*
- [ ] **Fuente IPP documentada** de forma inequívoca (manual vía `load_ipp_local`), sin stubs colgando. *(F2.1)*
- [ ] **Deploy Cloud reproducible**: la app pública refleja el último ciclo mensual sin errores. *(F5.4)*
- [ ] **Seguridad**: confirmado sin secretos en el repo (ya cumplido ✅).

---

## 5. Siguientes Pasos Inmediatos

Tres acciones de **arranque inmediato** (bajo riesgo, alto valor). **Propuestas — no ejecutadas sin tu visto bueno.**

1. **Actualizar el README obsoleto** (F5.1) — es la mentira más visible del repo público.
   `Editar README.md` → reemplazar la sección "Estado" por el estado real (9 fases, desplegado, 111 tests).

2. **Aplicar los auto-fixes de ruff + verificar** (F3.1) — limpieza gratuita y segura.
   ```powershell
   .venv\Scripts\python -m ruff check . --fix
   .venv\Scripts\python -m pytest tests/ -q   # confirmar que sigue en 111 passed
   ```

3. **Alinear el Dockerfile y el pin de Plotly** (F1.2 + F4.1) — coherencia local↔Cloud↔Docker.
   `Editar Dockerfile`: `python:3.11-slim` → `python:3.12-slim`.
   `Editar requirements.txt`: `plotly==6.6.0` → `plotly==6.8.0`.

> Sugerencia de orden: hacer los 3 en una sola rama `chore/audit-cleanup`, correr `pytest` + `ruff`,
> y abrir un PR. Eso deja el repo listo para añadir el CI (F5.2) con la suite ya limpia.

---

*Fin del informe. Auditoría realizada en solo lectura; el único archivo creado fue este informe.
Resultados de calidad verificados ejecutando `pytest` y `ruff` el 2026-06-24.*
