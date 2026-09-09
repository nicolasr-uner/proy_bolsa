# Manual de uso — proybolsa

Guía operativa para **correr, actualizar y desplegar** el sistema mes a mes. Para la *teoría*
(qué son los modelos, cómo se combinan, desempeño medido) ver **[docs/METODOLOGIA.md](METODOLOGIA.md)**.

> Última actualización: 2026-09-09. Refleja el estado desplegado tras la ronda de mejoras
> (IPP campeón = arima_drift, bolsa recalibrado, torneo de modelos, validación Pandera).

---

## 1. Qué es y qué produce

Sistema de proyección para el mercado eléctrico colombiano, con dos modelos:

| Modelo | Qué proyecta | Horizontes | Para qué |
|---|---|---|---|
| **Precio de bolsa** | COP/kWh (nivel diario → perfil horario) | 7d / 30d / 365d | Operación · presupuesto · valoración de PPAs |
| **IPP** | Índice de Precios del Productor (base dic-2014=100) | 12–24 meses | Indexación de contratos, costos |

Además: un **torneo de modelos** que publica el panel completo cada mes y mide en vivo cuál
pronostica mejor, y un **dashboard Streamlit** (local y en la nube) con pronósticos, escenarios
y seguimiento de precisión.

**Qué se despliega hoy (importante):**
- **IPP:** el pronóstico oficial es **`arima_drift`** (el mejor componente medido). El ensemble
  ponderado perdía contra él en todo horizonte, así que se dejó de desplegar.
- **Bolsa:** ensemble SARIMAX + LightGBM con pesos recalibrados (~0.38 / 0.62).
- **Honestidad:** el modelo de bolsa ~empata/pierde contra un naive; el IPP/arima_drift le gana.
  Cada corrida publica el `skill_vs_naive` por horizonte (ver §4).

---

## 2. Entorno (una sola vez)

- **Python 3.12** gestionado con `uv`, en `.venv\` — es el que corre **los scripts del modelo**.
- El **dashboard local** usa el Python del sistema (`C:\Users\Lenovo\AppData\Local\Python\bin\python.exe`),
  que necesita el stack completo (`streamlit`, `plotly`, `pandas`, `numpy`, `statsmodels`,
  `lightgbm`, `pandera`, `openpyxl`).

```powershell
# Crear el venv del modelo e instalar el paquete + dev
py -m uv venv --python 3.12 .venv
py -m uv pip install --python .venv\Scripts\python.exe -e ".[dev]"

# Verificación rápida
.venv\Scripts\python -m pytest            # los tests deben pasar
```

> **Ojo con el Python del sistema (dashboard):** si al abrir un tab sale
> `ModuleNotFoundError: No module named 'statsmodels'` (o lightgbm/pandera), instalá **sin pins**
> (ese intérprete es 3.14 y los pins de `requirements.txt` no tienen wheels para 3.14):
> ```powershell
> & "C:\Users\Lenovo\AppData\Local\Python\bin\python.exe" -m pip install statsmodels pandera lightgbm plotly openpyxl
> ```
> **NO** uses `-r requirements.txt` en ese intérprete. Los pins son para Streamlit Cloud (Python 3.12).

---

## 3. Ciclo mensual (el primer día hábil de cada mes)

### Paso 1 — Bolsa (automático)
```powershell
.venv\Scripts\python run_monthly_update.py --solo-bolsa
```
Descarga datos nuevos de XM y macro, reconstruye features, **evalúa el error del ciclo anterior**
contra el precio realizado, **actualiza el sesgo**, genera los pronósticos 7d/30d/365d y guarda
todo en `outputs/runs/{YYYY-MM}/`.

### Paso 2 — IPP (requiere carga manual de DANE)
El IPP de DANE no tiene API estable; se carga a mano una vez al mes:
1. Descargar el Excel histórico de [dane.gov.co](https://www.dane.gov.co) → Economía → Precios → IPP
   (archivo `anex-IPP-historicos-{mes}{año}.xlsx`).
2. Convertirlo a `data/raw/macro/ipp_manual.csv`:
   ```powershell
   .venv\Scripts\python scripts\parsear_ipp_dane.py anex-IPP-historicos-may2026.xlsx
   ```
   (formato: `fecha,ipp` — fecha = primer día del mes; base dic-2014=100).
3. Correr el ciclo IPP:
   ```powershell
   .venv\Scripts\python run_monthly_update.py --sin-descarga --solo-ipp
   ```

> Al reconstruir features, la **validación Pandera** corre automáticamente y **aborta con error**
> si algún dato es inválido (precio negativo, IPP fuera de rango, timestamps desordenados). Si eso
> pasa, revisá el dato crudo antes de seguir — no es un bug del código, es el dato.

### Paso 3 — Verificar el dashboard (local)
```powershell
$pyexe = "C:\Users\Lenovo\AppData\Local\Python\bin\python.exe"
& $pyexe -m streamlit run dashboard\app.py   # abrir http://localhost:8501
```
Confirmar los 5 tabs: Precio Bolsa (corto), Perfil Horario, Escenarios Bolsa, IPP y Escenarios,
y **Seguimiento de Precisión** (errores del ciclo anterior + leaderboard del torneo + gráfico de skill).

### Paso 4 — Desplegar a la nube (Streamlit Community Cloud)
La app pública carga modelos **pre-ajustados** (`outputs/models/*.pkl`) porque la nube no puede
re-entrenar. Tras cada ciclo hay que **re-serializar y pushear** o la nube queda con pronósticos viejos:
```powershell
.venv\Scripts\python scripts\serializar_modelos.py    # regenera los .pkl
git add data\processed\*.parquet outputs\runs\ outputs\models\*.pkl
git commit -m "ciclo {YYYY-MM}: features + modelos serializados"
git push origin main                                   # Cloud redespliega solo
```

---

## 4. Cómo leer las salidas

**En `outputs/runs/{YYYY-MM}/`:**
- `pronostico_bolsa_{corto_diario,corto_horario,tactico,largo}_*.parquet` — los pronósticos.
- `pronostico_ipp_{12m,24m}_*.parquet` — IPP.
- **`resumen_bolsa.json` / `resumen_ipp.json`** — el resumen del ciclo. Campos clave:
  - `errores_ciclo_anterior` — MAE/RMSE/sesgo/MAPE del ciclo pasado vs. lo realizado.
  - **`skill_vs_naive`** — por horizonte, `1 − RMSE_modelo/RMSE_naive`. **Positivo = le gana al
    naive; negativo = pierde.** Es el termómetro de honestidad de cada corrida.
  - Pesos, `sesgo_aplicado`, `fecha_run`.

**En el dashboard, tab "Seguimiento de Precisión":**
- Error del ciclo anterior y evolución de pesos.
- **Leaderboard del torneo** (histórico backtest + track record real, con skill vs naive).
- **Gráfico de skill en el tiempo** — se puebla a medida que los pronósticos del torneo vencen.
- Tabla de referencia del backtest de bolsa (dinámica, leída del parquet).

---

## 5. Alertas a revisar manualmente

En `outputs/runs/{YYYY-MM}/resumen_bolsa.json`:
- `skill_vs_naive` **muy negativo** en horizontes cortos → el modelo está perdiendo feo contra la
  persistencia; revisar régimen/datos.
- `errores_ciclo_anterior[sesgo] > 200` → sesgo alto, revisar si cambió el régimen ENSO.
- Si el **ONI supera +0.5 por 3 meses seguidos** → El Niño activo → precios al alza.

En `resumen_ipp.json`:
- `skill_vs_naive` del IPP debería ser **positivo** (arima_drift le gana al drift). Si se vuelve
  negativo, el torneo probablemente ya identificó otro campeón — revisar el leaderboard.

---

## 6. Troubleshooting

| Síntoma | Causa / solución |
|---|---|
| `ModuleNotFoundError` al abrir el dashboard | Falta el stack en el Python del sistema → instalar **sin pins** (ver §2). |
| El ciclo IPP dice "IPP no disponible" | Falta `data/raw/macro/ipp_manual.csv` → hacer el Paso 2. |
| `construir_features` aborta con `SchemaError` | Un dato crudo es inválido → revisar el dato que reporta el error (no relajar el esquema sin entender). |
| La app en la nube muestra pronósticos viejos | No se re-serializó/pusheó tras el ciclo → hacer el Paso 4. |
| El dashboard local tarda/se cuelga al abrir un tab | Está re-ajustando en vivo porque no encuentra los `.pkl` → correr `serializar_modelos.py`. |

---

## 7. Estructura de referencia

```
run_monthly_update.py     Entrypoint del ciclo mensual (--solo-bolsa / --solo-ipp)
scripts/                  descarga, features, serialización, backtest, parseo IPP
src/proybolsa/            paquete: ingest · validate · features · models/{bolsa,ipp} · backtest · torneo
dashboard/app.py          dashboard Streamlit (5 tabs)
data/                     raw (crudos por fuente) · processed (feature matrices)
outputs/                  runs/{YYYY-MM} · models/*.pkl · backtest · torneo · forecasts
docs/                     METODOLOGIA.md (teoría) · MANUAL.md (esto) · estudio_drivers_ipp.md · fuentes.md
```
