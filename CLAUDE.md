# proy_bolsa — Sistema de Proyecciones de Energía Colombia

Dos modelos de pronóstico para el mercado eléctrico colombiano:
- **Precio de bolsa horario** (multi-horizonte: 7d / 30d / 365d)
- **IPP mensual** (12–24 meses)

## Entorno

```powershell
# El proyecto usa Python de C:\Users\Lenovo\AppData\Local\Python\bin\python.exe
# y el venv en .venv\ para las librerías del modelo.
#
# Para ejecutar scripts del modelo:
.venv\Scripts\python run_monthly_update.py --sin-descarga

# Para el dashboard:
$pyexe = "C:\Users\Lenovo\AppData\Local\Python\bin\python.exe"
& $pyexe -m streamlit run dashboard/app.py
# Abrir http://localhost:8501
```

### Dependencias del Python del sistema (el del dashboard)

El dashboard se ajusta los modelos en vivo, así que el Python del **sistema** necesita TODO
el stack: `streamlit`, `plotly`, `openpyxl`, `pandas`, `numpy`, `statsmodels`, `lightgbm`,
`pandera`. Si al abrir un tab aparece `ModuleNotFoundError: No module named 'statsmodels'`
(o `lightgbm`/`pandera`), instálalos en ese intérprete:

```powershell
& "C:\Users\Lenovo\AppData\Local\Python\bin\python.exe" -m pip install statsmodels pandera lightgbm
```

> ⚠️ **NO uses `-r requirements.txt` en el Python del sistema.** Ese intérprete es 3.14 y
> `requirements.txt` fija `pandas==2.2.3` / `numpy==2.1.3`, versiones **sin wheels para 3.14**
> → el install falla al compilar y aborta toda la transacción (statsmodels nunca se instala).
> Instala los paquetes **sin pins**: pip resuelve `statsmodels-0.14.6` (con wheel cp314) sobre
> el pandas/numpy ya presentes. Los pins de `requirements.txt` son para Streamlit Cloud, que
> debe correr en **Python 3.12 o 3.13** (ahí sí hay wheels para esas versiones).

## Estructura del proyecto

```
proy_bolsa/
├── run_monthly_update.py        # Entrypoint del ciclo mensual (Fase 7)
├── dashboard/app.py             # Dashboard Streamlit (Fase 8)
├── scripts/
│   ├── descarga_historico.py    # Descarga XM + macro via pydataxm
│   ├── construir_features.py   # Feature matrices bolsa + IPP
│   └── ejecutar_modelo_bolsa.py # Genera pronosticos puntuales
├── src/proybolsa/
│   ├── ingest/                  # Conectores XM, BanRep, DANE, FRED
│   ├── validate/                # Esquemas pandera
│   ├── features/                # Calendar, hidro, ENSO, lags, perfil
│   ├── models/
│   │   ├── bolsa/               # EnsembleNivel, PronosticadorBolsa
│   │   └── ipp/                 # EnsembleIPP, PronosticadorIPP
│   └── backtest/                # RollingOriginBacktest, diebold_mariano
├── data/
│   ├── raw/                     # Parquets crudos por fuente
│   └── processed/               # Feature matrices listas para el modelo
├── outputs/
│   └── runs/{YYYY-MM}/          # Resultado de cada ciclo mensual
└── tests/                       # 108 tests (pytest)
```

## Checklist mensual (ejecutar el primer día hábil de cada mes)

### 1. Descarga y reconstrucción de features

```powershell
.venv\Scripts\python run_monthly_update.py --solo-bolsa
```

Esto hace automáticamente:
- `scripts/descarga_historico.py` — descarga datos nuevos de XM y macro
- `scripts/construir_features.py` — reconstruye feature matrices
- Evalúa error del ciclo anterior vs. precio real
- Actualiza sesgo del ensemble
- Genera pronósticos 7d / 30d / 365d
- Guarda todo en `outputs/runs/{YYYY-MM}/`

### 2. IPP (requiere carga manual de DANE)

El IPP de DANE no tiene API pública estable. Pasos:
1. Descargar el Excel histórico desde [dane.gov.co](https://www.dane.gov.co) → Economía → Precios → IPP
   (archivo: `anex-IPP-historicos-{mes}{año}.xlsx`)
2. Copiar el Excel a la raíz del proyecto y convertirlo:
   ```powershell
   .venv\Scripts\python scripts/parsear_ipp_dane.py anex-IPP-historicos-may2026.xlsx
   ```
   Esto genera `data/raw/macro/ipp_manual.csv` automáticamente.

   Formato alternativo si se carga a mano:
   ```
   fecha,ipp
   2014-12-01,100.0
   2015-01-01,101.3
   ...
   ```
   (fecha = primer día del mes; base dic-2014=100)
3. Si los macros (TRM/Brent/PPI) no tienen historia suficiente, extender primero:
   ```powershell
   .venv\Scripts\python scripts/descarga_historico.py --inicio 2015-01-01 --solo-macro --forzar
   ```
4. Ejecutar:
   ```powershell
   .venv\Scripts\python run_monthly_update.py --sin-descarga --solo-ipp
   ```

### 3. Verificar el dashboard

```powershell
$pyexe = "C:\Users\Lenovo\AppData\Local\Python\bin\python.exe"
& $pyexe -m streamlit run dashboard/app.py
```

Abrir http://localhost:8501 y confirmar:
- Tab "Precio Bolsa (corto)": pronóstico 7d con bandas de incertidumbre
- Tab "Perfil Horario": heatmap tipo_día × hora razonable
- Tab "Escenarios Bolsa": sliders aportes/volumen/ONI → seco > promedio > húmedo
- Tab "IPP y Escenarios": sliders TRM%/Brent% → alto > base > bajo
- Tab "Seguimiento de Precisión": errores del ciclo anterior registrados

### 3b. Actualizar el dashboard en la nube (Streamlit Community Cloud)

La app pública vive en **https://nicolasr-uner-proy-bolsa-dashboardapp-jep2eg.streamlit.app/**
(repo público [nicolasr-uner/proy_bolsa](https://github.com/nicolasr-uner/proy_bolsa), Python 3.12).

> ⚠️ Cloud **no ajusta los modelos en vivo** (el `.fit` de SARIMAX bloquea el CPU del tier
> gratuito → 503). En su lugar carga `outputs/models/{bolsa,ipp}.pkl` pre-ajustados. Tras
> cada ciclo mensual los parquets cambian, así que hay que **re-serializar y commitear** o la
> nube queda con pronósticos viejos:

```powershell
.venv\Scripts\python scripts/serializar_modelos.py   # regenera los .pkl (desde el .venv 3.12)
git add data/processed/*.parquet outputs/runs/ outputs/models/*.pkl
git commit -m "ciclo {YYYY-MM}: features + modelos serializados"
git push origin main                                  # Streamlit Cloud redespliega solo
```

(En local el dashboard cae a ajuste en vivo si no encuentra los `.pkl`, así que el paso 3 de
arriba funciona aunque no se hayan regenerado.)

### 4. Alertas a considerar

El loop mensual NO emite alertas automáticas (se diseñó para ser explícito). Revisar manualmente en `outputs/runs/{YYYY-MM}/resumen_bolsa.json`:

- `w_lgb > 0.99`: SARIMAX casi sin peso → posible problema de convergencia
- `errores_ciclo_anterior[sesgo] > 200`: sesgo alto, revisar si cambió el régimen ENSO
- Si el ONI supera +0.5 por 3 meses consecutivos → El Niño activo → precios al alza

## Archivos de datos clave

| Archivo | Descripción | Actualización |
|---------|-------------|--------------|
| `data/processed/bolsa_features_diario.parquet` | 1,261 días × 32 cols | Automática (descarga) |
| `data/processed/bolsa_features_horario.parquet` | 30,264 filas × 31 cols | Automática |
| `data/processed/perfil_horario.parquet` | 864 celdas (3 tipo×12 mes×24h) | Automática |
| `data/processed/ipp_features_mensual.parquet` | 137 meses × 34 cols | Manual (DANE) |
| `data/raw/macro/ipp_manual.csv` | Serie IPP cruda DANE | Manual mensual |

## Estado actual del modelo (2026-09)

> Ver **`docs/MANUAL.md`** (operación) y **`docs/METODOLOGIA.md`** (teoría + desempeño medido).

**Qué se despliega hoy:**
- **IPP:** el pronóstico oficial es **`arima_drift`** (estrategia "campeón por horizonte":
  `EnsembleIPP.fijar_pesos_campeon`, default en `PronosticadorIPP.fit`). Medido: el ensemble
  ponderado pierde contra `arima_drift` en todo horizonte, así que se dejó de desplegar. El
  VECM/SARIMAX/LGB quedan para escenarios. El **torneo** (`src/proybolsa/torneo/`) mide el panel
  completo en vivo y el campeón se auto-adapta si otro gana.
- **Bolsa:** ensemble SARIMAX+LGB con pesos **recalibrados honestos (~0.38 / 0.62)**. El 0.02/0.98
  anterior era un artefacto (la calibración premiaba al LGB por persistir con el precio real del
  futuro); ahora congela los lags de precio como en despliegue (`COLS_LAG_PRECIO`).

**Backtest honesto (evalúa el modelo desplegado) — RMSE COP/kWh:**

| Horizonte | RMSE bolsa (ensemble) | RMSE naive |
|-----------|:---:|:---:|
| 1d  | 330 | 277 |
| 7d  | 304 | 271 |
| 14d | 396 | 319 |
| 30d | 683 | 353 |

Lectura honesta: el ensemble de bolsa ~empata/pierde contra el naive (Diebold-Mariano no
significativo); el IPP/arima_drift le gana al drift. Cada corrida publica `skill_vs_naive` por
horizonte en `resumen_bolsa.json` / `resumen_ipp.json`. El sesgo (transición El Niño→post-Niño)
lo corrige el loop mensual vía `actualizar_sesgo()`.

## Tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
# 211 passed, 2 skipped
```

## Notas de diseño importantes

1. **Sesgo NO se auto-estima en `fit()`**: el sesgo depende del régimen ENSO y puede
   invertirse de signo entre entrenamiento y despliegue. Se actualiza desde el loop
   mensual vía `modelo.modelo.actualizar_sesgo({horizonte: sesgo_cop})`.

2. **Bandas de incertidumbre**: en **bolsa**, el ancho del CI viene de SARIMAX en log-espacio,
   centrado sobre la predicción (heurístico). En **IPP** las bandas se calibran empíricamente
   sobre los errores del backtest (`incertidumbre.py`, cuantiles por horizonte), no se heredan
   de un componente.

3. **Data leak en backtest**: para h > 1, `_fix_lags_precio()` fija los lags del
   precio en el último valor de entrenamiento (evita que LGB vea el futuro).

4. **VECM en IPP**: sistema de 2 variables `{ipp, brent_cop}`. Activado solo si
   Johansen detecta cointegración; degrada elegante si no (`_available=False`, o si la MLE
   es singular pese a Johansen). **Nota:** el pronóstico oficial del IPP ya no usa los pesos
   del ensemble sino el **campeón** (`arima_drift`); el VECM se usa en el **modo escenario**
   (sensibilidad a drivers), no para el punto.

9. **Validación Pandera**: `construir_features` valida los datos crudos (precio horario,
   demanda, ONI, hidrología, macro, IPP) contra `validate/schemas.py` y **lanza** ante datos
   inválidos. Ya no es código muerto.

5. **BanRep SDMX bloqueado**: `load_ipp_local()` es el método requerido para IPP.
   La función `cargar_ipp_banrep_sdmx()` existe pero falla en el entorno actual.

6. **Drivers IPP (ortogonales)**: el modelo usa `brent_cop = Brent×TRM` (VIF≈1.04),
   `brent_yoy` y `trm_yoy` (variaciones YoY en %). `ppi_usa` fue eliminado del set de
   features por alta multicolinealidad (VIF=7.65). La columna `ppi_usa_lag1m` puede
   seguir apareciendo en el parquet (columna huérfana del pipeline de features) pero
   ningún modelo la consume.

7. **`construir_futuro_drivers` en IPP**: firma actual es `(horizonte, trm_var_anual,
   brent_var_anual, oni)`. El parámetro `ppi_var_anual` fue eliminado. El dashboard
   usa esta función para los sliders de escenarios.

8. **Override de hidrología en Bolsa**: `pronosticar(aportes_pct=X, volumen_util_pct=Y)`
   acepta valores arbitrarios del slider. Es incompatible con `escenarios_multiples=True`
   (levanta `ValueError` si se usan juntos).
