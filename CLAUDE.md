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
└── tests/                       # 101 tests (pytest)
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
1. Descargar la serie IPP mensual desde [dane.gov.co](https://www.dane.gov.co) → Economía → Precios → IPP
2. Guardar como `data/raw/macro/ipp_manual.csv` con formato:
   ```
   fecha,ipp
   2014-12-01,100.0
   2015-01-01,101.3
   ...
   ```
   (fecha = primer día del mes; base dic-2014=100)
3. Ejecutar:
   ```powershell
   .venv\Scripts\python scripts/construir_features.py
   .venv\Scripts\python run_monthly_update.py --solo-ipp
   ```

### 3. Verificar el dashboard

```powershell
$pyexe = "C:\Users\Lenovo\AppData\Local\Python\bin\python.exe"
& $pyexe -m streamlit run dashboard/app.py
```

Abrir http://localhost:8501 y confirmar:
- Tab "Precio Bolsa": pronóstico 7d con bandas de incertidumbre
- Tab "Perfil Horario": heatmap tipo_día × hora razonable
- Tab "Escenarios": seco/promedio/húmedo coherentes (seco > promedio > húmedo)
- Tab "Seguimiento": errores del ciclo anterior registrados

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
| `data/processed/ipp_features_mensual.parquet` | 42 meses × 20 cols | Manual (DANE) |
| `data/raw/macro/ipp_manual.csv` | Serie IPP cruda DANE | Manual mensual |

## Resultados del backtest (junio 2026)

| Horizonte | RMSE ensemble | RMSE naive | Sesgo |
|-----------|:---:|:---:|:---:|
| 1d  | 352 COP/kWh | 271 | +62 |
| 7d  | 359 | 271 | +107 |
| 14d | 496 | 321 | +130 |
| 30d | 844 | 348 | +262 |

El sesgo positivo es esperado: el modelo fue entrenado mayormente con datos de
El Niño 2023-2024 (precios ~800–2000 COP/kWh) y el período de test es post-Niño
(~465 COP/kWh). El loop mensual corrige el sesgo via `actualizar_sesgo()`.

## Pesos del ensemble (junio 2026)

- SARIMAX: 4%, LGB: 96%, AIC SARIMAX: -739.2
- Top features LGB: `precio_bolsa_mean_lag1d` (gain 4415), `lag7d` (456), `aportes_pct_lag1d` (448)

## Tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
# 101 passed
```

## Notas de diseño importantes

1. **Sesgo NO se auto-estima en `fit()`**: el sesgo depende del régimen ENSO y puede
   invertirse de signo entre entrenamiento y despliegue. Se actualiza desde el loop
   mensual vía `modelo.modelo.actualizar_sesgo({horizonte: sesgo_cop})`.

2. **Bandas de incertidumbre**: el ancho del CI viene de SARIMAX en log-espacio,
   centrado sobre la predicción del ensemble. Garantiza CI siempre mayor al punto.

3. **Data leak en backtest**: para h > 1, `_fix_lags_precio()` fija los lags del
   precio en el último valor de entrenamiento (evita que LGB vea el futuro).

4. **VECM en IPP**: activado solo si el test de Johansen detecta cointegración.
   Con datos sintéticos o histórico corto puede no estar disponible → degradación
   elegante (`_available=False`, peso=0 en ensemble).

5. **BanRep SDMX bloqueado**: `load_ipp_local()` es el método requerido para IPP.
   La función `cargar_ipp_banrep_sdmx()` existe pero falla en el entorno actual.
