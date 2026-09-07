# Metodología — proy_bolsa

**Fuente única de la teoría** de los dos modelos de proyección: qué son, por qué se eligieron,
sus supuestos, cómo se combinan, cómo salen las bandas y los escenarios, y —lo más importante—
**qué tan bien funcionan medidos honestamente**.

> **Regla de este documento:** describe lo que el código *hace*, verificado contra el código real
> (no contra lo que los docstrings dicen que hace). Donde un docstring del código contradice esto,
> este documento manda y el docstring está desactualizado (ver §8).
>
> Última actualización: 2026-09-07 · Deriva de la auditoría técnica de esa fecha.

---

## 1. Propósito y alcance

Dos pronósticos para decisiones internas en el mercado eléctrico colombiano:

| Modelo | Qué proyecta | Horizontes | Decisión que alimenta |
|---|---|---|---|
| **Precio de bolsa** | COP/kWh horario (vía nivel diario) | 7d / 30d / 365d | Operación (7d) · presupuesto (90d) · valoración de PPAs (24m) |
| **IPP** | Índice de Precios del Productor, oferta interna (base dic-2014=100) | 12–24 meses | Indexación de contratos, costos |

Datos 100% públicos: XM (`pydataxm`), DANE (IPP, carga manual), FRED (Brent, PPI USA), datos.gov.co
(TRM), NOAA CPC (ONI/ENSO). Sin secretos.

---

## 2. Filosofía de modelado

- **Modelos interpretables + ML ligero, combinados en ensemble.** Ningún componente decide solo.
- **Auto-recalibración mensual** (`run_monthly_update.py`): al entrar datos nuevos, el sistema se
  reajusta, mide su error real contra lo que había predicho, y rebalancea.
- **Honestidad medible por diseño**: backtest walk-forward con drivers congelados, benchmarks
  ingenuos, Diebold-Mariano y calibración empírica de intervalos. La §6 reporta el resultado sin
  maquillaje.

> ⚠️ **Advertencia central (léase antes de confiar en un número puntual).** Medido en backtest,
> **ambos modelos pierden hoy contra baselines triviales** en los horizontes que importan (§6).
> El valor real del sistema no está en el punto, sino en la **incertidumbre calibrada** y en los
> **escenarios de drivers**. Tratar el pronóstico puntual como verdad es el error a evitar.

---

## 3. Modelo de precio de bolsa

### 3.1 Descomposición
```
precio_horario(día, hora) = nivel_diario(día) × perfil(hora | tipo_día, mes)
```
El modelo resuelve el **nivel diario**; el **perfil horario** (celdas tipo_día × mes × hora) lo
expande a 24 horas. `pronostico.py:PronosticadorBolsa` une ambos.

### 3.2 Nivel diario — ensemble en log-espacio
`nivel_diario.py:EnsembleNivel`. Se trabaja en `log(precio_diario)` y se exponencia al final:
```
log(precio) = w_sarimax · SARIMAX(exog) + w_lgb · LightGBM(features)
```
- **SARIMAX**: componente con estructura temporal explícita y exógenas (hidrología, escasez, ENSO).
- **LightGBM**: componente tabular no lineal sobre lags de precio, hidrología y calendario.
- **Pesos** por inverse-MSE sobre un tail de validación. *(Ver §6.3 para por qué el 98/2 actual
  no es un juicio de precisión defendible.)*

### 3.3 Qué señal domina cada horizonte
| Horizonte | Régimen | Señal dominante |
|---|---|---|
| corto (1–7d) | operativo | lags de precio (persistencia) |
| táctico (8–90d) | gestión | hidrología (aportes, embalses) + ENSO |
| largo (91–720d) | valoración | estructura estacional + reversión a la media + escenarios |

### 3.4 Drivers
Hidrología (`aportes_pct`, `volumen_util_pct` y sus lags), `precio_escasez`, ENSO (`oni_lag`,
`enso_el_nino`, `enso_la_nina`), lags y volatilidad rolling del propio precio, y calendario
(trig de mes). El ENSO se rezaga hacia adelante (pasado→presente), sin look-ahead.

### 3.5 Escenarios hidrológicos
`pronostico.py:ESCENARIOS_HIDRO` — presets que el dashboard puede sobrescribir con sliders:

| Escenario | aportes_pct | volumen_util_pct |
|---|---:|---:|
| seco | 65 | 40 |
| promedio | 100 | 60 |
| húmedo | 140 | 75 |

En pronóstico, los drivers futuros se **congelan** (último precio observado como lag inicial,
hidrología plana del escenario, ENSO constante). Consecuencia honesta: el pronóstico diario es en
la práctica **persistencia + forma estacional del calendario** modulada por el escenario.

### 3.6 Sesgo
No se auto-estima en `fit()`: el sesgo depende del régimen ENSO y puede invertir de signo entre
entrenamiento y despliegue. Se corrige desde el loop mensual (`actualizar_sesgo`), midiendo el error
del ciclo anterior. *(Limitación conocida: hoy es una constante por horizonte aplicada uniforme al
path; ver §7.)*

### 3.7 Incertidumbre
El ancho del intervalo proviene del SARIMAX en log-espacio, centrado sobre la predicción del
ensemble. Es **heurístico**: con `w_sarimax≈0.02`, la banda hereda el intervalo teórico de un
componente con ~2% de peso, sin garantía de cobertura. El backtest **sí mide** la cobertura real
(PICP) y el Winkler score, pero esa cobertura no está calibrada como en el IPP (§4.6). Es el
frente más débil del modelo de bolsa.

---

## 4. Modelo de IPP

### 4.1 Transformación
Se modela en **diferencias logarítmicas** `dlog(IPP)` para estacionariedad; el pronóstico del nivel
se reconstruye por integración. Requiere `IPP > 0` (índice base dic-2014=100).

### 4.2 Los cuatro componentes
`modelo_ipp.py:EnsembleIPP`:
1. **SARIMA** univariado — baseline robusto.
2. **SARIMAX** con drivers macro (exógenas: `brent_yoy_lag1m`, `trm_yoy_lag1m`, `enso_el_nino`, trig de mes).
3. **VECM** — sistema de 2 variables `{ipp, brent_cop}`, activado **solo** si Johansen detecta
   cointegración (ver §4.4).
4. **LightGBM** sobre features tabulares (`brent_cop` y lags, YoY, lags del IPP, ENSO, calendario).
5. Además, un componente univariado **ARIMADriftIPP** (deriva) cableado al ensemble.

### 4.3 Drivers ortogonales — el porqué
Detalle completo en [estudio_drivers_ipp.md](estudio_drivers_ipp.md). Resumen:
- El set original `{TRM, Brent, PPI USA}` sufría multicolinealidad (PPI USA VIF=7.65), con signos
  económicamente absurdos (TRM↑ → IPP↓).
- **Solución:** `brent_cop = Brent_USD × TRM` (VIF≈1.04) — costo del insumo importado en pesos,
  con signo correcto por construcción; más `trm_yoy` y `brent_yoy` (variaciones anuales en %) que
  capturan la *aceleración* del ciclo de costos.
- `ppi_usa` **fue eliminado**; su columna `ppi_usa_lag1m` puede sobrevivir en el parquet pero
  **ningún modelo la consume**.

Canal económico: costo de insumos importados (`precio_COP = precio_USD × TRM`) + ritmo de
depreciación/commodity (YoY). Transmisión con rezago típico de 1–3 meses.

### 4.4 La "regla honesta" del VECM
Con ~139 observaciones mensuales, la cointegración se acepta solo bajo una regla estricta
(`diagnosticos.py`): la traza de Johansen debe **rechazar r=0 al 95%** con un `k` dado **y**
sostenerse al 90% con `k±1`, con `alpha<0` significativo. Si no se cumple, `_available=False` y el
VECM pesa 0 (degradación elegante). *(En el backtest el VECM se activa en muy pocos orígenes; ver §6.)*

### 4.5 Ensemble
Pesos por inverse-MSE sobre los errores del rolling-origin, **por horizonte**, con *shrinkage*
hacia el reparto igual (evita pesos absurdos con pocos orígenes). Los errores se versionan
(`outputs/backtest/*.parquet`) para que Streamlit Cloud —que no reentrena— use pesos ya calibrados.

### 4.6 Incertidumbre — calibración empírica (el frente más fuerte del sistema)
`incertidumbre.py:CalibradorIntervalos`. Las bandas nominales del 90% cubrían realmente **44–75%**
(un intervalo que dice 90% y cubre 57% "no es un intervalo: es decoración"). Solución:
- **Cuantiles empíricos del error log del backtest**, por horizonte — sin suponer forma del modelo.
- **Log-espacio** (varianza multiplicativa del índice).
- **Cuantiles asimétricos** (`q_lo`, `q_hi` por separado): un sesgo sistemático se absorbe como
  desplazamiento del intervalo, en vez de restarse a mano.
- **Split 70/30**: calibra con el 70% inicial de orígenes, **mide** cobertura en el 30% final
  (evita autovalidación).
- **Honestidad terminológica**: es calibración empírica de cuantiles, **no** conformal prediction
  — la serie tiene regímenes (2021 al 18.6% anual, 2025 al −1.9%) que violan la intercambiabilidad.
  **No hay garantía de cobertura finita**; hay una estimación honesta basada en el pasado reciente.

---

## 5. Backtesting

`backtest/rolling_origin.py` (bolsa) y `backtest/ipp_mensual.py` (IPP).

- **Rolling-origin (walk-forward):** se ajusta con todos los datos hasta la fecha de corte, se
  pronostica a h pasos, se desplaza la ventana y se repite. Mide la degradación real del error con
  el horizonte, no una sola división. Refs: Tashman (2000); Hyndman & Athanasopoulos (2021), cap. 5.
- **Drivers congelados:** para h>1 se congelan los lags/valores que no se conocerían en el origen
  (lags de precio en bolsa; drivers y lags del IPP), el análogo correcto de no ver el futuro.
- **Benchmarks:** bolsa vs `naive` y media móvil 30d; IPP vs `random-walk`, `drift`, `drift12`,
  `Theta`.
- **Métricas:** RMSE, MAE, MAPE, sesgo; cobertura empírica (PICP) y Winkler score de las bandas;
  **Diebold-Mariano** (con varianza HAC/Newey-West) para significancia de la diferencia vs benchmark.

---

## 6. Desempeño medido (la verdad honesta)

> Estas cifras son del backtest real (`outputs/backtest/*.parquet`), verificadas en la auditoría
> 2026-09-07. Se reportan aquí porque **ningún otro documento del proyecto lo hacía**.

### 6.1 Bolsa — pierde contra el naive en todo horizonte
RMSE del ensemble vs random walk; *skill* = 1 − RMSE_modelo/RMSE_naive (negativo = peor):

| Horizonte | RMSE ensemble | RMSE naive | Skill vs naive |
|---|---:|---:|---:|
| 1d | 352 | 271 | **−30%** |
| 7d | 359 | 271 | **−32%** |
| 14d | 496 | 321 | **−54%** |
| 30d | 844 | 348 | **−143%** |

También pierde contra una media móvil de 30d en todo horizonte. En **operación real** el último
ciclo rindió bien (7d MAPE 17.5%) porque corrige sesgo y el período fue calmo — pero el modelo no
es robusto al shock (El Niño), que es cuando el backtest lo muestra dispararse.

### 6.2 IPP — no le gana a un random-walk con deriva
RMSE por horizonte, ensemble de producción vs el componente de deriva solo:

| Horizonte | Ensemble | arima_drift solo | bench_drift |
|---|---:|---:|---:|
| 6m | 7.29 | **7.08** | 7.60 |
| 12m | 14.17 | **13.50** | 14.02 |
| 24m | 25.30 | **24.11** | 24.25 |

Diebold-Mariano da *skill* negativo en todos los horizontes: **desplegar el `arima_drift` solo sería
más preciso** que el ensemble completo.

### 6.3 Advertencias metodológicas que inflan/distorsionan estos números
- **El backtest evalúa pesos que no son los de producción** (bolsa 50/50 en backtest vs 98/2
  desplegado; IPP pesos iguales vs calibrados). Corregir esto es prerrequisito para que los RMSE
  signifiquen algo.
- **El backtest de bolsa usa exógenas futuras reales** (hidrología/ENSO): es *optimista*, y aun así
  pierde → la conclusión "no supera al naive" es sólida, no artefacto de régimen.
- **El VECM del IPP se activa en ~1 de 56 orígenes**: su RMSE aparente excelente es una sola
  observación, no evidencia; la regla honesta lo mantiene con peso ~0.02 en el `.pkl` desplegado.
- **El LightGBM del IPP no producía predicciones válidas** en el backtest (bug de firma): su peso
  bajo era un error, no un juicio. *(Ver el plan de mejoras.)*

Detalle completo con evidencia `archivo:línea`: [PLAN_MEJORAS_2026-09.md](PLAN_MEJORAS_2026-09.md)
y el informe navegable de la auditoría.

---

## 7. Supuestos y limitaciones

| Supuesto / decisión | Dónde se rompe / a vigilar |
|---|---|
| El pronóstico puntual es útil | Falso en horizontes largos; el valor está en distribución y escenarios |
| Los drivers futuros se congelan/asumen | En la práctica el pronóstico ≈ persistencia + estacionalidad |
| Cointegración IPP–brent_cop estable | n≈139 es corto; el VECM casi nunca valida en walk-forward |
| Bandas de bolsa razonables | Son heurísticas, sin cobertura calibrada (a diferencia del IPP) |
| Sesgo constante por horizonte | Parche; no interpola horizontes intermedios ni varía dentro del path |
| El ensemble mejora sobre sus partes | Hoy no: el mejor componente solo iguala o supera al ensemble |

---

## 8. Estado de la documentación (deuda a cerrar)

- **Docstring stale:** el encabezado de `models/ipp/modelo_ipp.py` aún lista `PPI USA` como driver;
  el modelo real usa `brent_cop`/YoY (las constantes `_EXOG_SARIMAX`/`_FEATS_LGB` del mismo archivo
  son la verdad). Pendiente: corregir ese docstring.
- **README:** su sección "Estado" ("Fase 2-9 pendiente") está desactualizada; todo está implementado
  y desplegado. Y su enlace a `../.claude/plans/` apunta a una carpeta **inexistente**.
- Este documento es ahora la fuente única de la teoría; al cambiar un modelo, actualizar aquí primero.

## 9. Referencias

- **Interno:** [estudio_drivers_ipp.md](estudio_drivers_ipp.md) · [fuentes.md](fuentes.md) ·
  [PLAN_MEJORAS_2026-09.md](PLAN_MEJORAS_2026-09.md) · [AUDIT_proybolsa_2026-06-24.md](AUDIT_proybolsa_2026-06-24.md)
- **Académico:** Tashman (2000), *Out-of-sample tests of forecasting accuracy* · Hyndman &
  Athanasopoulos (2021), *Forecasting: Principles & Practice*, cap. 5 · literatura BanRep sobre
  cointegración del IPP con TRM y PPI internacional.
