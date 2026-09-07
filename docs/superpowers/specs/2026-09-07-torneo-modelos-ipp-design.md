# Spec — Torneo de modelos IPP (auto-evaluación en vivo)

**Fecha:** 2026-09-07 · **Autor:** Nicolas Roveda Salamanca (con Claude Code) · **Estado:** aprobado, listo para plan

## 1. Contexto y problema

La auditoría del 2026-09-07 midió que, en backtest, el ensemble del IPP y sus componentes **pierden contra baselines triviales** (naive / drift), y que solo `arima_drift` tiene skill positivo en todos los horizontes. En vez de colapsar a un único "campeón" elegido por un backtest de una sola vez, el usuario quiere **varias proyecciones que compitan y se auto-evalúen contra la realidad a medida que las fechas ocurren**, para que el campeón emerja de la evidencia real (out-of-sample), no de una decisión previa.

Esto es evaluación **prequencial**: publicar el pronóstico de cada modelo, y puntuarlo cuando el dato real llega.

## 2. Objetivo y no-objetivos

**Objetivo:** un sistema que, cada ciclo mensual, publica el pronóstico del **panel completo** de modelos del IPP, lo **registra**, y a medida que llega el IPP real **puntúa** cada pronóstico vencido, acumulando un **leaderboard en vivo**. El dashboard muestra dos capas: histórico (backtest) y track record real.

**No-objetivos (fuera de alcance ahora):**
- Bolsa (se diseña genérico para extender después; no se construye).
- Auto-selección/auto-despliegue del "ganador" (el humano decide; el sistema solo mide).
- Cambiar los pesos de producción del ensemble (eso es T1.1/T1.3, decisión aparte).

## 3. Decisiones tomadas (brainstorming 2026-09-07)

1. **Panel completo:** compiten `arima_drift`, `ensemble`, `sarimax`, `sarima`, `lgb`, `vecm` + benchmarks `drift`, `random-walk`, `theta`.
2. **IPP primero**, con las funciones genéricas para extender a bolsa después (no ahora).
3. **Híbrido:** el backtest rolling-origin da la evidencia histórica inmediata (~56 orígenes); el registro vivo acumula el track record real de lo publicado.
4. **Dashboard + registro versionado:** leaderboard en el tab "Seguimiento de Precisión"; el registro y el leaderboard se guardan como parquets versionados (Cloud los lee sin recalcular).

## 4. Arquitectura y módulos

Paquete nuevo `src/proybolsa/torneo/`, responsabilidades separadas y testeables:

- **`panel.py`** — genera el pronóstico del panel completo en un origen dado.
  - Reutiliza la fábrica de modelos del backtest: se **promueve `_fabricas()` de `backtest/ipp_mensual.py` a un `fabricas_ipp()` público** (única definición del panel; el backtest pasa a consumir el público).
  - Reutiliza `congelar_drivers_futuros` (drivers congelados → sin fuga, idéntico al backtest).
  - El `ensemble` entra con sus **pesos de producción** (los realmente publicados: fit con `errores_backtest`), no con pesos iguales.
  - Devuelve un DataFrame con una fila por `(modelo, horizonte)`: `modelo · horizonte · fecha_objetivo · y_pred · ci_lo90 · ci_hi90`. Si un modelo no pudo pronosticar (p. ej. VECM sin cointegración), `y_pred = NaN`.

- **`registro.py`** — registro append-only de lo publicado (`outputs/torneo/registro_ipp.parquet`).
  - `registrar_panel(origen_fecha, panel_df, fecha_run)` — append idempotente por `(origen_fecha, modelo, horizonte)` (re-correr el mismo mes no duplica).
  - `cargar_registro()` — lectura.

- **`evaluacion.py`** — el *scorer* y el leaderboard.
  - `resolver(registro_df, actuals)` — para cada fila del registro cuya `fecha_objetivo` ya tiene IPP real y no está resuelta, calcula `error = y_pred − y_real`; append a `outputs/torneo/resueltos_ipp.parquet`.
  - `agregar_leaderboard(resueltos_df)` — agrega por `(modelo, horizonte)`: `n_resueltos · rmse · mae · skill_vs_naive · ultima_fecha` → `outputs/torneo/leaderboard_ipp.parquet`.

## 5. Artefactos de datos (`outputs/torneo/`, versionados)

| Parquet | Grano | Columnas |
|---|---|---|
| `registro_ipp.parquet` | 1 fila por (origen, modelo, horizonte) | `origen_fecha · modelo · horizonte · fecha_objetivo · y_pred · ci_lo90 · ci_hi90 · fecha_run` |
| `resueltos_ipp.parquet` | 1 fila por pronóstico vencido | `origen_fecha · modelo · horizonte · fecha_objetivo · y_pred · y_real · error · fecha_resuelto` |
| `leaderboard_ipp.parquet` | 1 fila por (modelo, horizonte) | `modelo · horizonte · n_resueltos · rmse · mae · skill_vs_naive · ultima_fecha` |

## 6. Flujo (loop mensual, `run_monthly_update.py --solo-ipp`)

Cada ciclo, en orden:
1. **Puntear:** resolver todo pronóstico del `registro` cuya `fecha_objetivo` ya tenga IPP real (append a `resueltos`, recomputar `leaderboard`).
2. **Registrar:** generar el panel completo en el origen actual (drivers congelados) y append al `registro`.

Ambos pasos idempotentes: re-correr el mismo mes no duplica ni re-puntea. El paso corre después de que el ciclo ya cargó el IPP real del mes y ajustó los modelos.

## 7. Métrica y superficie

- **Métrica:** `skill = 1 − RMSE_modelo / RMSE_naive` por horizonte, con **naive = `ipp_bench_drift`** (la vara honesta). Más `rmse` y `mae` crudos. Skill > 0 = le gana a "no hacer casi nada".
- **Dashboard** (`dashboard/app.py`, tab "Seguimiento de Precisión" → leaderboard):
  - Tabla por horizonte con dos capas por modelo: **histórico** (backtest, `outputs/backtest/metricas_ipp.parquet`) y **track record real** (`leaderboard_ipp.parquet`, con su `n_resueltos` para exponer cuánta evidencia hay).
  - Gráfico del skill acumulándose en el tiempo (desde `resueltos_ipp.parquet`).
  - Solo lee parquets; no recalcula (Cloud-friendly).

## 8. Bordes y manejo de errores

- **Revisiones del IPP (DANE):** al resolver se usa el último IPP real disponible; una revisión puede mover levemente un score pasado. Aceptable; se documenta. (No se re-congela el valor "provisional".)
- **Modelo que no pronostica:** `y_pred = NaN` en el registro; no se resuelve (no inventa error). "No pronosticó" queda visible.
- **Idempotencia:** claves `(origen_fecha, modelo, horizonte)` en registro; `(origen_fecha, modelo, horizonte)` ya-resueltas se saltan en el scorer.
- **Sin datos aún:** leaderboard vacío / `n_resueltos = 0` se maneja en el dashboard (muestra "sin evidencia todavía", cae al histórico del backtest).
- **`ensemble` histórico ≠ vivo:** en la capa histórica el `ensemble` es el del backtest (pesos iguales, ver hallazgo I2 de la auditoría); en la capa viva es el desplegado (pesos de producción). Son configs distintas; el dashboard las etiqueta por separado. Unificarlas es T0.2 (que el backtest evalúe el modelo desplegado), fuera de este spec.

## 9. Estrategia de pruebas (TDD)

- **`registro`:** append idempotente (mismo origen dos veces → sin duplicados); esquema correcto.
- **`evaluacion`:** resuelve solo lo vencido (no toca pronósticos cuya fecha no llegó); `error` y `skill` correctos y con el signo correcto; agregación del leaderboard correcta.
- **`panel`:** una fila por `(modelo, horizonte)`; **sin fuga** — los drivers del tramo futuro quedan congelados (mismo patrón que `test_congelar_drivers_no_deja_pasar_el_futuro`).
- **Integración:** secuencia sintética multi-mes → el registro crece, el scorer resuelve al llegar los "actuals", y el modelo construido para ser mejor termina liderando el leaderboard.

## 10. Genericidad (bolsa después)

Las funciones de `registro.py` y `evaluacion.py` se parametrizan por `(serie, fábrica de panel, horizontes, fuente de actuals)`, de modo que extender a bolsa sea un caller nuevo (`panel_bolsa`) + rutas `*_bolsa.parquet`, no una reescritura. No se implementa bolsa en este spec.

## 11. Definition of Done

- [ ] `src/proybolsa/torneo/{panel,registro,evaluacion}.py` con tests verdes (unitarios + integración).
- [ ] `fabricas_ipp()` público en `ipp_mensual.py`; el backtest lo consume (sin regresión de sus tests).
- [ ] `run_monthly_update.py --solo-ipp` puntea-y-registra, idempotente.
- [ ] Los 3 parquets se generan en `outputs/torneo/`.
- [ ] El tab del dashboard muestra el leaderboard (histórico + vivo) sin recalcular.
- [ ] Suite completa verde; sin cambiar los pesos de producción del ensemble.
