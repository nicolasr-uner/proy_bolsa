# proy_bolsa — Plan de mejoras en tareas ONESHOT

> Cada tarea está redactada para copiar-pegar como prompt autocontenido a una sesión de Claude Code,
> con el archivo exacto y el criterio de "listo cuando…". Deriva de la auditoría del 2026-09-07.
> Orden sugerido: Tier 0 → Tier 1 → Tier 2 → Tier 3. Esfuerzo: **S** ≤30 min · **M** ≤2 h · **L** ≤medio día.

---

## TIER 0 — Honestidad y destrabado (días)

### T0.1 · Mergear y pushear el trabajo varado  ·  S  ·  [EN CURSO]
Merge `fase2/ipp` → `main` (fast-forward, ya hecho local) + `git push origin main`. Sincroniza Streamlit Cloud
con las bandas calibradas y la regla honesta del VECM.
**Listo cuando:** `origin/main` está en `b9810fb` y la app pública carga sin error los 5 tabs.

### T0.2 · Que el backtest evalúe el modelo desplegado  ·  M
> "En `scripts/ejecutar_backtest.py:66` (`_fit_ensemble`) y en el backtest de IPP (`src/proybolsa/backtest/ipp_mensual.py:219,257`), el ensemble se ajusta con `df_val=None` / sin `errores_backtest`, así que corre con pesos por defecto (bolsa 50/50, IPP iguales) en vez de los pesos de producción (bolsa 98/2, IPP calibrados). Refactoriza para que en cada corte del walk-forward se calibren los pesos con un split interno train→val, de modo que el backtest evalúe EXACTAMENTE la configuración desplegada. Reporta los pesos efectivos por corte. No cambies la lógica de los modelos, solo el protocolo de calibración del backtest. Corre pytest al final."
**Listo cuando:** los RMSE del backtest corresponden a los pesos de producción, documentado en el resumen.

### T0.3 · Publicar skill-vs-naive + cobertura en resúmenes y dashboard  ·  M
> "Añade a `src/proybolsa/backtest/rolling_origin.py` y `ipp_mensual.py` el cálculo de skill score (1 − RMSE_modelo/RMSE_naive) y MASE por horizonte. Propágalos a `outputs/runs/{YYYY-MM}/resumen_bolsa.json` y `resumen_ipp.json`, junto con la cobertura empírica (PICP) que ya se calcula. En `dashboard/app.py` añade un panel 'Honestidad' que muestre, por horizonte: skill vs naive, MAPE, y cobertura real de las bandas. No inventes números; usa los que ya produce el backtest."
**Listo cuando:** el dashboard muestra explícitamente dónde el modelo gana o pierde contra el naive.

### T0.4 · Orquestador falla ruidoso  ·  S
> "En `run_monthly_update.py:301-319`, cuando el ciclo aborta por datos rancios o descarga fallida, reemplaza el `logger.error(...) + return` por `raise SystemExit(1)`. Propaga el peor exit code de los subprocesos de `descarga_historico.py` (que ya devuelve 0/1/2). Un cron/CI debe ver el fallo."
**Listo cuando:** un ciclo abortado retorna exit code ≠ 0.

### T0.5 · Cablear Pandera (o borrarlo) + README al día  ·  M
> "Los esquemas de `src/proybolsa/validate/schemas.py` nunca se ejecutan. Opción A (recomendada): invócalos en `scripts/construir_features.py` tras leer cada parquet crudo y en `ingest/store.py:upsert_parquet` antes de persistir. Opción B: si no se van a usar, elimínalos. Además, actualiza `README.md`: la sección 'Estado' dice 'Fase 2-9 pendiente' cuando todo está implementado y desplegado — reemplázala por el estado real."
**Listo cuando:** o los esquemas corren y rechazan datos inválidos en un test, o no existen; y el README no miente.

---

## TIER 1 — Arreglar el núcleo hueco (1–2 semanas)

### T1.1 · Definir el baseline campeón honesto  ·  L
> "Instituye un principio de despliegue: un componente/ensemble solo entra en producción si su skill-vs-naive en el backtest walk-forward es > 0 en el horizonte objetivo. Aplica: para IPP, dado que el ensemble pierde contra `arima_drift` solo, configura el pronóstico oficial para usar `arima_drift` (o un ensemble que demostrablemente lo supere). Para bolsa, establece un baseline seasonal-naive/MA con intervalos calibrados como el campeón a batir. Documenta la decisión en CLAUDE.md."
**Listo cuando:** lo que se despliega tiene skill ≥ 0 vs naive, o se documenta por qué no.

### T1.2 · Arreglar el bug de firma del LGB en IPP  ·  M
> "En `src/proybolsa/backtest/ipp_mensual.py:185-209` (`_llamar_forecast`), a `LGBDriversIPP.predict` (que espera un DataFrame) se le pasa un entero → AttributeError → 0 predicciones válidas en el backtest. Dale a `LGBDriversIPP` un método `forecast(horizon, exog_future=…)` coherente con los otros componentes, o corrige la llamada para pasarle el `df_h`. Decide explícitamente si el ensemble usa `LGBDriversIPP` o el mejorado `LGBDlogIPP`. Verifica que el LGB produzca predicciones válidas en el backtest."
**Listo cuando:** `ipp_lgb_*` tiene n_validas > 0 y su peso refleja precisión medida, no un error.

### T1.3 · Recalibrar pesos con el protocolo real + bajar shrinkage  ·  M
> "Calibra los pesos del ensemble de bolsa (`nivel_diario.py:304-322`) con el mismo protocolo multi-paso del despliegue: lags de precio congelados y exógenas de escenario, por horizonte — no comparar SARIMAX-multipaso vs LGB-1-paso-con-lag-real. En IPP, baja `lambda_shrink` (modelo_ipp.py:716, hoy 10.0) o hazlo dependiente del horizonte, y valida el λ por su efecto en el RMSE del ensemble, no a ojo. Corre el backtest antes y después y compara."
**Listo cuando:** los pesos reflejan el desempeño multi-paso real y el ensemble no queda peor que su mejor componente.

### T1.4 · Matar el fallback silencioso con fuga de datos  ·  M
> "En `src/proybolsa/models/ipp/modelo_ipp.py:827-853`, la ruta `_calibrar_pesos` con `df_val` le da al SARIMAX foresight perfecto de TRM/Brent y al LGB los lags reales del futuro (fuga de datos), y se activa en silencio (solo logger.warning) cuando falta el backtest. Vuélvela opt-in explícito o elimínala; que `fit` falle ruidoso si no encuentra los errores de backtest en vez de calibrar con fuga. Verifica que `resumen_ipp.json` y el `.pkl` salgan siempre del mismo `fit` (hoy divergen: 73% vs 2%)."
**Listo cuando:** no existe ninguna ruta que calibre pesos con datos futuros, y resumen == pkl.

### T1.5 · Calibración de intervalos como titular + sesgo continuo  ·  M
> "Expón `calibrador.cobertura_medida` (IPP) y la PICP de bolsa en los resúmenes. Si la cobertura se desvía de 90%, recalibra con residuos empíricos/conformal en vez de heredar el CI de SARIMAX (nivel_diario.py:369-378). Además, en `run_monthly_update.py:79,119` filtra `escenario == 'promedio'` antes del merge de seguimiento del sesgo a 30d (hoy promedia 3 escenarios, n_obs inflado 3×), y modela el sesgo como función continua del horizonte, aplicado por paso."
**Listo cuando:** las bandas desplegadas tienen cobertura medida ~90% y el sesgo táctico no está contaminado.

---

## TIER 2 — Reencuadrar el producto (semanas)

- **T2.1** · Dashboard lidera con **bandas de riesgo + abanico de escenarios**; el punto se demota. (`dashboard/app.py`)
- **T2.2** · Vista unificada **exposición/margen**: combinar la distribución de precio de bolsa (ingreso) con IPP (costo).
- **T2.3** · **Backtest de la capa horaria** (hoy sin validar): comparar `pred_horaria` vs precio horario real. (`pronostico.py:255`, `hourly_profile.py`)
- **T2.4** · Añadir **drivers domésticos del IPP** (salario mínimo YoY, ACPM/gas COP) ya identificados en `docs/estudio_drivers_ipp.md §6`.
- **T2.5** · **Validación de datos anti-anomalías** más allá del IPP (precios negativos, gaps, saltos, drift).

## TIER 3 — Dónde está el foso real (mes+)

- **T3.1** · Modelos **distribucionales/cuantílicos condicionados a hidrología/ENSO**: régimen ENSO → embalses → riesgo de precio de escasez.
- **T3.2** · **Backtest de cobertura de escenarios** + **VaR de PPA** como entregable de la vista de valoración.
- **T3.3** · **Regresión numérica dorada** en tests + **CI** (GitHub Actions) que corra el backtest y publique skill-vs-naive como *gate* de merge.
- **T3.4** · Pregunta estratégica: ¿el pronóstico puntual de precio es el producto, o lo es la **señal temprana de riesgo de escasez**?

---

*Deriva de la auditoría 2026-09-07. Informe navegable: https://claude.ai/code/artifact/2b2f859d-e568-4423-afed-7c734f6cf6a2*
