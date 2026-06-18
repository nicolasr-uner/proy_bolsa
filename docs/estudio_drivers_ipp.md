# Estudio de Drivers del IPP Oferta Interna — Colombia

**Fecha:** junio 2026  
**Propósito:** Diagnosticar la multicolinealidad del set de drivers actual y proponer
variables ortogonales con fundamento económico.

---

## 1. Diagnóstico: Problema de multicolinealidad

### 1.1 El set original: {TRM, Brent, PPI USA}

| Par | Correlación |
|-----|-------------|
| TRM – PPI USA | **0.80** |
| Brent – PPI USA | **0.82** |
| TRM – Brent | 0.50 |

VIF (Variance Inflation Factors):

| Variable | VIF | Estado |
|----------|-----|--------|
| TRM | 3.42 | Aceptable |
| Brent | 3.68 | Aceptable |
| PPI USA | **7.65** | **Problemático (>5)** |

**Causa raíz:** Las tres variables son proxies del mismo ciclo global de inflación de
dólares (2021-2024). Cuando se juntan en un modelo SARIMAX, los coeficientes se vuelven
inestables. El efecto observable: con el set actual, TRM ↑ 30% → IPP proyectado **baja**
(signo incorrecto). El coeficiente de TRM en SARIMAX es ≈ −0.00002 (casi cero, signo
negativo), económicamente absurdo.

### 1.2 Elasticidades log-log OLS con set original

| Variable | β log-log | Signo esperado | Estado |
|----------|-----------|----------------|--------|
| TRM | +0.25 | ✓ positivo | OK (OLS) |
| Brent | **−0.13** | ❌ negativo | VIF distorsiona |
| PPI USA | +1.45 | ✓ positivo | Absorbe la variación |

El PPI USA absorbe la mayor parte de la varianza y deja a Brent con signo incorrecto.

---

## 2. Canal económico teórico

El IPP Oferta Interna mide los precios que reciben los productores colombianos por su
producción doméstica. Los dos canales principales son:

### Canal 1 — Costo de insumos importados (en pesos)
El precio efectivo que paga un productor colombiano por un insumo importado es:
```
precio_COP = precio_USD × TRM
```
Esto sugiere usar **Brent × TRM** (= `brent_cop`) como variable única, en lugar de
Brent y TRM por separado. Ventajas:
- Captura **conjuntamente** el shock de precio y el shock cambiario
- Elimina la multicolinealidad entre Brent y TRM
- Tiene el signo correcto por construcción: si Brent sube O si el peso se deprecia → `brent_cop` sube → IPP sube ✓

### Canal 2 — Tasa de cambio de los costos (YoY)
Los productores ajustan precios más en respuesta a **cuánto variaron** sus costos en el
último año que al nivel absoluto. Esto sugiere usar variaciones anuales:
- `trm_yoy`: % de depreciación/apreciación del peso en los últimos 12 meses
- `brent_yoy`: % de cambio del Brent en los últimos 12 meses

Estas variables son ortogonales al nivel del tipo de cambio y al nivel del petróleo,
capturando la *aceleración* del ciclo de costos.

---

## 3. Evidencia empírica (datos enero 2015 – mayo 2026, n=137)

### 3.1 Correlación con IPP mensual (cambio log)

| Variable | Corr con ipp_log_dif |
|----------|---------------------|
| brent_yoy | **0.43** ← mejor |
| brent_cop | **0.30** |
| trm_yoy | 0.16 |
| TRM (nivel) | 0.11 |
| Brent (nivel) | 0.25 |
| PPI USA (nivel) | 0.11 |

### 3.2 Ortogonalidad del set propuesto

| Par | Correlación |
|-----|-------------|
| brent_cop – real_trm | −0.20 |
| brent_cop – trm_yoy | −0.10 |
| brent_yoy – trm_yoy | −0.36 |

VIF del set propuesto {brent_cop, real_trm}:

| Variable | VIF |
|----------|-----|
| brent_cop | **1.04** |
| real_trm | **1.04** |

### 3.3 Elasticidades con set propuesto

| Variable | β log-log | Signo | Estado |
|----------|-----------|-------|--------|
| brent_cop | +0.45 | ✓ | Correcto |
| real_trm | +0.29 | ✓ | Correcto |

**Con el set propuesto: brent_cop ↑ → IPP ↑ (correcto)** ✓

---

## 4. Set de drivers recomendado

### 4.1 Variables principales

| Variable | Fórmula | Canal económico | Uso |
|----------|---------|-----------------|-----|
| `brent_cop` | Brent_USD × TRM | Costo importado en pesos | SARIMAX, LGB, VECM |
| `brent_cop_lag1m` | brent_cop desplazado 1m | Transmisión con rezago | SARIMAX, LGB |
| `brent_cop_lag2m` | brent_cop desplazado 2m | Transmisión media | SARIMAX |
| `trm_yoy` | (TRM_t / TRM_{t-12} − 1) × 100 | Ritmo de depreciación | LGB |
| `trm_yoy_lag1m` | trm_yoy desplazado 1m | Rezago de ajuste | SARIMAX |
| `brent_yoy` | (Brent_t / Brent_{t-12} − 1) × 100 | Ciclo commodity | LGB |
| `brent_yoy_lag1m` | brent_yoy desplazado 1m | Rezago commodity | LGB |

### 4.2 Variables que se eliminan del modelo

| Variable eliminada | Razón |
|--------------------|-------|
| `ppi_usa` y lags | VIF=7.65; ya capturado por brent_cop |
| `trm` (nivel) | Ya capturado por brent_cop y trm_yoy |
| `brent` (nivel) | Ya capturado por brent_cop y brent_yoy |

### 4.3 Variables que se mantienen

- `ipp_lag1m`, `ipp_lag2m`, `ipp_lag3m`, `ipp_lag12m` — persistencia autoregresiva (se agregan a LGB)
- `oni_lag`, `enso_el_nino`, `enso_la_nina` — canal climático/estacional
- `mes`, `cos_mes`, `sin_mes` — estacionalidad

### 4.4 Sistema VECM actualizado

El VECM pasa de 3 variables {IPP, TRM, Brent} a 2 variables {IPP, brent_cop}:
- Sistema más parsimonioso (más estable con 137 obs)
- `log(brent_cop) = log(Brent) + log(TRM)` — relación de largo plazo única

---

## 5. Implicaciones para el dashboard interactivo

Con `brent_cop` como driver principal, el usuario puede exponer dos sliders intuitivos:
- **Precio del petróleo (USD/barril)**: Brent asumido
- **TRM (COP/USD)**: tipo de cambio asumido

Internamente: `brent_cop = brent_slider × trm_slider`.

**Dirección de escenarios (post-fix):**
- TRM ↑ (depreciación peso) → brent_cop ↑ → IPP ↑ ✓
- Brent ↑ → brent_cop ↑ → IPP ↑ ✓
- Tanto TRM como Brent tienen efecto correcto ✓

---

## 6. Variables no incluidas por falta de fuente automática

Las siguientes variables serían teóricamente útiles pero requieren carga manual:

| Variable | Canal | Fuente | Disponibilidad |
|----------|-------|--------|---------------|
| Salario mínimo YoY | Costo laboral doméstico | MinTrabajo DANE | Anual (enero) |
| ACPM (diesel) COP | Energía y transporte | SICOM | Mensual, manual |
| Gas natural COP | Energía industrial | CREG SICOM | Mensual, manual |
| IPI (prod. industrial) | Demanda-pull | DANE | 60 días de rezago |

Se documentan como extensiones futuras para aumentar el poder explicativo del componente
doméstico del IPP.

---

*Script de análisis: `scripts/analisis_drivers_ipp.py`*  
*Datos: `data/processed/ipp_features_mensual.parquet`*
