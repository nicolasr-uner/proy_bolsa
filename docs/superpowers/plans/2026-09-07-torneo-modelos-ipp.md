# Torneo de modelos IPP — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar cada ciclo el pronóstico del panel completo de modelos del IPP, registrarlo, y puntuarlo contra el IPP real a medida que las fechas ocurren, acumulando un leaderboard en vivo visible en el dashboard.

**Architecture:** Paquete nuevo `src/proybolsa/torneo/` con tres unidades (`panel` genera pronósticos del panel reusando la fábrica del backtest con drivers congelados; `registro` es un log append-only idempotente; `evaluacion` resuelve pronósticos vencidos y agrega el leaderboard). El loop mensual hace "puntear-luego-registrar". El dashboard lee parquets versionados en `outputs/torneo/`.

**Tech Stack:** Python 3.12, pandas 2.2.3, pyarrow (parquet), pytest. Reutiliza `backtest/ipp_mensual.py` (fábrica de modelos + `congelar_drivers_futuros`) y `models/ipp/`.

**Spec:** `docs/superpowers/specs/2026-09-07-torneo-modelos-ipp-design.md`

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `src/proybolsa/backtest/ipp_mensual.py` (modificar) | Promover `_fabricas()` → `fabricas_ipp()` público (única definición del panel) |
| `src/proybolsa/torneo/__init__.py` (crear) | Exporta la API pública del paquete |
| `src/proybolsa/torneo/registro.py` (crear) | Registro append-only de lo publicado |
| `src/proybolsa/torneo/evaluacion.py` (crear) | Scorer (resolver) + agregación del leaderboard |
| `src/proybolsa/torneo/panel.py` (crear) | Pronóstico del panel completo en un origen, drivers congelados |
| `src/proybolsa/torneo/ciclo.py` (crear) | Orquesta puntear-luego-registrar (un caller, testeable) |
| `run_monthly_update.py` (modificar) | Llama al ciclo del torneo en `--solo-ipp` |
| `dashboard/app.py` (modificar) | Tab "Seguimiento de Precisión" → leaderboard |
| `tests/test_torneo_*.py` (crear) | Tests por unidad + integración |

Convención del repo: docstrings explicativos, español, `outputs/` versionado, TDD con fixtures sintéticos (ver `tests/test_backtest_ipp.py::df_ipp`).

---

## Task 1: Promover `fabricas_ipp()` a público

**Files:**
- Modify: `src/proybolsa/backtest/ipp_mensual.py` (función `_fabricas`, ~línea 234)
- Test: `tests/test_torneo_panel.py` (crear)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_torneo_panel.py
from __future__ import annotations

def test_fabricas_ipp_es_publica_y_trae_el_panel():
    from proybolsa.backtest.ipp_mensual import fabricas_ipp
    fab = fabricas_ipp()
    esperados = {
        "ipp_bench_rw", "ipp_bench_drift", "ipp_bench_theta",
        "ipp_arima_drift", "ipp_sarimax_actual", "ipp_vecm",
        "ipp_lgb_actual", "ipp_ensemble",
    }
    assert esperados <= set(fab), f"faltan modelos: {esperados - set(fab)}"
    # cada valor es una fábrica sin argumentos
    modelo = fab["ipp_arima_drift"]()
    assert hasattr(modelo, "fit")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_panel.py::test_fabricas_ipp_es_publica_y_trae_el_panel -v`
Expected: FAIL — `ImportError: cannot import name 'fabricas_ipp'`

- [ ] **Step 3: Rename `_fabricas` → `fabricas_ipp`, dejar alias**

En `ipp_mensual.py`, renombrar `def _fabricas(...)` a `def fabricas_ipp(...)`, actualizar su docstring, y añadir justo debajo de la función:

```python
# Alias interno retro-compatible (los llamadores previos usaban el nombre privado).
_fabricas = fabricas_ipp
```

Verificar que `backtest_ipp` y cualquier uso interno de `_fabricas()` sigan funcionando (el alias los cubre).

- [ ] **Step 4: Run tests to verify pass + no regression**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_panel.py tests/test_backtest_ipp.py -v`
Expected: PASS (el nuevo test + toda la suite del backtest)

- [ ] **Step 5: Commit**

```bash
git add src/proybolsa/backtest/ipp_mensual.py tests/test_torneo_panel.py
git commit -m "refactor(ipp): fabricas_ipp() publica (unica definicion del panel)"
```

---

## Task 2: `registro.py` — registro append-only idempotente

**Files:**
- Create: `src/proybolsa/torneo/__init__.py`
- Create: `src/proybolsa/torneo/registro.py`
- Test: `tests/test_torneo_registro.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_torneo_registro.py
from __future__ import annotations
import pandas as pd
from proybolsa.torneo.registro import registrar_panel, cargar_registro

def _panel(origen, y):
    return pd.DataFrame({
        "modelo": ["ipp_arima_drift", "ipp_bench_drift"],
        "horizonte": [1, 1],
        "fecha_objetivo": [pd.Timestamp("2026-02-01")] * 2,
        "y_pred": [y, y + 1.0],
        "ci_lo90": [float("nan")] * 2,
        "ci_hi90": [float("nan")] * 2,
    })

def test_registrar_es_idempotente_por_origen_modelo_horizonte(tmp_path):
    ruta = tmp_path / "registro_ipp.parquet"
    origen = pd.Timestamp("2026-01-01")
    registrar_panel(_panel(origen, 100.0), origen_fecha=origen,
                    fecha_run="20260101", ruta=ruta)
    # segunda corrida del MISMO origen: no debe duplicar
    registrar_panel(_panel(origen, 999.0), origen_fecha=origen,
                    fecha_run="20260101", ruta=ruta)
    reg = cargar_registro(ruta)
    assert len(reg) == 2, "el mismo origen no debe duplicar filas"
    # conserva el primer valor registrado (no lo pisa)
    fila = reg[reg["modelo"] == "ipp_arima_drift"].iloc[0]
    assert fila["y_pred"] == 100.0
    assert set(reg.columns) >= {
        "origen_fecha", "modelo", "horizonte", "fecha_objetivo",
        "y_pred", "ci_lo90", "ci_hi90", "fecha_run",
    }

def test_registrar_dos_origenes_distintos_acumula(tmp_path):
    ruta = tmp_path / "registro_ipp.parquet"
    registrar_panel(_panel(pd.Timestamp("2026-01-01"), 100.0),
                    origen_fecha=pd.Timestamp("2026-01-01"), fecha_run="20260101", ruta=ruta)
    registrar_panel(_panel(pd.Timestamp("2026-02-01"), 101.0),
                    origen_fecha=pd.Timestamp("2026-02-01"), fecha_run="20260201", ruta=ruta)
    assert len(cargar_registro(ruta)) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_registro.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'proybolsa.torneo'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/proybolsa/torneo/__init__.py
"""Torneo de modelos: registro prequencial de pronosticos y leaderboard en vivo."""
```

```python
# src/proybolsa/torneo/registro.py
"""Registro append-only de los pronosticos publicados por el panel de modelos.

Una fila por (origen_fecha, modelo, horizonte). Idempotente: registrar dos veces el mismo
origen no duplica ni pisa lo ya escrito, para que re-correr un ciclo mensual sea seguro.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNAS = [
    "origen_fecha", "modelo", "horizonte", "fecha_objetivo",
    "y_pred", "ci_lo90", "ci_hi90", "fecha_run",
]
_CLAVE = ["origen_fecha", "modelo", "horizonte"]


def registrar_panel(panel_df: pd.DataFrame, origen_fecha, fecha_run: str,
                    ruta: str | Path) -> pd.DataFrame:
    """Agrega el panel de un origen al registro. No duplica (origen, modelo, horizonte)."""
    nuevo = panel_df.copy()
    nuevo["origen_fecha"] = pd.Timestamp(origen_fecha)
    nuevo["fecha_run"] = str(fecha_run)
    nuevo = nuevo[COLUMNAS]

    ruta = Path(ruta)
    if ruta.exists():
        prev = pd.read_parquet(ruta)
        ya = set(map(tuple, prev[_CLAVE].astype(str).to_numpy()))
        mask = [tuple(map(str, r)) not in ya for r in nuevo[_CLAVE].to_numpy()]
        combinado = pd.concat([prev, nuevo[mask]], ignore_index=True)
    else:
        combinado = nuevo

    ruta.parent.mkdir(parents=True, exist_ok=True)
    combinado.to_parquet(ruta, index=False)
    return combinado


def cargar_registro(ruta: str | Path) -> pd.DataFrame:
    ruta = Path(ruta)
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS)
    return pd.read_parquet(ruta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_registro.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/proybolsa/torneo/__init__.py src/proybolsa/torneo/registro.py tests/test_torneo_registro.py
git commit -m "feat(torneo): registro append-only idempotente de pronosticos publicados"
```

---

## Task 3: `evaluacion.py` — scorer y leaderboard

**Files:**
- Create: `src/proybolsa/torneo/evaluacion.py`
- Test: `tests/test_torneo_evaluacion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_torneo_evaluacion.py
from __future__ import annotations
import numpy as np
import pandas as pd
from proybolsa.torneo.evaluacion import resolver, agregar_leaderboard

def _registro():
    # dos modelos, horizonte 1, dos origenes; fechas objetivo ene y feb 2026
    return pd.DataFrame({
        "origen_fecha": pd.to_datetime(["2025-12-01", "2025-12-01",
                                        "2026-01-01", "2026-01-01"]),
        "modelo": ["ipp_arima_drift", "ipp_bench_drift"] * 2,
        "horizonte": [1, 1, 1, 1],
        "fecha_objetivo": pd.to_datetime(["2026-01-01", "2026-01-01",
                                          "2026-02-01", "2026-02-01"]),
        "y_pred": [100.0, 103.0, 101.0, 104.0],
        "ci_lo90": [np.nan] * 4, "ci_hi90": [np.nan] * 4,
        "fecha_run": ["x"] * 4,
    })

def _actuals():
    # solo enero tiene real todavia (feb aun no ocurre)
    return pd.DataFrame({"fecha": pd.to_datetime(["2026-01-01"]), "ipp": [101.0]})

def test_resolver_solo_puntea_lo_vencido():
    res = resolver(_registro(), _actuals())
    # solo las 2 filas con fecha_objetivo 2026-01-01 se resuelven
    assert len(res) == 2
    assert set(res["fecha_objetivo"].dt.month) == {1}
    fila = res[res["modelo"] == "ipp_arima_drift"].iloc[0]
    assert fila["y_real"] == 101.0
    assert fila["error"] == 100.0 - 101.0  # y_pred - y_real

def test_leaderboard_skill_vs_naive():
    res = resolver(_registro(), _actuals())
    lb = agregar_leaderboard(res, naive="ipp_bench_drift")
    fila = lb[(lb["modelo"] == "ipp_arima_drift") & (lb["horizonte"] == 1)].iloc[0]
    # arima erro 1.0, naive (drift) erro 2.0 -> skill = 1 - 1/2 = 0.5
    assert fila["rmse"] == 1.0
    assert fila["skill_vs_naive"] == 0.5
    assert fila["n_resueltos"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_evaluacion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'proybolsa.torneo.evaluacion'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/proybolsa/torneo/evaluacion.py
"""Scorer prequencial: resuelve pronosticos vencidos contra el IPP real y agrega el leaderboard.

`resolver` empareja cada fila del registro cuya fecha_objetivo ya tiene IPP real con ese valor
y calcula el error. `agregar_leaderboard` resume por (modelo, horizonte) con skill vs. naive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def resolver(registro_df: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por pronostico vencido: y_pred, y_real, error.

    actuals: DataFrame con columnas fecha (mensual, primer dia) e ipp (real).
    Solo se resuelven filas con y_pred no-nulo y cuya fecha_objetivo existe en actuals.
    """
    reales = actuals[["fecha", "ipp"]].rename(columns={"fecha": "fecha_objetivo", "ipp": "y_real"})
    df = registro_df.dropna(subset=["y_pred"]).merge(reales, on="fecha_objetivo", how="inner")
    df["error"] = df["y_pred"] - df["y_real"]
    df["fecha_resuelto"] = pd.Timestamp.now().normalize()
    return df[["origen_fecha", "modelo", "horizonte", "fecha_objetivo",
               "y_pred", "y_real", "error", "fecha_resuelto"]].reset_index(drop=True)


def agregar_leaderboard(resueltos_df: pd.DataFrame,
                        naive: str = "ipp_bench_drift") -> pd.DataFrame:
    """Agrega por (modelo, horizonte): n, rmse, mae, skill vs. naive, ultima fecha."""
    if resueltos_df.empty:
        return pd.DataFrame(columns=["modelo", "horizonte", "n_resueltos",
                                     "rmse", "mae", "skill_vs_naive", "ultima_fecha"])

    def _agg(g):
        return pd.Series({
            "n_resueltos": len(g),
            "rmse": float(np.sqrt((g["error"] ** 2).mean())),
            "mae": float(g["error"].abs().mean()),
            "ultima_fecha": g["fecha_objetivo"].max(),
        })

    lb = resueltos_df.groupby(["modelo", "horizonte"]).apply(_agg).reset_index()

    # skill = 1 - rmse_modelo / rmse_naive, por horizonte
    rmse_naive = lb[lb["modelo"] == naive].set_index("horizonte")["rmse"]
    lb["skill_vs_naive"] = lb.apply(
        lambda r: (1 - r["rmse"] / rmse_naive[r["horizonte"]])
        if r["horizonte"] in rmse_naive.index and rmse_naive[r["horizonte"]] > 0 else np.nan,
        axis=1,
    )
    return lb
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_evaluacion.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/proybolsa/torneo/evaluacion.py tests/test_torneo_evaluacion.py
git commit -m "feat(torneo): scorer prequencial + leaderboard con skill vs naive"
```

---

## Task 4: `panel.py` — pronóstico del panel con drivers congelados

**Files:**
- Create: `src/proybolsa/torneo/panel.py`
- Test: `tests/test_torneo_panel.py` (añadir)

- [ ] **Step 1: Write the failing test** (añadir al archivo existente)

```python
# tests/test_torneo_panel.py  (añadir; reusar el fixture df_ipp del backtest)
import numpy as np
import pandas as pd
import pytest
from tests.test_backtest_ipp import df_ipp  # reusa el fixture con drivers

def test_panel_una_fila_por_modelo_y_horizonte(df_ipp):
    from proybolsa.torneo.panel import pronosticar_panel
    origen = df_ipp["fecha"].iloc[-1]
    panel = pronosticar_panel(df_ipp, origen_fecha=origen, horizontes=(1, 6, 12))
    assert set(panel.columns) >= {"modelo", "horizonte", "fecha_objetivo",
                                  "y_pred", "ci_lo90", "ci_hi90"}
    # arima_drift debe estar y dar 3 horizontes finitos
    ad = panel[panel["modelo"] == "ipp_arima_drift"]
    assert sorted(ad["horizonte"]) == [1, 6, 12]
    assert np.isfinite(ad["y_pred"]).all()
    # las fechas objetivo son meses futuros del origen
    assert (ad["fecha_objetivo"] > origen).all()

def test_panel_no_deja_pasar_el_futuro(df_ipp):
    """El pronostico del panel usa drivers CONGELADOS: cambiar el futuro real no lo altera."""
    from proybolsa.torneo.panel import pronosticar_panel
    origen = df_ipp["fecha"].iloc[-13]  # deja 12 meses de 'futuro' en el df
    p1 = pronosticar_panel(df_ipp, origen_fecha=origen, horizontes=(6,))
    # alterar los drivers del futuro real no debe cambiar el pronostico (estan congelados)
    df2 = df_ipp.copy()
    fut = df2["fecha"] > origen
    df2.loc[fut, ["brent_cop", "trm_yoy", "brent_yoy"]] *= 3.0
    p2 = pronosticar_panel(df2, origen_fecha=origen, horizontes=(6,))
    m = "ipp_sarimax_actual"
    v1 = p1[p1.modelo == m]["y_pred"].iloc[0]
    v2 = p2[p2.modelo == m]["y_pred"].iloc[0]
    assert v1 == pytest.approx(v2), "el SARIMAX vio el futuro: hay fuga"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'proybolsa.torneo.panel'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/proybolsa/torneo/panel.py
"""Pronostico del panel completo de modelos en un origen dado, con drivers CONGELADOS.

Reusa la fabrica del backtest (`fabricas_ipp`) y `congelar_drivers_futuros`, de modo que el
panel del torneo mide exactamente lo mismo que el backtest: multi-paso real sin foresight. El
ensemble entra con sus pesos de produccion (fit con errores_backtest); el resto con fit(train).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from proybolsa.backtest.ipp_mensual import (
    _llamar_forecast,
    congelar_drivers_futuros,
    fabricas_ipp,
)

HORIZONTES = (1, 3, 6, 12, 24)


def _marco_futuro(train: pd.DataFrame, origen_fecha: pd.Timestamp, h_max: int) -> pd.DataFrame:
    """Filas futuras con el calendario real y los drivers copiados del ultimo train
    (congelar_drivers_futuros los re-congela; el calendario SI es conocido)."""
    fechas = pd.date_range(origen_fecha, periods=h_max + 1, freq="MS")[1:]
    fut = pd.DataFrame({"fecha": fechas})
    fut["mes"] = fut["fecha"].dt.month
    fut["cos_mes"] = np.cos(2 * np.pi * fut["mes"] / 12)
    fut["sin_mes"] = np.sin(2 * np.pi * fut["mes"] / 12)
    ultimo = train.iloc[-1]
    for col in train.columns:
        if col not in fut.columns and col not in ("fecha", "ipp"):
            fut[col] = ultimo[col]
    return fut


def pronosticar_panel(df: pd.DataFrame, origen_fecha, horizontes=HORIZONTES,
                      errores_backtest: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.sort_values("fecha").reset_index(drop=True)
    origen_fecha = pd.Timestamp(origen_fecha)
    train = df[df["fecha"] <= origen_fecha].reset_index(drop=True)
    h_max = max(horizontes)
    fut = _marco_futuro(train, origen_fecha, h_max)
    exog = congelar_drivers_futuros(fut, train, modo="congelado")
    fechas_obj = fut["fecha"].to_numpy()

    filas = []
    for nombre, fabrica in fabricas_ipp().items():
        modelo = fabrica()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if nombre == "ipp_ensemble":
                modelo.fit(train, errores_backtest=errores_backtest)
            else:
                modelo.fit(train)
            fc = _llamar_forecast(modelo, h_max, exog)
        for h in horizontes:
            filas.append({
                "modelo": nombre,
                "horizonte": h,
                "fecha_objetivo": pd.Timestamp(fechas_obj[h - 1]),
                "y_pred": float(fc["pred"].iloc[h - 1]),
                "ci_lo90": float(fc["ci_lo90"].iloc[h - 1]),
                "ci_hi90": float(fc["ci_hi90"].iloc[h - 1]),
            })
    return pd.DataFrame(filas)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_panel.py -v`
Expected: PASS (los 3 tests: fabricas + los 2 de panel)

- [ ] **Step 5: Commit**

```bash
git add src/proybolsa/torneo/panel.py tests/test_torneo_panel.py
git commit -m "feat(torneo): pronostico del panel con drivers congelados (sin fuga)"
```

---

## Task 5: `ciclo.py` — orquesta puntear-luego-registrar + integración

**Files:**
- Create: `src/proybolsa/torneo/ciclo.py`
- Test: `tests/test_torneo_integracion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_torneo_integracion.py
from __future__ import annotations
import numpy as np
import pandas as pd
from tests.test_backtest_ipp import df_ipp

def test_ciclo_registra_puntea_y_lidera_el_mejor(df_ipp, tmp_path):
    """Corriendo el ciclo mes a mes, el registro crece, el scorer resuelve al llegar los
    reales, y el leaderboard termina con evidencia (n_resueltos>0) para arima_drift."""
    from proybolsa.torneo.ciclo import correr_ciclo_torneo
    from proybolsa.torneo.evaluacion import agregar_leaderboard
    import proybolsa.torneo.ciclo as ciclo

    dir_out = tmp_path
    # simular 6 ciclos: cada uno ve un mes mas de historia
    corte0 = len(df_ipp) - 8
    for i in range(6):
        hist = df_ipp.iloc[: corte0 + i].reset_index(drop=True)
        correr_ciclo_torneo(hist, dir_salida=dir_out, fecha_run=f"2026{i:02d}01",
                            horizontes=(1, 3))
    reg = pd.read_parquet(dir_out / "registro_ipp.parquet")
    res = pd.read_parquet(dir_out / "resueltos_ipp.parquet")
    lb = pd.read_parquet(dir_out / "leaderboard_ipp.parquet")
    assert len(reg) > 0 and len(res) > 0
    ad = lb[(lb.modelo == "ipp_arima_drift") & (lb.horizonte == 1)]
    assert not ad.empty and ad["n_resueltos"].iloc[0] > 0

def test_ciclo_es_idempotente(df_ipp, tmp_path):
    from proybolsa.torneo.ciclo import correr_ciclo_torneo
    hist = df_ipp.iloc[:-4].reset_index(drop=True)
    correr_ciclo_torneo(hist, dir_salida=tmp_path, fecha_run="20260101", horizontes=(1,))
    n1 = len(pd.read_parquet(tmp_path / "registro_ipp.parquet"))
    correr_ciclo_torneo(hist, dir_salida=tmp_path, fecha_run="20260101", horizontes=(1,))
    n2 = len(pd.read_parquet(tmp_path / "registro_ipp.parquet"))
    assert n1 == n2, "re-correr el mismo ciclo no debe duplicar el registro"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_integracion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'proybolsa.torneo.ciclo'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/proybolsa/torneo/ciclo.py
"""Orquesta un ciclo del torneo: puntear lo vencido, luego registrar el panel del origen.

El origen es el ultimo mes con IPP real en `df`. Los 'actuals' para puntear son la propia
serie de IPP de `df` (todos los meses observados). Diseñado con dir_salida parametrizable
para testear sin tocar outputs/ reales.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from proybolsa.torneo.evaluacion import agregar_leaderboard, resolver
from proybolsa.torneo.panel import HORIZONTES, pronosticar_panel
from proybolsa.torneo.registro import cargar_registro, registrar_panel


def correr_ciclo_torneo(df: pd.DataFrame, dir_salida: str | Path, fecha_run: str,
                        horizontes=HORIZONTES, errores_backtest: pd.DataFrame | None = None
                        ) -> None:
    dir_salida = Path(dir_salida)
    r_reg = dir_salida / "registro_ipp.parquet"
    r_res = dir_salida / "resueltos_ipp.parquet"
    r_lb = dir_salida / "leaderboard_ipp.parquet"

    df = df.sort_values("fecha").dropna(subset=["ipp"]).reset_index(drop=True)
    origen = df["fecha"].iloc[-1]

    # 1. PUNTEAR: resolver el registro previo contra los IPP reales conocidos hoy.
    reg_prev = cargar_registro(r_reg)
    if not reg_prev.empty:
        res = resolver(reg_prev, df[["fecha", "ipp"]])
        if not res.empty:
            dir_salida.mkdir(parents=True, exist_ok=True)
            res.to_parquet(r_res, index=False)
            agregar_leaderboard(res).to_parquet(r_lb, index=False)

    # 2. REGISTRAR: el panel del origen actual (drivers congelados).
    panel = pronosticar_panel(df, origen_fecha=origen, horizontes=horizontes,
                              errores_backtest=errores_backtest)
    registrar_panel(panel, origen_fecha=origen, fecha_run=fecha_run, ruta=r_reg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_torneo_integracion.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/proybolsa/torneo/ciclo.py tests/test_torneo_integracion.py
git commit -m "feat(torneo): ciclo puntear-luego-registrar idempotente"
```

---

## Task 6: Cablear en `run_monthly_update.py --solo-ipp`

**Files:**
- Modify: `run_monthly_update.py` (función del ciclo IPP; buscar `--solo-ipp` / `ciclo_ipp`)
- Test: manual (el orquestador es un script; su lógica ya está testeada en Task 5)

- [ ] **Step 1: Localizar el punto de inserción**

Run: `.venv\Scripts\python -c "import ast,sys; print([n.name for n in ast.walk(ast.parse(open('run_monthly_update.py',encoding='utf-8').read())) if isinstance(n,ast.FunctionDef)])"`
Buscar la función que ejecuta el ciclo del IPP (p. ej. `ciclo_ipp`). Leerla para ver de dónde sale el DataFrame de features del IPP y `outputs/`.

- [ ] **Step 2: Añadir la llamada al torneo tras generar los pronósticos del IPP**

En la función del ciclo IPP, después de que se cargó el IPP real y se generaron los pronósticos, añadir:

```python
from proybolsa.torneo.ciclo import correr_ciclo_torneo

# Torneo de modelos: puntear lo vencido y registrar el panel de este origen.
try:
    correr_ciclo_torneo(
        df_ipp_features,                     # el DataFrame mensual con ipp + drivers
        dir_salida=ROOT / "outputs" / "torneo",
        fecha_run=fecha_run,                 # el mismo YYYYMMDD del ciclo
        errores_backtest=_cargar_errores_backtest(RUTA_BACKTEST_IPP),
    )
    logger.info("Torneo IPP actualizado en outputs/torneo/")
except Exception:
    logger.exception("El torneo IPP fallo; el ciclo principal continua")
```

Ajustar los nombres reales (`df_ipp_features`, `fecha_run`, `_cargar_errores_backtest`, la ruta del backtest) a los que use el archivo. El `try/except` evita que un fallo del torneo tumbe el ciclo principal.

- [ ] **Step 3: Verificar que el ciclo corre end-to-end en seco**

Run: `.venv\Scripts\python run_monthly_update.py --sin-descarga --solo-ipp`
Expected: corre sin excepción; aparece "Torneo IPP actualizado"; se crean `outputs/torneo/registro_ipp.parquet` y (si hay reales vencidos) `resueltos_ipp.parquet` + `leaderboard_ipp.parquet`.

- [ ] **Step 4: Commit**

```bash
git add run_monthly_update.py
git commit -m "feat(torneo): cablear el torneo IPP al ciclo mensual"
```

---

## Task 7: Leaderboard en el dashboard

**Files:**
- Modify: `dashboard/app.py` (tab "Seguimiento de Precisión")
- Test: manual (Streamlit)

- [ ] **Step 1: Localizar el tab**

Run: `.venv\Scripts\python -c "import re; s=open('dashboard/app.py',encoding='utf-8').read(); [print(i, l) for i,l in enumerate(s.splitlines(),1) if 'Seguimiento' in l or 'Precisi' in l]"`
Leer esa sección para ver cómo se estructuran los tabs y de dónde leen los parquets.

- [ ] **Step 2: Añadir la lectura de los parquets del torneo**

Cerca de la carga de datos del tab, añadir (con `@st.cache_data` si el archivo usa ese patrón):

```python
from pathlib import Path
import pandas as pd

_TORNEO = Path(__file__).resolve().parent.parent / "outputs" / "torneo"

def _cargar_leaderboard():
    r = _TORNEO / "leaderboard_ipp.parquet"
    return pd.read_parquet(r) if r.exists() else pd.DataFrame()

def _cargar_backtest_ipp():
    r = Path(__file__).resolve().parent.parent / "outputs" / "backtest" / "metricas_ipp.parquet"
    return pd.read_parquet(r) if r.exists() else pd.DataFrame()
```

- [ ] **Step 3: Renderizar las dos capas**

Dentro del tab:

```python
st.subheader("Leaderboard de modelos IPP")
st.caption("Histórico = backtest rolling-origin. Track record real = pronósticos publicados, "
           "puntuados cuando el IPP real llegó (n = cuántos ya vencieron).")

lb = _cargar_leaderboard()
if lb.empty:
    st.info("Aún no hay pronósticos vencidos: el track record real empieza a acumularse "
            "con los próximos ciclos. Mientras, mirá el histórico del backtest abajo.")
else:
    tabla = lb.pivot_table(index="modelo", columns="horizonte",
                           values="skill_vs_naive").round(3)
    st.markdown("**Skill vs. naive (drift) — track record real** (positivo = le gana)")
    st.dataframe(tabla)
    st.markdown("**Evidencia (n_resueltos) por modelo y horizonte**")
    st.dataframe(lb.pivot_table(index="modelo", columns="horizonte",
                                values="n_resueltos"))

bt = _cargar_backtest_ipp()
if not bt.empty:
    st.markdown("**Histórico (backtest rolling-origin) — RMSE por horizonte**")
    vista = bt[bt.get("n_suficiente", True)] if "n_suficiente" in bt.columns else bt
    st.dataframe(vista.pivot_table(index="modelo", columns="horizonte",
                                   values="rmse").round(2))
```

Ajustar a los helpers/estilo reales del archivo (nombres de tabs, uso de `st.tabs`, cache).

- [ ] **Step 4: Verificar en local**

Run: `& "C:\Users\Lenovo\AppData\Local\Python\bin\python.exe" -m streamlit run dashboard/app.py`
Abrir http://localhost:8501, ir al tab "Seguimiento de Precisión", confirmar que el leaderboard renderiza (o el mensaje "aún no hay evidencia" + el histórico del backtest).

- [ ] **Step 5: Commit**

```bash
git add dashboard/app.py
git commit -m "feat(torneo): leaderboard IPP (historico + track record) en el dashboard"
```

---

## Task 8: Cierre — suite completa, .gitkeep y verificación de DoD

**Files:**
- Create: `outputs/torneo/.gitkeep`

- [ ] **Step 1: Placeholder de la carpeta versionada**

```bash
mkdir -p outputs/torneo && echo "" > outputs/torneo/.gitkeep
```

- [ ] **Step 2: Suite completa verde**

Run: `.venv\Scripts\python -m pytest tests/ -q`
Expected: todos verdes (los previos + los nuevos de torneo). Registrar el conteo.

- [ ] **Step 3: Verificar DoD del spec** (checklist §11): módulos con tests; `fabricas_ipp()` público sin regresión; ciclo idempotente; 3 parquets se generan; dashboard muestra el leaderboard; ensemble de producción sin cambios.

- [ ] **Step 4: Commit final**

```bash
git add outputs/torneo/.gitkeep
git commit -m "chore(torneo): carpeta de salida versionada + cierre"
```

---

## Notas para el ejecutor

- **TDD estricto:** cada test primero, verlo fallar, mínimo para pasar, verde, commit. No escribir implementación sin test rojo antes.
- **Verificar contra el código real:** los nombres exactos en `run_monthly_update.py` (Task 6) y `dashboard/app.py` (Task 7) hay que leerlos del archivo; el plan da el patrón, no adivines.
- **No cambiar los pesos de producción del ensemble** (fuera de alcance; es T1.1/T1.3).
- **Sin fuga:** el panel usa `congelar_drivers_futuros`; el test `test_panel_no_deja_pasar_el_futuro` lo protege.
