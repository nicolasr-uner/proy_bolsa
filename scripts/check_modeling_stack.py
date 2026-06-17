"""Smoke test del stack de modelado (Fase 1).

Verifica que las librerias criticas importan y que los dos motores principales
(statsforecast/AutoARIMA con numba, y statsmodels/SARIMAX) corren de punta a
punta en este entorno (Python 3.12 + numpy 2.1.x + pandas 2.2.x).
"""

import importlib

modules = [
    "numpy", "pandas", "statsforecast", "statsmodels", "lightgbm",
    "mlforecast", "utilsforecast", "sklearn", "pandera", "yaml",
    "matplotlib", "plotly", "pytest",
]
print("=== versiones ===")
for name in modules:
    try:
        mod = importlib.import_module(name)
        print(f"  {name}: {getattr(mod, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name}: IMPORT FALLO -> {exc!r}")

import numpy as np
import pandas as pd

# Serie sintetica diaria con estacionalidad semanal + tendencia.
n = 180
rng = np.random.default_rng(0)
trend = np.linspace(100, 130, n)
weekly = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
y = trend + weekly + rng.normal(0, 3, n)
dates = pd.date_range("2025-01-01", periods=n, freq="D")

print("\n=== statsforecast / AutoARIMA (compila numba) ===")
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    sf_df = pd.DataFrame({"unique_id": "s1", "ds": dates, "y": y})
    sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D")
    fc = sf.forecast(df=sf_df, h=14)
    print("  OK -> forecast shape:", fc.shape)
    print("  ultimos 3:", np.round(fc["AutoARIMA"].tail(3).to_numpy(), 2))
except Exception as exc:  # noqa: BLE001
    print("  FALLO:", repr(exc))

print("\n=== statsmodels / SARIMAX (con exogena) ===")
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    x = rng.normal(0, 1, n)
    y2 = y + 5 * x
    model = SARIMAX(
        y2, exog=x, order=(1, 1, 1), seasonal_order=(1, 0, 0, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    pred = res.forecast(steps=7, exog=rng.normal(0, 1, 7))
    print("  OK -> AIC:", round(res.aic, 1), "| pred[:3]:", np.round(pred[:3], 2))
except Exception as exc:  # noqa: BLE001
    print("  FALLO:", repr(exc))

print("\n=== lightgbm (smoke) ===")
try:
    import lightgbm as lgb
    from sklearn.datasets import make_regression

    xb, yb = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=0)
    booster = lgb.LGBMRegressor(n_estimators=20, verbose=-1).fit(xb, yb)
    print("  OK -> score:", round(booster.score(xb, yb), 3))
except Exception as exc:  # noqa: BLE001
    print("  FALLO:", repr(exc))

print("\n=== FIN smoke test ===")
