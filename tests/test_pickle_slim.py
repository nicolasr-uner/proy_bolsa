"""Round-trip del pickle compacto de los wrappers SARIMAX.

El objetivo del protocolo (`proybolsa/models/sarimax_pickle.py`) es no serializar el objeto
`SARIMAXResults`, que arrastra las matrices del filtro de Kalman de toda la muestra (~21 MB
en el modelo de bolsa). Estos tests fijan las dos propiedades que lo hacen seguro:

  1. El forecast tras deserializar es **idéntico**, punto y bandas.
  2. Un pickle antiguo (con `_result` dentro) sigue cargando con el código nuevo — es lo que
     evita tumbar la app pública en el despliegue donde coexisten .pkl viejo y código nuevo.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from proybolsa.models.bolsa.nivel_diario import SARIMAXNivel
from proybolsa.models.ipp.modelo_ipp import SARIMABaselineIPP, SARIMAXDriversIPP
from proybolsa.models.sarimax_pickle import VERSION_PICKLE


@pytest.fixture
def df_bolsa() -> pd.DataFrame:
    """Serie diaria sintética con los exógenos que consume SARIMAXNivel.

    Se construye en log-espacio (que es donde trabaja el modelo) como un AR(1) con reversión
    a la media más un ciclo semanal, para que el SARIMAX(1,1,1)(1,0,1,7) esté bien
    identificado. Un random walk con clip produce tramos planos, el modelo queda mal
    identificado y el forecast desborda a inf.
    """
    n = 240
    rng = np.random.default_rng(7)
    fechas = pd.date_range("2024-01-01", periods=n, freq="D")

    ar = np.zeros(n)
    ruido = rng.normal(0, 0.05, n)
    for i in range(1, n):
        ar[i] = 0.7 * ar[i - 1] + ruido[i]
    log_precio = np.log(450) + 0.12 * np.sin(2 * np.pi * np.arange(n) / 7) + ar

    return pd.DataFrame({
        "fecha": fechas,
        "precio_bolsa_mean": np.exp(log_precio),
        "mes": fechas.month,
        "aportes_pct": 100 + rng.normal(0, 10, n),
        "volumen_util_pct": 60 + rng.normal(0, 5, n),
    })


@pytest.fixture
def df_ipp() -> pd.DataFrame:
    """Serie mensual sintética con los exógenos que consume SARIMAXDriversIPP."""
    n = 120
    rng = np.random.default_rng(11)
    fechas = pd.date_range("2016-01-01", periods=n, freq="MS")
    ipp = 100 * np.exp(np.cumsum(rng.normal(0.004, 0.006, n)))
    return pd.DataFrame({
        "fecha": fechas,
        "ipp": ipp,
        "mes": fechas.month,
        "cos_mes": np.cos(2 * np.pi * fechas.month / 12),
        "sin_mes": np.sin(2 * np.pi * fechas.month / 12),
        "brent_yoy_lag1m": rng.normal(5, 20, n),
        "trm_yoy_lag1m": rng.normal(3, 12, n),
        "enso_el_nino": rng.integers(0, 2, n),
    })


def _round_trip(obj):
    blob = pickle.dumps(obj)
    return pickle.loads(blob), len(blob)


def _assert_forecast_identico(a, b, horizon, exog=None, tol=0.0):
    fa = a.forecast(horizon, exog_future=exog) if exog is not None else a.forecast(horizon)
    fb = b.forecast(horizon, exog_future=exog) if exog is not None else b.forecast(horizon)
    for col in ("pred", "ci_lo90", "ci_hi90"):
        diff = np.abs(fa[col].values - fb[col].values).max()
        assert diff <= tol, f"{col}: diff maxima {diff} > {tol}"
    return fa


def test_sarimax_nivel_round_trip_identico(df_bolsa):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAXNivel().fit(df_bolsa)

    futuro = df_bolsa.tail(14).reset_index(drop=True)
    m2, _ = _round_trip(m)

    assert m2._result is not None, "el resultado no se reconstruyó al deserializar"
    _assert_forecast_identico(m, m2, 14, exog=futuro)


def test_sarimax_nivel_no_serializa_el_results(df_bolsa):
    """El estado NO debe llevar el SARIMAXResults, solo los parámetros."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAXNivel().fit(df_bolsa)

    estado = m.__getstate__()
    assert "_result" not in estado
    assert estado["_version_pickle"] == VERSION_PICKLE
    assert isinstance(estado["_params"], np.ndarray)

    # El pickle del wrapper con 240 obs debe ser holgadamente chico. El umbral es amplio
    # a propósito: fija el orden de magnitud (KB, no MB), no un número frágil.
    _, tamano = _round_trip(m)
    assert tamano < 512_000, f"pickle de {tamano} bytes: el results se está colando"


def test_pickle_legacy_sigue_cargando(df_bolsa):
    """Un estado v1 (con _result completo) debe cargar sin excepción y pronosticar igual."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAXNivel().fit(df_bolsa)

    # Estado al estilo viejo: __dict__ crudo, con _result y sin _params.
    estado_v1 = dict(m.__dict__)
    assert "_result" in estado_v1

    legacy = SARIMAXNivel.__new__(SARIMAXNivel)
    legacy.__setstate__(estado_v1)

    futuro = df_bolsa.tail(10).reset_index(drop=True)
    _assert_forecast_identico(m, legacy, 10, exog=futuro)


def test_sarima_baseline_ipp_round_trip(df_ipp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMABaselineIPP().fit(df_ipp)

    m2, _ = _round_trip(m)
    assert m2._result is not None
    assert m2._ipp_init == pytest.approx(m._ipp_init)
    _assert_forecast_identico(m, m2, 12)


def test_sarimax_drivers_ipp_round_trip(df_ipp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAXDriversIPP().fit(df_ipp)

    futuro = df_ipp.tail(12).reset_index(drop=True)
    m2, _ = _round_trip(m)
    assert m2._result is not None
    assert m2._exog_cols == m._exog_cols, "se perdieron las columnas exógenas"
    _assert_forecast_identico(m, m2, 12, exog=futuro)


def test_modelo_sin_ajustar_sobrevive_el_pickle():
    """Un wrapper no ajustado debe poder serializarse sin explotar."""
    m2, _ = _round_trip(SARIMAXNivel())
    assert m2._result is None
    with pytest.raises(RuntimeError):
        m2.forecast(5)
