"""Tests del backtest mensual del IPP y de los componentes nuevos.

El test más importante es `test_congelar_drivers_no_deja_pasar_el_futuro`: sin esa función,
el "backtest" le entrega al modelo los valores futuros reales de TRM, Brent y los lags del
propio IPP, y mide una precisión que en producción no existe. Un data leak no hace fallar
nada — solo produce números demasiado buenos, que es la peor forma de estar equivocado.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from proybolsa.backtest.ipp_mensual import (
    DriftIPP,
    RandomWalkIPP,
    ThetaIPP,
    _COLS_DRIVER_FUTURO,
    backtest_ipp,
    congelar_drivers_futuros,
)
from proybolsa.models.ipp.componentes import ARIMADriftIPP, ETSDampedIPP, LGBDlogIPP


@pytest.fixture
def df_ipp() -> pd.DataFrame:
    """Serie mensual con deriva positiva y drivers que crecen con el tiempo."""
    n = 140
    rng = np.random.default_rng(3)
    fechas = pd.date_range("2015-01-01", periods=n, freq="MS")
    # dlog con media positiva: una deriva real que un modelo sin drift no puede capturar.
    dlog = rng.normal(0.004, 0.006, n)
    ipp = 100 * np.exp(np.cumsum(dlog))
    t = np.arange(n, dtype=float)
    return pd.DataFrame({
        "fecha": fechas,
        "ipp": ipp,
        "mes": fechas.month,
        "cos_mes": np.cos(2 * np.pi * fechas.month / 12),
        "sin_mes": np.sin(2 * np.pi * fechas.month / 12),
        "trm": 3000 + t * 5,
        "brent": 60 + t * 0.1,
        "brent_cop": (3000 + t * 5) * (60 + t * 0.1),
        "brent_cop_lag1m": (3000 + t * 5) * (60 + t * 0.1),
        "trm_yoy": t, "trm_yoy_lag1m": t,
        "brent_yoy": t, "brent_yoy_lag1m": t,
        "ipp_lag1m": pd.Series(ipp).shift(1).fillna(100.0),
        "ipp_lag2m": pd.Series(ipp).shift(2).fillna(100.0),
        "ipp_lag3m": pd.Series(ipp).shift(3).fillna(100.0),
        "ipp_lag12m": pd.Series(ipp).shift(12).fillna(100.0),
        "oni_lag": rng.normal(0, 0.5, n),
        "enso_el_nino": 0, "enso_la_nina": 0,
    })


# ---------------------------------------------------------------------------
# Defensa contra data leak
# ---------------------------------------------------------------------------

def test_congelar_drivers_no_deja_pasar_el_futuro(df_ipp):
    """Cada driver del tramo futuro debe quedar fijo en su último valor de entrenamiento."""
    train, futuro = df_ipp.iloc[:100], df_ipp.iloc[100:112]
    fut = congelar_drivers_futuros(futuro, train, modo="congelado")

    presentes = [c for c in _COLS_DRIVER_FUTURO if c in fut.columns]
    assert presentes, "el fixture debe traer drivers"
    for c in presentes:
        assert fut[c].nunique() == 1, f"{c} varía en el futuro: hay leak"
        assert fut[c].iloc[0] == pytest.approx(train[c].iloc[-1]), f"{c} no quedó en el último de train"


def test_congelar_preserva_el_calendario(df_ipp):
    """La fecha y los términos del mes SÍ se conocen para el futuro: no deben congelarse."""
    train, futuro = df_ipp.iloc[:100], df_ipp.iloc[100:112]
    fut = congelar_drivers_futuros(futuro, train, modo="congelado")

    assert list(fut["fecha"]) == list(futuro["fecha"])
    assert fut["cos_mes"].nunique() > 1, "cos_mes se congeló y es determinista"


def test_modo_perfecto_deja_pasar_todo(df_ipp):
    """El modo 'perfecto' existe para medir la cota superior del valor de los drivers."""
    train, futuro = df_ipp.iloc[:100], df_ipp.iloc[100:112]
    fut = congelar_drivers_futuros(futuro, train, modo="perfecto")
    pd.testing.assert_frame_equal(fut, futuro)


def test_modo_escenario_exige_trayectoria(df_ipp):
    with pytest.raises(ValueError, match="df_escenario"):
        congelar_drivers_futuros(df_ipp.iloc[100:], df_ipp.iloc[:100], modo="escenario")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [RandomWalkIPP, DriftIPP, ThetaIPP])
def test_benchmark_forecast_bien_formado(df_ipp, cls):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = cls().fit(df_ipp)
        fc = m.forecast(12)

    assert len(fc) == 12
    assert {"pred", "ci_lo90", "ci_hi90"} <= set(fc.columns)
    assert (fc["pred"] > 0).all()
    if fc["ci_lo90"].notna().all():
        assert (fc["ci_lo90"] <= fc["pred"]).all()
        assert (fc["pred"] <= fc["ci_hi90"]).all()


def test_random_walk_es_plano_y_su_banda_se_abre(df_ipp):
    m = RandomWalkIPP().fit(df_ipp)
    fc = m.forecast(24)
    assert fc["pred"].nunique() == 1, "el random walk debe proyectar el último valor"
    ancho = fc["ci_hi90"] - fc["ci_lo90"]
    assert ancho.is_monotonic_increasing, "la banda de un paseo aleatorio crece con el horizonte"


def test_drift_sigue_la_deriva(df_ipp):
    """Con una serie que sube, el drift debe proyectar hacia arriba."""
    m = DriftIPP().fit(df_ipp)
    fc = m.forecast(24)
    assert fc["pred"].iloc[-1] > fc["pred"].iloc[0]
    assert fc["pred"].iloc[-1] > df_ipp["ipp"].iloc[-1]


def test_drift_con_ventana_usa_solo_los_ultimos_meses(df_ipp):
    completo = DriftIPP().fit(df_ipp)
    ventana = DriftIPP(ventana=12).fit(df_ipp)
    assert completo._drift != ventana._drift


# ---------------------------------------------------------------------------
# Componentes nuevos
# ---------------------------------------------------------------------------

def test_arima_drift_proyecta_la_tendencia(df_ipp):
    """El bug que motiva toda la fase: sin drift el pronóstico sale plano."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ARIMADriftIPP(p_max=2, q_max=1).fit(df_ipp)
        fc = m.forecast(24)

    ultimo = df_ipp["ipp"].iloc[-1]
    assert fc["pred"].iloc[-1] > ultimo * 1.02, (
        f"a 24 meses proyecta {fc['pred'].iloc[-1]:.1f} contra un último de {ultimo:.1f}: "
        "sigue plano, el término de deriva no se está aplicando"
    )
    assert m.diagnostico["trend"] == "c", "debió elegir un modelo con deriva"


def test_arima_drift_reporta_su_diagnostico(df_ipp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ARIMADriftIPP(p_max=1, q_max=1).fit(df_ipp)
    d = m.diagnostico
    assert set(d) >= {"order", "trend", "aic", "bic", "n_candidatos_evaluados"}
    assert d["n_candidatos_evaluados"] >= 1


def test_arima_drift_orden_fijo_no_busca(df_ipp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ARIMADriftIPP(orden_fijo=(1, 1, 0), trend_fijo="c").fit(df_ipp)
    assert m.diagnostico["order"] == (1, 1, 0)
    assert m.diagnostico["n_candidatos_evaluados"] == 1


def test_el_default_es_orden_fijo_con_deriva(df_ipp):
    """Guardarraíl del hallazgo central de la Fase 2.

    Medido sobre 56 orígenes reales, buscar el orden por AIC dentro de cada origen casi
    DUPLICA el error a 24 meses (48.01 contra 24.11 del orden fijo): la rejilla elige un
    modelo sin deriva en 7 de 56 orígenes y eso basta para arruinar el horizonte largo.
    Si alguien vuelve a poner la búsqueda como default, este test lo detiene.
    """
    m = ARIMADriftIPP()
    assert m.reseleccionar_orden is False, "el default NO debe buscar el orden"
    assert m.orden_fijo == (1, 1, 0)
    assert m.trend_fijo == "c", "el default debe llevar deriva"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ajustado = m.fit(df_ipp)
    assert ajustado.diagnostico["n_candidatos_evaluados"] == 1


def test_reseleccionar_orden_reactiva_la_busqueda(df_ipp):
    """El default es fijo, pero el estudio necesita poder pedir la rejilla explícitamente."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ARIMADriftIPP(reseleccionar_orden=True, p_max=1, q_max=1).fit(df_ipp)
    assert m.diagnostico["n_candidatos_evaluados"] > 1


def test_ets_damped_bien_formado(df_ipp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = ETSDampedIPP().fit(df_ipp).forecast(12)
    assert len(fc) == 12
    assert (fc["pred"] > 0).all()


def test_lgb_dlog_extrapola(df_ipp):
    """El LGB anterior modelaba el NIVEL con árboles y no podía salirse del rango visto."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LGBDlogIPP().fit(df_ipp)
        fc = m.forecast(12, exog_future=df_ipp.tail(12).reset_index(drop=True))

    assert len(fc) == 12
    assert (fc["pred"] > 0).all()
    assert (fc["ci_lo90"] <= fc["pred"]).all() and (fc["pred"] <= fc["ci_hi90"]).all()
    ancho = (fc["ci_hi90"] - fc["ci_lo90"]).to_numpy()
    assert ancho[-1] > ancho[0], "la banda debe abrirse con el horizonte"


def test_lgb_dlog_conserva_el_alias_predict(df_ipp):
    """`EnsembleIPP._calibrar_pesos` llama `.predict(df_val)`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LGBDlogIPP().fit(df_ipp)
        p = m.predict(df_ipp.tail(6).reset_index(drop=True))
    assert len(p) == 6


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

def test_backtest_ipp_mismos_origenes_para_todos(df_ipp):
    err, met = backtest_ipp(
        df_ipp, ["ipp_bench_rw", "ipp_bench_drift"],
        horizontes=(1, 6), min_train=80, step=5,
    )
    assert not met.empty
    assert set(met["variable"]) == {"ipp"}
    n = met.groupby("modelo")["n_predicciones"].apply(set)
    assert all(len(s) == 1 for s in n), "cada modelo debe tener el mismo n en todos los horizontes"
    assert len(set(map(lambda s: next(iter(s)), n))) == 1, "todos los modelos, el mismo n"


def test_backtest_rechaza_modelo_desconocido(df_ipp):
    with pytest.raises(KeyError, match="desconocidos"):
        backtest_ipp(df_ipp, ["no_existe"], min_train=80)


def test_el_error_crece_con_el_horizonte(df_ipp):
    _, met = backtest_ipp(df_ipp, ["ipp_bench_rw"], horizontes=(1, 12), min_train=80, step=5)
    r = met.set_index("horizonte")["rmse"]
    assert r.loc[12] > r.loc[1], "el error a 12 meses debe superar al de 1 mes"
