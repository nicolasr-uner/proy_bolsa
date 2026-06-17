"""Tests del modulo de backtesting rolling-origin."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from proybolsa.backtest.rolling_origin import (
    RollingOriginBacktest,
    calcular_metricas,
    diebold_mariano,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _serie_sintetica(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    fechas = [dt.date(2022, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    precio = 300 + np.cumsum(rng.normal(0, 5, n))
    precio = np.maximum(precio, 50.0)
    return pd.DataFrame({"fecha": fechas, "y": precio})


def _fn_fit_naive(df_train: pd.DataFrame):
    """Modelo naive: pronostica el ultimo valor."""
    return {"ultimo": df_train["y"].iloc[-1]}


def _fn_predict_naive(modelo: dict, df_test: pd.DataFrame, h: int) -> pd.DataFrame:
    """Pronostico naive con CI ±50."""
    pred = modelo["ultimo"]
    return pd.DataFrame({
        "pred": [pred] * h,
        "ci_lo90": [pred - 50] * h,
        "ci_hi90": [pred + 50] * h,
    })


# ---------------------------------------------------------------------------
# Tests metricas
# ---------------------------------------------------------------------------

class TestMetricas:
    def _errores(self):
        rng = np.random.default_rng(1)
        y_real = 400 + rng.normal(0, 30, 60)
        y_pred = y_real + rng.normal(5, 20, 60)   # sesgo positivo de ~5
        h = [1] * 30 + [7] * 30
        return pd.DataFrame({
            "horizonte": h,
            "y_real": y_real,
            "y_pred": y_pred,
            "ci_lo90": y_pred - 80,
            "ci_hi90": y_pred + 80,
        })

    def test_metricas_devuelve_df(self):
        df = calcular_metricas(self._errores())
        assert isinstance(df, pd.DataFrame)
        assert "rmse" in df.columns
        assert "mae" in df.columns
        assert "mape_pct" in df.columns
        assert "sesgo" in df.columns

    def test_metricas_por_horizonte(self):
        df = calcular_metricas(self._errores())
        assert set(df["horizonte"].tolist()) == {1, 7}

    def test_sesgo_positivo(self):
        # y_pred = y_real + 5 + ruido -> sesgo ~ +5
        df = calcular_metricas(self._errores())
        sesgo_h1 = df.loc[df["horizonte"] == 1, "sesgo"].iloc[0]
        assert sesgo_h1 > 0, "Sesgo deberia ser positivo"

    def test_cobertura_ci(self):
        df = calcular_metricas(self._errores())
        cob = df["cobertura_ci90_pct"].iloc[0]
        assert 0 <= cob <= 100

    def test_rmse_mayor_mae(self):
        df = calcular_metricas(self._errores())
        for _, row in df.iterrows():
            assert row["rmse"] >= row["mae"], "RMSE siempre >= MAE"

    def test_nan_no_rompe(self):
        errores = pd.DataFrame({
            "horizonte": [1, 1, 1],
            "y_real": [400, np.nan, 380],
            "y_pred": [410, 420, np.nan],
        })
        df = calcular_metricas(errores)
        assert df["rmse"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests Diebold-Mariano
# ---------------------------------------------------------------------------

class TestDieboldMariano:
    def test_identical_models_no_diferencia(self):
        rng = np.random.default_rng(3)
        e = pd.Series(rng.normal(0, 10, 100) ** 2)
        res = diebold_mariano(e, e, h=1)
        assert res["p_valor"] > 0.05 or np.isnan(res["p_valor"])

    def test_modelo_a_mejor(self):
        rng = np.random.default_rng(4)
        e_a = pd.Series(rng.normal(0, 5, 200) ** 2)     # errores pequenos
        e_b = pd.Series(rng.normal(0, 30, 200) ** 2)    # errores grandes
        res = diebold_mariano(e_a, e_b, h=1, alternativa="a_mejor")
        assert res["p_valor"] < 0.05, "A deberia ser significativamente mejor"

    def test_pocos_datos_devuelve_nan(self):
        e = pd.Series([1.0, 2.0, 3.0])
        res = diebold_mariano(e, e, h=1)
        assert res["estadistico_dm"] != res["estadistico_dm"] or "insuficientes" in res["conclusion"]

    def test_keys_presentes(self):
        e = pd.Series(np.random.default_rng(5).normal(0, 1, 100) ** 2)
        res = diebold_mariano(e, e * 2, h=1)
        assert "estadistico_dm" in res
        assert "p_valor" in res
        assert "conclusion" in res


# ---------------------------------------------------------------------------
# Tests RollingOriginBacktest
# ---------------------------------------------------------------------------

class TestRollingOriginBacktest:
    def test_evaluar_basico(self):
        df = _serie_sintetica(500)
        bt = RollingOriginBacktest(
            horizonte_max=7, step_dias=14,
            min_train_dias=100, horizontes_evaluar=[1, 7],
        )
        errores, metricas = bt.evaluar(
            df, "fecha", "y",
            fn_fit=_fn_fit_naive,
            fn_predict=_fn_predict_naive,
            verbose=False,
        )
        assert len(errores) > 0
        assert "horizonte" in errores.columns
        assert "rmse" in metricas.columns

    def test_metricas_por_horizonte_correctas(self):
        df = _serie_sintetica(500)
        bt = RollingOriginBacktest(
            horizonte_max=7, step_dias=14,
            min_train_dias=100, horizontes_evaluar=[1, 7],
        )
        _, metricas = bt.evaluar(
            df, "fecha", "y",
            fn_fit=_fn_fit_naive,
            fn_predict=_fn_predict_naive,
            verbose=False,
        )
        assert set(metricas["horizonte"].tolist()) == {1, 7}

    def test_rmse_h7_mayor_h1(self):
        """El error a 7 dias deberia ser >= al de 1 dia para un modelo simple."""
        df = _serie_sintetica(500)
        bt = RollingOriginBacktest(
            horizonte_max=7, step_dias=7,
            min_train_dias=150, horizontes_evaluar=[1, 7],
        )
        _, metricas = bt.evaluar(
            df, "fecha", "y",
            fn_fit=_fn_fit_naive,
            fn_predict=_fn_predict_naive,
            verbose=False,
        )
        rmse_1 = metricas.loc[metricas["horizonte"] == 1, "rmse"].iloc[0]
        rmse_7 = metricas.loc[metricas["horizonte"] == 7, "rmse"].iloc[0]
        assert rmse_7 >= rmse_1, "RMSE a 7 dias debe ser >= 1 dia"

    def test_pocos_datos_levanta_error(self):
        df = _serie_sintetica(50)
        bt = RollingOriginBacktest(min_train_dias=365, horizonte_max=30)
        with pytest.raises(ValueError, match="suficientes datos"):
            bt.evaluar(df, "fecha", "y", _fn_fit_naive, _fn_predict_naive, verbose=False)

    def test_refit_cada_no_rompe(self):
        df = _serie_sintetica(400)
        bt = RollingOriginBacktest(
            horizonte_max=7, step_dias=14,
            min_train_dias=100, horizontes_evaluar=[1],
            refit_cada=3,
        )
        errores, _ = bt.evaluar(
            df, "fecha", "y",
            fn_fit=_fn_fit_naive,
            fn_predict=_fn_predict_naive,
            verbose=False,
        )
        assert len(errores) > 0

    def test_n_predicciones_coherente(self):
        df = _serie_sintetica(500)
        bt = RollingOriginBacktest(
            horizonte_max=7, step_dias=14,
            min_train_dias=100, horizontes_evaluar=[1, 7],
        )
        errores, metricas = bt.evaluar(
            df, "fecha", "y",
            fn_fit=_fn_fit_naive,
            fn_predict=_fn_predict_naive,
            verbose=False,
        )
        n_h1 = metricas.loc[metricas["horizonte"] == 1, "n_predicciones"].iloc[0]
        n_h7 = metricas.loc[metricas["horizonte"] == 7, "n_predicciones"].iloc[0]
        assert n_h1 == n_h7, "Ambos horizontes deben tener el mismo numero de origenes"
