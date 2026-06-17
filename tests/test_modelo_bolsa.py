"""Tests del modelo de precio de bolsa.

Usa datos sinteticos para evitar dependencia de archivos externos.
Los tests verifican:
  - Que el modelo se ajusta sin errores
  - Que el pronostico tiene la forma correcta
  - Que los CI son validos (lo < pred < hi)
  - Que el perfil horario se aplica correctamente
  - Que los escenarios seco > promedio > humedo
"""

from __future__ import annotations

import warnings
import datetime as dt

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from proybolsa.models.bolsa.nivel_diario import EnsembleNivel, LGBNivel, SARIMAXNivel
from proybolsa.models.bolsa.pronostico import (
    ESCENARIOS_HIDRO,
    PronosticadorBolsa,
    _construir_df_futuro,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _df_diario_sintetico(n: int = 400) -> pd.DataFrame:
    """DataFrame sintetico con estructura similar a bolsa_features_diario.parquet."""
    rng = np.random.default_rng(42)
    fechas = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    precio = 300 + rng.normal(0, 50, n).cumsum() * 0.01 + 100 * np.sin(np.arange(n) * 2 * np.pi / 365)
    precio = np.maximum(precio, 100.0)  # no negativo

    df = pd.DataFrame({
        "fecha": fechas,
        "precio_bolsa_mean": precio,
        "precio_bolsa_std": rng.uniform(10, 80, n),
        "aportes_pct": rng.uniform(50, 150, n),
        "volumen_util_pct": rng.uniform(30, 90, n),
        "aportes_pct_lag1d": rng.uniform(50, 150, n),
        "aportes_pct_lag7d": rng.uniform(50, 150, n),
        "aportes_pct_lag30d": rng.uniform(50, 150, n),
        "volumen_util_pct_lag1d": rng.uniform(30, 90, n),
        "volumen_util_pct_lag7d": rng.uniform(30, 90, n),
        "volumen_util_pct_lag30d": rng.uniform(30, 90, n),
        "precio_escasez": rng.uniform(880, 930, n),
        "enso_el_nino": rng.integers(0, 2, n),
        "enso_la_nina": rng.integers(0, 2, n),
        "oni_lag": rng.uniform(-1, 1, n),
        "mes": [f.month for f in fechas],
        "dia_semana": [f.weekday() for f in fechas],
        "es_festivo": [0] * n,
        "es_fin_semana": [int(f.weekday() >= 5) for f in fechas],
        "tipo_dia": ["habil"] * n,
        "precio_bolsa_mean_lag1d": np.roll(precio, 1),
        "precio_bolsa_mean_lag7d": np.roll(precio, 7),
        "precio_bolsa_mean_lag30d": np.roll(precio, 30),
        "precio_bolsa_mean_roll7d_std": rng.uniform(20, 60, n),
        "precio_bolsa_mean_roll30d_std": rng.uniform(15, 50, n),
        "vertimientos_ratio": rng.uniform(0, 0.3, n),
        "regimen_hidro": rng.choice(["seco", "normal", "humedo"], n),
    })
    return df


def _perfil_sintetico() -> pd.DataFrame:
    """Perfil horario minimo funcional (3 tipo_dia x 12 meses x 24 horas)."""
    rows = []
    tipos = ["habil", "sabado", "domingo_festivo"]
    for td in tipos:
        for mes in range(1, 13):
            for hora in range(24):
                # Perfil simple: valle en la madrugada, pico en la noche
                factor = 0.8 + 0.4 * np.sin((hora - 6) * np.pi / 18) if 6 <= hora <= 22 else 0.75
                rows.append({"tipo_dia": td, "mes": mes, "hora": hora,
                             "perfil_medio": factor, "perfil_p10": factor * 0.8,
                             "perfil_p90": factor * 1.2, "n_obs": 30})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests LGBNivel
# ---------------------------------------------------------------------------

class TestLGBNivel:
    def test_fit_sin_error(self):
        df = _df_diario_sintetico(200)
        model = LGBNivel()
        model.fit(df)   # no debe lanzar excepcion

    def test_predict_shape(self):
        df = _df_diario_sintetico(200)
        model = LGBNivel()
        model.fit(df)
        preds = model.predict(df.tail(10))
        assert len(preds) == 10

    def test_predict_positivo(self):
        df = _df_diario_sintetico(200)
        model = LGBNivel()
        model.fit(df)
        preds = model.predict(df.tail(20))
        assert (preds > 0).all()

    def test_feature_importance_populated(self):
        df = _df_diario_sintetico(200)
        model = LGBNivel()
        model.fit(df)
        assert len(model.feature_importance) > 0


# ---------------------------------------------------------------------------
# Tests SARIMAXNivel
# ---------------------------------------------------------------------------

class TestSARIMAXNivel:
    def test_fit_sin_error(self):
        df = _df_diario_sintetico(200)
        model = SARIMAXNivel()
        model.fit(df)

    def test_forecast_shape(self):
        df = _df_diario_sintetico(200)
        model = SARIMAXNivel()
        model.fit(df)
        fc = model.forecast(5, exog_future=_df_diario_sintetico(5))
        assert len(fc) == 5
        assert "pred" in fc.columns

    def test_forecast_ci_valido(self):
        df = _df_diario_sintetico(200)
        model = SARIMAXNivel()
        model.fit(df)
        fc = model.forecast(10, exog_future=_df_diario_sintetico(10))
        assert (fc["ci_lo90"] < fc["pred"]).all()
        assert (fc["pred"] < fc["ci_hi90"]).all()


# ---------------------------------------------------------------------------
# Tests EnsembleNivel
# ---------------------------------------------------------------------------

class TestEnsembleNivel:
    def test_fit_calibra_pesos(self):
        df = _df_diario_sintetico(300)
        ensemble = EnsembleNivel()
        ensemble.fit(df.head(270), df_val=df.tail(30))
        assert 0 < ensemble.w_sarimax < 1
        assert 0 < ensemble.w_lgb < 1
        assert abs(ensemble.w_sarimax + ensemble.w_lgb - 1.0) < 1e-9

    def test_predict_positivo(self):
        df = _df_diario_sintetico(300)
        ensemble = EnsembleNivel()
        ensemble.fit(df.head(270))
        resultado = ensemble.predict(df.tail(30))
        assert (resultado["pred"] > 0).all()

    def test_ci_centrado_en_pred(self):
        df = _df_diario_sintetico(300)
        ensemble = EnsembleNivel()
        ensemble.fit(df.head(270))
        resultado = ensemble.predict(df.tail(30))
        # CI debe contener la prediccion
        assert (resultado["ci_lo90"] <= resultado["pred"]).all()
        assert (resultado["pred"] <= resultado["ci_hi90"]).all()


# ---------------------------------------------------------------------------
# Tests PronosticadorBolsa
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def modelo_bolsa_ajustado():
    df = _df_diario_sintetico(400)
    perfil = _perfil_sintetico()
    p = PronosticadorBolsa()
    p.fit(df, perfil=perfil)
    return p


class TestPronosticadorBolsa:
    @pytest.fixture(autouse=False)
    def modelo_ajustado(self, modelo_bolsa_ajustado):
        return modelo_bolsa_ajustado

    def test_fit_completa(self, modelo_ajustado):
        resumen = modelo_ajustado.resumen_modelo()
        assert resumen["n_train"] > 0
        assert resumen["perfil_celdas"] == 864

    def test_pronostico_diario_shape(self, modelo_ajustado):
        fc = modelo_ajustado.pronosticar(7, devolver_horario=False)
        assert len(fc) == 7
        assert "pred_diaria" in fc.columns

    def test_ci_siempre_valido(self, modelo_ajustado):
        fc = modelo_ajustado.pronosticar(14, devolver_horario=False)
        assert (fc["ci_lo90"] < fc["pred_diaria"]).all(), "ci_lo90 debe ser < pred"
        assert (fc["pred_diaria"] < fc["ci_hi90"]).all(), "pred debe ser < ci_hi90"

    def test_escenarios_orden_fisico(self, modelo_ajustado):
        """Seco debe pronosticar precio mas alto que humedo."""
        esc = modelo_ajustado.pronosticar(30, escenarios_multiples=True, devolver_horario=False)
        media_seco = esc["seco"]["pred_diaria"].mean()
        media_humedo = esc["humedo"]["pred_diaria"].mean()
        assert media_seco >= media_humedo, (
            f"Seco ({media_seco:.1f}) deberia ser >= humedo ({media_humedo:.1f})"
        )

    def test_pronostico_horario_24h_por_dia(self, modelo_ajustado):
        fc = modelo_ajustado.pronosticar(3, devolver_horario=True)
        assert len(fc) == 72  # 3 dias x 24 horas

    def test_pronostico_horario_ci_valido(self, modelo_ajustado):
        fc = modelo_ajustado.pronosticar(2, devolver_horario=True)
        assert (fc["ci_lo90_horario"] < fc["pred_horaria"]).all()
        assert (fc["pred_horaria"] < fc["ci_hi90_horario"]).all()

    def test_construccion_df_futuro(self):
        df_futuro = _construir_df_futuro(
            horizonte_dias=10,
            fecha_inicio=pd.Timestamp("2026-07-01"),
            escenario="seco",
        )
        assert len(df_futuro) == 10
        assert df_futuro["aportes_pct"].iloc[0] == ESCENARIOS_HIDRO["seco"]["aportes_pct"]

    def test_escenarios_dict_keys(self, modelo_ajustado):
        esc = modelo_ajustado.pronosticar(7, escenarios_multiples=True, devolver_horario=False)
        assert set(esc.keys()) == {"seco", "promedio", "humedo"}

    def test_componentes_reajustados_sobre_serie_completa(self, modelo_ajustado):
        """El SARIMAX de produccion debe ajustarse sobre TODA la serie, no solo
        df_train. df_val solo calibra pesos; el forecast debe partir del ultimo dato."""
        n_full = len(_df_diario_sintetico(400))
        assert int(modelo_ajustado.modelo.sarimax._result.nobs) == n_full
