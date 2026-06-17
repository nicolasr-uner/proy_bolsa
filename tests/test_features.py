"""Tests unitarios para src/proybolsa/features/.

Todos sin red. Se usan fechas reales conocidas para verificar festivos colombianos.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from proybolsa.features.calendar import (
    agregar_features_calendario,
    agregar_features_fecha,
    festivos_colombia,
    tipo_dia,
)
from proybolsa.features.enso import agregar_features_enso, clasificar_enso, dummy_enso
from proybolsa.features.hydro import (
    clasificar_regimen_hidro,
    escalar_fracciones,
)
from proybolsa.features.lags import (
    agregar_lags_diarios,
    agregar_lags_horarios,
    agregar_rolling,
)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

class TestFestivosYTipoDia:
    def test_navidad_es_festivo(self):
        assert dt.date(2025, 12, 25) in festivos_colombia(2025)

    def test_independencia_es_festivo(self):
        assert dt.date(2025, 7, 20) in festivos_colombia(2025)

    def test_lunes_habil_no_festivo(self):
        # 2025-06-16 fue lunes, no festivo
        assert tipo_dia(dt.date(2025, 6, 16)) == "habil"

    def test_sabado_es_sabado(self):
        # 2025-06-14 fue sabado
        assert tipo_dia(dt.date(2025, 6, 14)) == "sabado"

    def test_domingo_es_domingo_festivo(self):
        # 2025-06-15 fue domingo
        assert tipo_dia(dt.date(2025, 6, 15)) == "domingo_festivo"

    def test_festivo_en_lunes(self):
        # Navidad 2025 = jueves, pero el 8 diciembre (Inmaculada) es festivo
        # 2025-12-08 = lunes -> festivo
        assert tipo_dia(dt.date(2025, 12, 8)) == "domingo_festivo"

    def test_año_nuevo_es_festivo(self):
        assert tipo_dia(dt.date(2026, 1, 1)) == "domingo_festivo"  # Jueves y festivo


class TestAgregarFeaturesCalendario:
    def _df_horario(self):
        ts = pd.date_range("2025-06-16 00:00", periods=48, freq="h")  # lun+mar
        return pd.DataFrame({"timestamp": ts, "precio": range(48)})

    def test_columnas_agregadas(self):
        df = agregar_features_calendario(self._df_horario())
        for col in ("hora", "dia_semana", "mes", "anio", "tipo_dia", "es_festivo", "es_fin_semana"):
            assert col in df.columns, f"Falta columna: {col}"

    def test_hora_range(self):
        df = agregar_features_calendario(self._df_horario())
        assert df["hora"].min() == 0
        assert df["hora"].max() == 23

    def test_tipo_dia_correcto(self):
        df = agregar_features_calendario(self._df_horario())
        # 2025-06-16 = lunes, no festivo -> habil
        assert (df[df["hora"] == 0]["tipo_dia"] == "habil").all()

    def test_mes_correcto(self):
        df = agregar_features_calendario(self._df_horario())
        assert (df["mes"] == 6).all()


# ---------------------------------------------------------------------------
# Hydro
# ---------------------------------------------------------------------------

class TestEscalarFracciones:
    def _df(self):
        return pd.DataFrame({"aportes_pct": [0.73, 1.20, 0.45], "volumen_util_pct": [0.69, 0.80, 0.35]})

    def test_escala_a_porcentaje(self):
        df = escalar_fracciones(self._df())
        assert df["aportes_pct"].iloc[0] == pytest.approx(73.0)
        assert df["volumen_util_pct"].iloc[0] == pytest.approx(69.0)

    def test_no_toca_otras_columnas(self):
        df_orig = self._df()
        df_orig["otra"] = 5.0
        df = escalar_fracciones(df_orig)
        assert df["otra"].iloc[0] == 5.0


class TestClasificarRegimenHidro:
    def _df(self, aportes, embalse):
        return pd.DataFrame({"aportes_pct": [aportes], "volumen_util_pct": [embalse]})

    def test_seco_aportes_bajos(self):
        df = clasificar_regimen_hidro(self._df(aportes_pct := 50, embalse_pct := 60))
        assert df["regimen_hidro"].iloc[0] == "seco"

    def test_humedo_aportes_altos_y_embalse_lleno(self):
        df = clasificar_regimen_hidro(self._df(aportes_pct := 160, embalse_pct := 80))
        assert df["regimen_hidro"].iloc[0] == "humedo"

    def test_normal(self):
        df = clasificar_regimen_hidro(self._df(aportes_pct := 100, embalse_pct := 60))
        assert df["regimen_hidro"].iloc[0] == "normal"

    def test_walrus_compat(self):
        # Asegurar que los valores de los tests anteriores son correctos
        df_seco = clasificar_regimen_hidro(pd.DataFrame({"aportes_pct": [50], "volumen_util_pct": [60]}))
        assert df_seco["regimen_hidro"].iloc[0] == "seco"


# ---------------------------------------------------------------------------
# ENSO
# ---------------------------------------------------------------------------

class TestClasificarEnso:
    def test_el_nino(self):
        assert clasificar_enso(1.2) == "el_nino"

    def test_la_nina(self):
        assert clasificar_enso(-0.8) == "la_nina"

    def test_neutro(self):
        assert clasificar_enso(0.3) == "neutro"
        assert clasificar_enso(-0.3) == "neutro"

    def test_umbral_exacto_nino(self):
        assert clasificar_enso(0.5) == "el_nino"

    def test_umbral_exacto_nina(self):
        assert clasificar_enso(-0.5) == "la_nina"


class TestAgregarFeaturesEnso:
    def _df_hidro(self):
        fechas = pd.date_range("2025-03-01", periods=90, freq="D").date
        return pd.DataFrame({"fecha": fechas, "volumen_util_pct": range(90)})

    def _df_oni(self):
        return pd.DataFrame({
            "fecha": [dt.date(2025, 1, 1), dt.date(2025, 2, 1), dt.date(2025, 3, 1)],
            "oni": [0.8, 0.6, 0.45],
        })

    def test_columnas_agregadas(self):
        df = agregar_features_enso(self._df_hidro(), self._df_oni(), rezago_meses=2)
        assert "oni_lag" in df.columns
        assert "enso_regime" in df.columns

    def test_regimen_valido(self):
        df = agregar_features_enso(self._df_hidro(), self._df_oni(), rezago_meses=2)
        regimenes_validos = {"el_nino", "la_nina", "neutro"}
        assert set(df["enso_regime"].unique()).issubset(regimenes_validos)

    def test_dummy_enso_valores(self):
        df = pd.DataFrame({"enso_regime": ["el_nino", "la_nina", "neutro"]})
        df = dummy_enso(df)
        assert df.loc[0, "enso_el_nino"] == 1
        assert df.loc[1, "enso_la_nina"] == 1
        assert df.loc[2, "enso_el_nino"] == 0


# ---------------------------------------------------------------------------
# Lags
# ---------------------------------------------------------------------------

class TestAgregarLagsDiarios:
    def _df(self):
        return pd.DataFrame({
            "fecha": pd.date_range("2025-01-01", periods=40, freq="D").date,
            "precio": range(40),
        })

    def test_columnas_lag_creadas(self):
        df = agregar_lags_diarios(self._df(), "precio", lags_dias=[1, 7])
        assert "precio_lag1d" in df.columns
        assert "precio_lag7d" in df.columns

    def test_primer_lag_nan(self):
        df = agregar_lags_diarios(self._df(), "precio", lags_dias=[1])
        assert pd.isna(df["precio_lag1d"].iloc[0])

    def test_lag1_coincide_con_valor_anterior(self):
        df = agregar_lags_diarios(self._df(), "precio", lags_dias=[1])
        assert df["precio_lag1d"].iloc[5] == df["precio"].iloc[4]

    def test_lag7_coincide(self):
        df = agregar_lags_diarios(self._df(), "precio", lags_dias=[7])
        assert df["precio_lag7d"].iloc[10] == df["precio"].iloc[3]


class TestAgregarRolling:
    def _df(self):
        return pd.DataFrame({
            "fecha": pd.date_range("2025-01-01", periods=30, freq="D").date,
            "precio": [float(i) for i in range(30)],
        })

    def test_columnas_rolling_creadas(self):
        df = agregar_rolling(self._df(), "precio", ventanas_dias=[7])
        assert "precio_roll7d_mean" in df.columns
        assert "precio_roll7d_std" in df.columns

    def test_rolling_mean_correcto(self):
        df = agregar_rolling(self._df(), "precio", ventanas_dias=[3])
        # En la posicion 5 (valor=5), rolling 3: mean([3,4,5])=4.0
        assert df["precio_roll3d_mean"].iloc[5] == pytest.approx(4.0)


class TestAgregarLagsHorarios:
    def _df(self):
        return pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="h"),
            "precio": range(100),
        })

    def test_lag24h_dia_anterior(self):
        df = agregar_lags_horarios(self._df(), "precio", lags_horas=[24])
        assert df["precio_lag24h"].iloc[24] == 0  # la primera hora del dia 1
