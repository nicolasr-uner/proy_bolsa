"""Tests del calibrador empírico de intervalos del IPP.

Contexto: las bandas nominales del 90% cubrían entre 44% y 75% real según modelo y horizonte.
Estos tests fijan las propiedades que hacen seguro el reemplazo, no el número de cobertura
—que depende de los datos y de cuántos orígenes haya.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proybolsa.models.ipp.incertidumbre import CalibradorIntervalos


def _errores(n_origenes: int = 60, horizontes=(1, 3, 6, 12, 24), sesgo: float = 0.0,
             sigma: float = 0.01) -> pd.DataFrame:
    """Errores sintéticos con dispersión creciente en el horizonte."""
    rng = np.random.default_rng(5)
    filas = []
    for o in range(n_origenes):
        for h in horizontes:
            pred = 150.0
            e_log = rng.normal(sesgo * np.sqrt(h), sigma * np.sqrt(h))
            real = pred * np.exp(e_log)
            filas.append({
                "modelo": "m", "horizonte": h, "fecha_corte": pd.Timestamp("2020-01-01") +
                pd.DateOffset(months=o),
                "y_real": real, "y_pred": pred,
                "ci_lo90": pred * 0.98, "ci_hi90": pred * 1.02,
            })
    return pd.DataFrame(filas)


def test_calibra_y_queda_disponible():
    cal = CalibradorIntervalos.desde_errores(_errores(), "m")
    assert cal.disponible
    assert set(cal.q_lo) == {1, 3, 6, 12, 24}
    assert cal.n_origenes == 60


def test_la_banda_se_abre_con_el_horizonte():
    cal = CalibradorIntervalos.desde_errores(_errores(), "m")
    anchos = [cal.q_hi[h] - cal.q_lo[h] for h in (1, 3, 6, 12, 24)]
    assert all(b > a for a, b in zip(anchos, anchos[1:])), f"anchos no crecientes: {anchos}"


def test_ci_lo_menor_que_pred_menor_que_ci_hi():
    cal = CalibradorIntervalos.desde_errores(_errores(), "m")
    pred = np.full(24, 150.0)
    lo, hi = cal.aplicar(pred, np.arange(1, 25))
    assert (lo < pred).all() and (pred < hi).all()


def test_el_ancho_aplicado_es_monotono():
    """Un intervalo a 12 meses no puede ser más angosto que uno a 6."""
    cal = CalibradorIntervalos.desde_errores(_errores(), "m")
    lo, hi = cal.aplicar(np.full(24, 150.0), np.arange(1, 25))
    ancho = hi - lo
    assert np.all(np.diff(ancho) >= -1e-9)


def test_interpola_horizontes_no_calibrados():
    cal = CalibradorIntervalos.desde_errores(_errores(horizontes=(1, 12)), "m")
    q_lo_6, q_hi_6 = cal._cuantiles(6)
    assert cal.q_lo[12] <= q_lo_6 <= cal.q_lo[1]
    assert cal.q_hi[1] <= q_hi_6 <= cal.q_hi[12]


def test_extrapola_mas_alla_del_maximo_calibrado():
    """Fuera del rango, la ley de potencia debe seguir abriendo la banda."""
    cal = CalibradorIntervalos.desde_errores(_errores(horizontes=(1, 3, 6, 12)), "m")
    _, _ = cal._cuantiles(12)
    a24 = np.subtract(*reversed(cal._cuantiles(24)))
    a12 = np.subtract(*reversed(cal._cuantiles(12)))
    assert a24 > a12, "la banda a 24m debe ser mas ancha que a 12m"


def test_los_cuantiles_asimetricos_absorben_el_sesgo():
    """Con un modelo que subestima sistemáticamente, la banda se desplaza hacia arriba."""
    cal = CalibradorIntervalos.desde_errores(_errores(sesgo=0.02), "m")
    for h in (6, 12, 24):
        centro = (cal.q_hi[h] + cal.q_lo[h]) / 2
        assert centro > 0, f"h={h}: la banda no se desplazo pese al sesgo positivo"


def test_sin_datos_suficientes_no_queda_disponible():
    cal = CalibradorIntervalos.desde_errores(_errores(n_origenes=6), "m")
    assert not cal.disponible


def test_fallback_ensancha_el_ci_nominal():
    """Sin calibración, se usa el nominal escalado, nunca el nominal crudo."""
    cal = CalibradorIntervalos()
    cal.factor_vs_nominal = {1: 2.0}
    pred = np.array([100.0])
    lo, hi = cal.aplicar(pred, np.array([1]), ci_nominal=(np.array([99.0]), np.array([101.0])))
    ancho_nominal = 101.0 - 99.0
    assert (hi - lo)[0] > ancho_nominal, "el fallback debe ensanchar, no copiar"


def test_fallback_sin_nominal_no_explota():
    cal = CalibradorIntervalos()
    lo, hi = cal.aplicar(np.array([100.0, 100.0]), np.array([1, 12]))
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert (hi - lo)[1] > (hi - lo)[0]


def test_modelo_inexistente_devuelve_calibrador_vacio():
    cal = CalibradorIntervalos.desde_errores(_errores(), "no_existe")
    assert not cal.disponible
    assert cal.resumen().empty


def test_errores_vacios_no_rompen():
    assert not CalibradorIntervalos.desde_errores(pd.DataFrame(), "m").disponible


def test_resumen_reporta_cobertura_y_factor():
    cal = CalibradorIntervalos.desde_errores(_errores(), "m")
    r = cal.resumen()
    assert not r.empty
    assert {"horizonte", "cobertura_held_out_pct", "factor_vs_nominal"} <= set(r.columns)
    # El nominal sintetico es +-2% fijo: a 24 meses la banda real es mucho mas ancha.
    f24 = r.loc[r.horizonte == 24, "factor_vs_nominal"].iloc[0]
    assert f24 > 1.0


def test_correccion_de_muestra_finita_ensancha_la_banda():
    """La banda calibrada nunca es más angosta que el cuantil empírico crudo.

    Comparar anchos entre tamaños de muestra distintos no sirve: con n chico el sorteo es
    otro y domina el ruido. La propiedad que sí se puede fijar es que, sobre LOS MISMOS datos,
    la corrección de muestra finita produce una banda al menos tan ancha como pedir el 5/95
    directo. Sin ella las bandas salen sistemáticamente angostas, que es el defecto que este
    módulo existe para corregir.
    """
    err = _errores(n_origenes=60)
    cal = CalibradorIntervalos.desde_errores(err, "m", fraccion_validacion=0.3)

    # Reproducir la porción de calibración: el 70% inicial de orígenes.
    e = err.copy()
    e["e_log"] = np.log(e["y_real"]) - np.log(e["y_pred"])
    origenes = sorted(e["fecha_corte"].unique())
    corte = origenes[int(len(origenes) * 0.7)]
    cal_df = e[e["fecha_corte"] < corte]

    for h in (1, 6, 24):
        g = cal_df[cal_df["horizonte"] == h]["e_log"]
        ancho_crudo = float(np.quantile(g, 0.95) - np.quantile(g, 0.05))
        ancho_corregido = cal.q_hi[h] - cal.q_lo[h]
        assert ancho_corregido >= ancho_crudo - 1e-12, (
            f"h={h}: corregido {ancho_corregido:.5f} < crudo {ancho_crudo:.5f}"
        )


def test_cobertura_se_mide_fuera_de_la_muestra_de_calibracion():
    """La cobertura reportada no puede ser autovalidación."""
    err = _errores(n_origenes=60)
    cal = CalibradorIntervalos.desde_errores(err, "m", fraccion_validacion=0.3)
    assert cal.cobertura_medida, "no se midio cobertura"
    # Con datos bien comportados la cobertura held-out debe rondar el nominal, no ser 100%
    # por construccion en todos los horizontes.
    assert any(v < 100.0 for v in cal.cobertura_medida.values()), (
        "todas las coberturas dan 100%: se esta midiendo sobre los mismos datos de calibracion"
    )


@pytest.mark.parametrize("h", [1, 2, 7, 18, 36, 60])
def test_cualquier_horizonte_produce_banda_finita(h):
    cal = CalibradorIntervalos.desde_errores(_errores(), "m")
    lo, hi = cal.aplicar(np.array([150.0]), np.array([h]))
    assert np.isfinite(lo[0]) and np.isfinite(hi[0])
    assert lo[0] < 150.0 < hi[0]
