"""Tests de la persistencia incremental de series crudas.

El test que más importa es `test_upsert_no_pierde_filas_viejas`: la implementación anterior
hacía `df.to_parquet(ruta)` con lo recién descargado, así que una descarga incremental habría
borrado la historia. Es la clase de bug que solo se nota meses después.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from proybolsa.ingest.series import SerieSpec, resolver
from proybolsa.ingest.store import (
    estado_series,
    rango_a_descargar,
    ultima_fecha,
    upsert_parquet,
)


@pytest.fixture
def spec_diaria() -> SerieSpec:
    return SerieSpec(
        nombre="prueba_diaria", ruta_rel="macro/prueba_diaria.parquet",
        col_tiempo="fecha", freq="D", ventana_revision_dias=5,
        tolerancia_frescura_dias=4, fetcher_ref=("proybolsa.ingest.macros", "fetch_trm"),
        inicio_default=dt.date(2023, 1, 1),
    )


def _df(fechas: list[str], valores: list[float], col: str = "fecha",
        como_date: bool = True) -> pd.DataFrame:
    f = pd.to_datetime(fechas)
    return pd.DataFrame({col: f.date if como_date else f, "valor": valores})


# ---------------------------------------------------------------------------
# rango_a_descargar
# ---------------------------------------------------------------------------

def test_sin_parquet_arranca_en_inicio_default(tmp_path, spec_diaria):
    r = rango_a_descargar(spec_diaria, tmp_path, dt.date(2026, 8, 11))
    assert r == (dt.date(2023, 1, 1), dt.date(2026, 8, 10))


def test_incremental_re_baja_la_ventana_de_revision(tmp_path, spec_diaria):
    upsert_parquet(_df(["2026-08-01", "2026-08-02"], [1.0, 2.0]), spec_diaria, tmp_path)
    desde, hasta = rango_a_descargar(spec_diaria, tmp_path, dt.date(2026, 8, 11))
    # ultima = 2026-08-02, ventana 5 dias -> arranca en 07-28, no en 08-03
    assert desde == dt.date(2026, 7, 28)
    assert hasta == dt.date(2026, 8, 10)


def test_resync_amplia_la_cola(tmp_path, spec_diaria):
    upsert_parquet(_df(["2026-08-02"], [2.0]), spec_diaria, tmp_path)
    desde, _ = rango_a_descargar(spec_diaria, tmp_path, dt.date(2026, 8, 11), resync_dias=30)
    assert desde == dt.date(2026, 6, 28)


def test_modo_completo_ignora_lo_existente(tmp_path, spec_diaria):
    upsert_parquet(_df(["2026-08-02"], [2.0]), spec_diaria, tmp_path)
    r = rango_a_descargar(spec_diaria, tmp_path, dt.date(2026, 8, 11), modo="completo")
    assert r == (dt.date(2023, 1, 1), dt.date(2026, 8, 10))


def test_nada_por_hacer_devuelve_none(tmp_path, spec_diaria):
    """Si la serie ya está más adelante que la cola, no hay rango que pedir."""
    upsert_parquet(_df(["2026-08-20"], [1.0]), spec_diaria, tmp_path)
    assert rango_a_descargar(spec_diaria, tmp_path, dt.date(2026, 8, 11)) is None


def test_reemplazo_total_siempre_pide_todo(tmp_path):
    spec = SerieSpec(
        nombre="upme", ruta_rel="xm/upme.parquet", col_tiempo="fecha", freq="M",
        ventana_revision_dias=0, tolerancia_frescura_dias=400,
        fetcher_ref=("proybolsa.ingest.xm", "fetch_demanda_upme"), reemplazo_total=True,
        inicio_default=dt.date(2023, 1, 1),
    )
    upsert_parquet(_df(["2026-08-01"], [1.0]), spec, tmp_path)
    r = rango_a_descargar(spec, tmp_path, dt.date(2026, 8, 11))
    assert r == (dt.date(2023, 1, 1), dt.date(2026, 8, 10))


# ---------------------------------------------------------------------------
# upsert_parquet
# ---------------------------------------------------------------------------

def test_upsert_no_pierde_filas_viejas(tmp_path, spec_diaria):
    """Regresión del bug de `to_parquet` con overwrite total."""
    upsert_parquet(_df(["2026-01-01", "2026-01-02", "2026-01-03"], [1.0, 2.0, 3.0]),
                   spec_diaria, tmp_path)
    res = upsert_parquet(_df(["2026-01-04"], [4.0]), spec_diaria, tmp_path)

    assert res.filas_antes == 3
    assert res.filas_despues == 4
    assert res.filas_nuevas == 1
    df = pd.read_parquet(tmp_path / spec_diaria.ruta_rel)
    assert sorted(df["valor"].tolist()) == [1.0, 2.0, 3.0, 4.0]


def test_el_dato_nuevo_gana_en_el_dedup(tmp_path, spec_diaria):
    upsert_parquet(_df(["2026-01-01", "2026-01-02"], [1.0, 2.0]), spec_diaria, tmp_path)
    res = upsert_parquet(_df(["2026-01-02"], [99.0]), spec_diaria, tmp_path)

    df = pd.read_parquet(tmp_path / spec_diaria.ruta_rel)
    assert len(df) == 2, "no debe duplicar la clave temporal"
    assert df.loc[pd.to_datetime(df["fecha"]) == "2026-01-02", "valor"].iloc[0] == 99.0
    assert res.filas_nuevas == 0
    assert res.filas_revisadas == 1


def test_reporta_el_detalle_de_las_revisiones(tmp_path, spec_diaria):
    upsert_parquet(_df(["2026-01-01"], [10.0]), spec_diaria, tmp_path)
    res = upsert_parquet(_df(["2026-01-01"], [10.5]), spec_diaria, tmp_path)

    assert res.filas_revisadas == 1
    assert not res.revisiones.empty
    assert res.revisiones["valor_ant"].iloc[0] == 10.0
    assert res.revisiones["valor_new"].iloc[0] == 10.5


def test_reescribir_el_mismo_dato_no_cuenta_como_revision(tmp_path, spec_diaria):
    """Idempotencia: re-descargar la cola sin cambios no debe reportar movimiento."""
    d = _df(["2026-01-01", "2026-01-02"], [1.0, 2.0])
    upsert_parquet(d, spec_diaria, tmp_path)
    res = upsert_parquet(d, spec_diaria, tmp_path)

    assert res.filas_nuevas == 0
    assert res.filas_revisadas == 0
    assert not res.avanzo


def test_salida_ordenada_y_sin_duplicados(tmp_path, spec_diaria):
    upsert_parquet(_df(["2026-01-03", "2026-01-01"], [3.0, 1.0]), spec_diaria, tmp_path)
    upsert_parquet(_df(["2026-01-02"], [2.0]), spec_diaria, tmp_path)

    df = pd.read_parquet(tmp_path / spec_diaria.ruta_rel)
    f = pd.to_datetime(df["fecha"])
    assert f.is_monotonic_increasing
    assert not f.duplicated().any()


def test_preserva_el_dtype_date_del_parquet(tmp_path, spec_diaria):
    """Las series diarias del repo guardan `fecha` como datetime.date, no datetime64.

    Cambiarlo silenciosamente rompería a los consumidores por un motivo cosmético.
    """
    upsert_parquet(_df(["2026-01-01"], [1.0], como_date=True), spec_diaria, tmp_path)
    upsert_parquet(_df(["2026-01-02"], [2.0], como_date=False), spec_diaria, tmp_path)

    df = pd.read_parquet(tmp_path / spec_diaria.ruta_rel)
    assert isinstance(df["fecha"].iloc[0], dt.date)
    assert not isinstance(df["fecha"].iloc[0], dt.datetime)


def test_preserva_datetime64_en_series_horarias(tmp_path):
    spec = SerieSpec(
        nombre="horaria", ruta_rel="xm/horaria.parquet", col_tiempo="timestamp", freq="H",
        ventana_revision_dias=15, tolerancia_frescura_dias=5,
        fetcher_ref=("proybolsa.ingest.xm", "fetch_precio_bolsa_horario"),
    )
    ts = pd.date_range("2026-01-01", periods=48, freq="h")
    upsert_parquet(pd.DataFrame({"timestamp": ts, "precio_bolsa": range(48)}), spec, tmp_path)

    df = pd.read_parquet(tmp_path / spec.ruta_rel)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert len(df) == 48


def test_no_deja_tmp_tras_escribir(tmp_path, spec_diaria):
    upsert_parquet(_df(["2026-01-01"], [1.0]), spec_diaria, tmp_path)
    assert not list((tmp_path / "macro").glob("*.tmp")), "quedó un archivo temporal"


def test_falta_la_columna_de_tiempo_es_error_explicito(tmp_path, spec_diaria):
    with pytest.raises(ValueError, match="columna de tiempo"):
        upsert_parquet(pd.DataFrame({"otra": [1]}), spec_diaria, tmp_path)


def test_ultima_fecha_sin_archivo_es_none(tmp_path, spec_diaria):
    assert ultima_fecha(spec_diaria, tmp_path) is None


# ---------------------------------------------------------------------------
# estado_series y registro
# ---------------------------------------------------------------------------

def test_estado_series_marca_atraso(tmp_path):
    """Es el gate que habría cazado la congelación de junio."""
    est = estado_series(tmp_path, dt.date(2026, 8, 11))
    assert len(est) == 11, "el registro debe cubrir las 11 series (el IPP va aparte)"
    assert not est["existe"].any()
    assert not est["frescura_ok"].any()
    assert set(est.columns) >= {"serie", "ultima_fecha", "dias_atraso", "frescura_ok"}


def test_resolver_grupos_y_nombres():
    assert len(resolver("xm")) == 7
    assert len(resolver("macro")) == 4
    assert [s.nombre for s in resolver("trm_diaria,brent_diario")] == ["trm_diaria", "brent_diario"]
    assert len(resolver(None)) == 11
    # sin duplicados aunque se pidan dos veces
    assert len(resolver("macro,trm_diaria")) == 4
    with pytest.raises(KeyError, match="desconocida"):
        resolver("no_existe")
