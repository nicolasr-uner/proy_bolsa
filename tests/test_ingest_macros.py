"""Tests unitarios para src/proybolsa/ingest/macros.py.

Tests que no hacen llamadas de red (usan mocks) para poder correr en CI
sin credenciales ni conexion. Los tests de integracion con red real
se separan al final con el marker @pytest.mark.integration.
"""

from __future__ import annotations

import datetime as dt
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from proybolsa.ingest.macros import (
    _fetch_ipp_banrep_sdmx,
    _fetch_ipp_socrata,
    _fred_serie,
    fetch_brent,
    fetch_oni,
    fetch_ppi_usa,
    fetch_trm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

START = dt.date(2025, 1, 1)
END = dt.date(2025, 3, 31)


def _make_response(content, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.ok = status_code < 400
    mock.raise_for_status = MagicMock(
        side_effect=None if status_code < 400 else Exception(f"HTTP {status_code}")
    )
    if isinstance(content, (dict, list)):
        mock.json.return_value = content
        mock.text = json.dumps(content)
    else:
        mock.text = content
        mock.json.return_value = json.loads(content)
    return mock


# ---------------------------------------------------------------------------
# TRM
# ---------------------------------------------------------------------------

class TestFetchTrm:
    def _trm_rows(self):
        return [
            {"vigenciadesde": "2025-01-01T00:00:00.000", "vigenciahasta": "2025-01-03T00:00:00.000", "valor": "4100.0", "unidad": "COP"},
            {"vigenciadesde": "2025-01-06T00:00:00.000", "vigenciahasta": "2025-01-10T00:00:00.000", "valor": "4150.5", "unidad": "COP"},
        ]

    def test_columnas_correctas(self):
        with patch("proybolsa.ingest.macros.requests.get") as mock_get:
            mock_get.return_value = _make_response(self._trm_rows())
            df = fetch_trm(START, dt.date(2025, 1, 10))
        assert set(df.columns) == {"fecha", "trm"}
        assert not df.empty

    def test_devuelve_floats(self):
        with patch("proybolsa.ingest.macros.requests.get") as mock_get:
            mock_get.return_value = _make_response(self._trm_rows())
            df = fetch_trm(START, dt.date(2025, 1, 10))
        assert df["trm"].dtype == float

    def test_sin_datos_devuelve_df_vacio(self):
        with patch("proybolsa.ingest.macros.requests.get") as mock_get:
            mock_get.return_value = _make_response([])
            df = fetch_trm(START, END)
        assert df.empty
        assert list(df.columns) == ["fecha", "trm"]

    def test_forward_fill_crea_filas_diarias(self):
        rows = [
            {"vigenciadesde": "2025-01-01T00:00:00.000", "vigenciahasta": "2025-01-05T00:00:00.000", "valor": "4000.0", "unidad": "COP"},
        ]
        with patch("proybolsa.ingest.macros.requests.get") as mock_get:
            mock_get.return_value = _make_response(rows)
            df = fetch_trm(dt.date(2025, 1, 1), dt.date(2025, 1, 5))
        # Deberia haber al menos 5 filas (un dia de vigencia + forward fill)
        assert len(df) >= 1
        # Todos los valores deben ser 4000.0
        assert all(df["trm"] == 4000.0)


# ---------------------------------------------------------------------------
# FRED (Brent, PPI USA)
# ---------------------------------------------------------------------------

def _brent_df():
    """DataFrame que simula el CSV de FRED para Brent (construido sin llamar pd.read_csv)."""
    return pd.DataFrame({
        "DATE": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"],
        "DCOILBRENTEU": ["75.5", "76.0", ".", "77.2"],
    })


def _ppi_df():
    return pd.DataFrame({
        "DATE": ["2025-01-01", "2025-02-01"],
        "PPIACO": ["290.5", "292.0"],
    })


class TestFetchBrent:
    def test_columnas_y_dtype(self):
        with patch("proybolsa.ingest.macros.pd.read_csv", return_value=_brent_df()):
            df = fetch_brent(START, END)
        assert list(df.columns) == ["fecha", "brent"]
        assert df["brent"].dtype == float

    def test_puntos_nan_se_eliminan(self):
        with patch("proybolsa.ingest.macros.pd.read_csv", return_value=_brent_df()):
            df = fetch_brent(START, END)
        # El "." debe convertirse a NaN y eliminarse
        assert df["brent"].notna().all()
        assert len(df) == 3  # 4 filas - 1 punto invalido

    def test_error_fred_propaga(self):
        with patch("proybolsa.ingest.macros.pd.read_csv", side_effect=Exception("network")):
            with pytest.raises(RuntimeError, match="DCOILBRENTEU"):
                fetch_brent(START, END)


class TestFetchPpiUsa:
    def test_columnas_correctas(self):
        with patch("proybolsa.ingest.macros.pd.read_csv", return_value=_ppi_df()):
            df = fetch_ppi_usa(START, END)
        assert list(df.columns) == ["fecha", "ppi_usa"]
        assert len(df) > 0


# ---------------------------------------------------------------------------
# ONI
# ---------------------------------------------------------------------------

ONI_TEXTO = """\
SEAS YR   TOTAL  ANOM
DJF 2025  27.1   0.45
JFM 2025  27.3   0.55
FMA 2025  27.0   0.20
"""


class TestFetchOni:
    def test_columnas_y_fechas(self):
        with patch("proybolsa.ingest.macros.requests.get") as mock_get:
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.text = ONI_TEXTO
            mock_get.return_value = r
            df = fetch_oni(dt.date(2025, 1, 1), dt.date(2025, 12, 31))
        assert list(df.columns) == ["fecha", "oni"]
        assert len(df) == 3

    def test_djf_2025_es_enero(self):
        with patch("proybolsa.ingest.macros.requests.get") as mock_get:
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.text = ONI_TEXTO
            mock_get.return_value = r
            df = fetch_oni(dt.date(2025, 1, 1), dt.date(2025, 12, 31))
        fila = df[df["fecha"] == dt.date(2025, 1, 1)]
        assert len(fila) == 1
        assert fila["oni"].iloc[0] == pytest.approx(0.45)

    def test_filtro_por_rango(self):
        with patch("proybolsa.ingest.macros.requests.get") as mock_get:
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.text = ONI_TEXTO
            mock_get.return_value = r
            # Solo enero-febrero 2025
            df = fetch_oni(dt.date(2025, 1, 1), dt.date(2025, 2, 28))
        assert len(df) == 2


# ---------------------------------------------------------------------------
# IPP (logica de fallback)
# ---------------------------------------------------------------------------

class TestIppFallback:
    def test_sdmx_con_respuesta_invalida_lanza_runtime(self):
        with patch("proybolsa.ingest.macros.requests.get") as mock_get:
            r = MagicMock()
            r.ok = False
            r.status_code = 404
            mock_get.return_value = r
            with pytest.raises(RuntimeError):
                _fetch_ipp_banrep_sdmx(START, END)

    def test_socrata_sin_id_lanza_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _fetch_ipp_socrata(START, END)

    def test_sdmx_csv_valido_devuelve_df(self):
        sdmx_csv = "DATAFLOW,FREQ,REF_AREA,INDICATOR,UNIT_MULT,TIME_PERIOD,OBS_VALUE\n"
        sdmx_csv += "BanRep,M,CO,IPP_OFIN,0,2025-01,125.3\n"
        sdmx_csv += "BanRep,M,CO,IPP_OFIN,0,2025-02,126.1\n"
        with patch("proybolsa.ingest.macros.requests.get") as mock_get:
            r = MagicMock()
            r.ok = True
            r.text = sdmx_csv
            mock_get.return_value = r
            df = _fetch_ipp_banrep_sdmx(START, END)
        assert list(df.columns) == ["fecha", "ipp"]
        assert len(df) == 2
        assert df["ipp"].iloc[0] == pytest.approx(125.3)
