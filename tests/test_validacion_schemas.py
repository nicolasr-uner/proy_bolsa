"""La validacion Pandera cableada rechaza datos invalidos (no es decoracion)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from proybolsa.validate.schemas import PRECIO_BOLSA_HORARIO, IPP_MENSUAL


def test_precio_negativo_es_rechazado():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
        "precio_bolsa": [100.0, -5.0, 200.0],  # negativo invalido
    })
    with pytest.raises(pa.errors.SchemaErrors):
        PRECIO_BOLSA_HORARIO.validate(df, lazy=True)


def test_ipp_acepta_fecha_datetime64():
    # regresion del fix de dtype: fecha datetime64 debe pasar
    df = pd.DataFrame({
        "fecha": pd.date_range("2020-01-01", periods=3, freq="MS"),
        "ipp": [100.0, 101.0, 102.0],
    })
    IPP_MENSUAL.validate(df, lazy=True)  # no debe lanzar


def test_precio_valido_pasa():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
        "precio_bolsa": [100.0, 150.0, 200.0],
    })
    PRECIO_BOLSA_HORARIO.validate(df, lazy=True)  # no debe lanzar
