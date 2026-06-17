"""Tests base del andamiaje (no requieren red)."""

import pandas as pd

import proybolsa
from proybolsa.ingest.xm import horas_anchas_a_largo


def test_version():
    assert proybolsa.__version__ == "0.1.0"


def _df_ancho_un_dia():
    fila = {"Id": "Sistema", "Values_code": "Sistema"}
    for h in range(1, 25):
        fila[f"Values_Hour{h:02d}"] = 100.0 + h
    fila["Date"] = pd.Timestamp("2026-05-01")
    return pd.DataFrame([fila])


def test_horas_anchas_a_largo_estructura():
    largo = horas_anchas_a_largo(_df_ancho_un_dia(), value_name="precio")
    assert len(largo) == 24
    assert largo["hora"].tolist() == list(range(24))
    assert largo["timestamp"].iloc[0] == pd.Timestamp("2026-05-01 00:00:00")
    assert largo["timestamp"].iloc[-1] == pd.Timestamp("2026-05-01 23:00:00")


def test_horas_anchas_a_largo_mapea_hora01_a_medianoche():
    largo = horas_anchas_a_largo(_df_ancho_un_dia(), value_name="precio")
    # Values_Hour01 = 101.0 debe quedar en la hora 0 (medianoche).
    assert largo.loc[largo["hora"] == 0, "precio"].iloc[0] == 101.0
    assert largo.loc[largo["hora"] == 23, "precio"].iloc[0] == 124.0
