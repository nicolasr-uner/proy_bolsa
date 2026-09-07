from __future__ import annotations
import pandas as pd
from proybolsa.torneo.registro import registrar_panel, cargar_registro

def _panel(origen, y):
    return pd.DataFrame({
        "modelo": ["ipp_arima_drift", "ipp_bench_drift"],
        "horizonte": [1, 1],
        "fecha_objetivo": [pd.Timestamp("2026-02-01")] * 2,
        "y_pred": [y, y + 1.0],
        "ci_lo90": [float("nan")] * 2,
        "ci_hi90": [float("nan")] * 2,
    })

def test_registrar_es_idempotente_por_origen_modelo_horizonte(tmp_path):
    ruta = tmp_path / "registro_ipp.parquet"
    origen = pd.Timestamp("2026-01-01")
    registrar_panel(_panel(origen, 100.0), origen_fecha=origen,
                    fecha_run="20260101", ruta=ruta)
    registrar_panel(_panel(origen, 999.0), origen_fecha=origen,
                    fecha_run="20260101", ruta=ruta)
    reg = cargar_registro(ruta)
    assert len(reg) == 2, "el mismo origen no debe duplicar filas"
    fila = reg[reg["modelo"] == "ipp_arima_drift"].iloc[0]
    assert fila["y_pred"] == 100.0
    assert set(reg.columns) >= {
        "origen_fecha", "modelo", "horizonte", "fecha_objetivo",
        "y_pred", "ci_lo90", "ci_hi90", "fecha_run",
    }

def test_registrar_dos_origenes_distintos_acumula(tmp_path):
    ruta = tmp_path / "registro_ipp.parquet"
    registrar_panel(_panel(pd.Timestamp("2026-01-01"), 100.0),
                    origen_fecha=pd.Timestamp("2026-01-01"), fecha_run="20260101", ruta=ruta)
    registrar_panel(_panel(pd.Timestamp("2026-02-01"), 101.0),
                    origen_fecha=pd.Timestamp("2026-02-01"), fecha_run="20260201", ruta=ruta)
    assert len(cargar_registro(ruta)) == 4
