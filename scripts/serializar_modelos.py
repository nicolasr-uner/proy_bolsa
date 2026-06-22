"""Pre-ajusta y serializa los modelos para Streamlit Community Cloud.

El dashboard, por defecto, ajusta los modelos en vivo (`.fit()`). El `.fit()` es
caro (optimización MLE de SARIMAX + Johansen/VECM); en la CPU compartida del tier
gratuito de Streamlit Cloud bloquea el hilo por minutos y dispara 503 en el
health-check. El `.pronosticar()` (forecast) en cambio es barato.

Solución: ajustar UNA vez aquí (desde el .venv, cuyas versiones calzan EXACTO con
los pins de requirements.txt que usa Cloud) y serializar. El dashboard carga el
.pkl al instante y solo hace forecast en vivo (escenarios interactivos).

Ejecutar desde el .venv (NO el Python del sistema 3.14):
    .venv\\Scripts\\python scripts/serializar_modelos.py

Re-ejecutar tras cada ciclo mensual (cuando cambian los parquets de features) y
commitear los .pkl actualizados.
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def _fit_bolsa():
    from proybolsa.models.bolsa.pronostico import PronosticadorBolsa
    df = pd.read_parquet(PROCESSED / "bolsa_features_diario.parquet")
    m = PronosticadorBolsa()
    m.fit(df, ruta_perfil=str(PROCESSED / "perfil_horario.parquet"))
    return m


def _fit_ipp():
    from proybolsa.models.ipp.modelo_ipp import PronosticadorIPP
    df = pd.read_parquet(PROCESSED / "ipp_features_mensual.parquet").dropna(subset=["ipp"])
    m = PronosticadorIPP()
    m.fit(df)
    return m


def main() -> int:
    print(f"Python {sys.version.split()[0]} — serializando modelos en {MODELS}")

    # --- Bolsa ---
    print("Ajustando PronosticadorBolsa...")
    bolsa = _fit_bolsa()
    p_bolsa = MODELS / "bolsa.pkl"
    with open(p_bolsa, "wb") as f:
        pickle.dump(bolsa, f)
    print(f"  guardado {p_bolsa.name} ({p_bolsa.stat().st_size/1024:.0f} KB)")

    # --- IPP ---
    print("Ajustando PronosticadorIPP...")
    ipp = _fit_ipp()
    p_ipp = MODELS / "ipp.pkl"
    with open(p_ipp, "wb") as f:
        pickle.dump(ipp, f)
    print(f"  guardado {p_ipp.name} ({p_ipp.stat().st_size/1024:.0f} KB)")

    # --- Verificación de round-trip: cargar en limpio y pronosticar ---
    print("Verificando round-trip (cargar + pronosticar)...")
    with open(p_bolsa, "rb") as f:
        b2 = pickle.load(f)
    escs = b2.pronosticar(30, escenarios_multiples=True, devolver_horario=False)
    ms = escs["seco"]["pred_diaria"].mean()
    mh = escs["humedo"]["pred_diaria"].mean()
    assert ms >= mh, "Bolsa: seco debe ser >= humedo tras recargar"
    print(f"  Bolsa OK: seco={ms:.1f} >= humedo={mh:.1f}")

    with open(p_ipp, "rb") as f:
        i2 = pickle.load(f)
    fut_alto = i2.construir_futuro_drivers(24, trm_var_anual=0.20, brent_var_anual=0.20)
    fut_bajo = i2.construir_futuro_drivers(24, trm_var_anual=-0.20, brent_var_anual=-0.20)
    a = i2.pronosticar(24, df_futuro=fut_alto)["pred"].iloc[-1]
    b = i2.pronosticar(24, df_futuro=fut_bajo)["pred"].iloc[-1]
    assert abs(a - b) > 5.0, "IPP: el escenario debe mover el pronóstico tras recargar"
    print(f"  IPP OK: alto={a:.2f}  bajo={b:.2f}  spread={a-b:+.2f}")

    print("OK — modelos serializados y verificados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
