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
commitear los .pkl actualizados junto a los .meta.json.

Además del .pkl, se escribe `{nombre}.meta.json` con el mtime/tamaño/n_filas de los parquets
que alimentaron el ajuste. El dashboard compara esa huella contra los parquets en disco y
avisa si el .pkl quedó obsoleto: antes la decisión era solo `pkl.exists()`, así que un
parquet actualizado sin re-serializar servía un modelo viejo en silencio.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

P_BOLSA = PROCESSED / "bolsa_features_diario.parquet"
P_PERFIL = PROCESSED / "perfil_horario.parquet"
P_IPP = PROCESSED / "ipp_features_mensual.parquet"


def _huella(ruta: Path) -> dict:
    """Huella de un parquet de entrada, para detectar .pkl obsoletos."""
    st = ruta.stat()
    return {
        "archivo": ruta.name,
        "mtime": round(st.st_mtime, 3),
        "bytes": st.st_size,
        "n_filas": int(len(pd.read_parquet(ruta, columns=[]))),
    }


def _escribir_meta(nombre: str, fuentes: list[Path], extra: dict) -> Path:
    meta = {
        "modelo": nombre,
        "fecha_serializacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "fuentes": [_huella(p) for p in fuentes],
        **extra,
    }
    ruta = MODELS / f"{nombre}.meta.json"
    ruta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ruta


def _fit_bolsa():
    from proybolsa.models.bolsa.pronostico import PronosticadorBolsa
    df = pd.read_parquet(P_BOLSA)
    m = PronosticadorBolsa()
    m.fit(df, ruta_perfil=str(P_PERFIL))
    return m


def _fit_ipp():
    from proybolsa.models.ipp.modelo_ipp import PronosticadorIPP
    df = pd.read_parquet(P_IPP).dropna(subset=["ipp"])
    m = PronosticadorIPP()
    m.fit(df)
    return m


def _guardar(modelo, nombre: str) -> tuple[Path, int]:
    ruta = MODELS / f"{nombre}.pkl"
    previo = ruta.stat().st_size if ruta.exists() else 0
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    ahora = ruta.stat().st_size
    if previo:
        print(f"  {ruta.name}: {previo/1024/1024:.2f} MB -> {ahora/1024/1024:.2f} MB "
              f"({100 * (ahora / previo - 1):+.0f}%)")
    else:
        print(f"  {ruta.name}: {ahora/1024/1024:.2f} MB")
    return ruta, ahora


def _max_abs_diff(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> float:
    return max(float(np.abs(a[c].values - b[c].values).max()) for c in cols)


def main() -> int:
    print(f"Python {sys.version.split()[0]} — serializando modelos en {MODELS}")

    # --- Bolsa ---
    print("Ajustando PronosticadorBolsa...")
    bolsa = _fit_bolsa()
    p_bolsa, _ = _guardar(bolsa, "bolsa")
    resumen_b = bolsa.resumen_modelo()
    meta_b = _escribir_meta("bolsa", [P_BOLSA, P_PERFIL], {"resumen": resumen_b})

    # --- IPP ---
    print("Ajustando PronosticadorIPP...")
    ipp = _fit_ipp()
    p_ipp, _ = _guardar(ipp, "ipp")
    resumen_i = ipp.resumen_modelo()
    meta_i = _escribir_meta("ipp", [P_IPP], {"resumen": resumen_i})

    # --- Verificación de round-trip ---
    # Lo que importa no es solo que el .pkl cargue, sino que el modelo deserializado
    # pronostique EXACTAMENTE igual que el que quedó en memoria. Desde que el pickle guarda
    # parámetros en vez del SARIMAXResults, esta comparación es la red que detecta una
    # reconstrucción defectuosa antes de que llegue a la nube.
    print("Verificando round-trip (identidad del forecast tras recargar)...")
    cols_b = ["pred_diaria", "ci_lo90", "ci_hi90"]
    esperado_b = bolsa.pronosticar(30, devolver_horario=False)
    with open(p_bolsa, "rb") as f:
        b2 = pickle.load(f)
    obtenido_b = b2.pronosticar(30, devolver_horario=False)
    d_b = _max_abs_diff(esperado_b, obtenido_b, cols_b)
    assert d_b == 0.0, f"Bolsa: el forecast cambió tras recargar (diff max {d_b})"
    print(f"  Bolsa: forecast identico (diff max {d_b:.1e})")

    cols_i = ["pred", "ci_lo90", "ci_hi90"]
    esperado_i = ipp.pronosticar(12)
    with open(p_ipp, "rb") as f:
        i2 = pickle.load(f)
    obtenido_i = i2.pronosticar(12)
    d_i = _max_abs_diff(esperado_i, obtenido_i, cols_i)
    assert d_i == 0.0, f"IPP: el forecast cambió tras recargar (diff max {d_i})"
    print(f"  IPP: forecast identico (diff max {d_i:.1e})")

    # --- Verificación de comportamiento (dirección de los escenarios) ---
    escs = b2.pronosticar(30, escenarios_multiples=True, devolver_horario=False)
    ms = escs["seco"]["pred_diaria"].mean()
    mh = escs["humedo"]["pred_diaria"].mean()
    assert ms >= mh, "Bolsa: seco debe ser >= humedo tras recargar"
    print(f"  Bolsa escenarios OK: seco={ms:.1f} >= humedo={mh:.1f}")

    comp = b2.pronosticar(30, devolver_horario=False, devolver_componentes=True)
    faltan = {"pred_sarimax", "pred_lgb"} - set(comp.columns)
    assert not faltan, f"Bolsa: faltan componentes tras recargar: {faltan}"
    print(f"  Bolsa componentes OK: {sorted({'pred_sarimax', 'pred_lgb'})}")

    fut_alto = i2.construir_futuro_drivers(24, trm_var_anual=0.20, brent_var_anual=0.20)
    fut_bajo = i2.construir_futuro_drivers(24, trm_var_anual=-0.20, brent_var_anual=-0.20)
    a = i2.pronosticar(24, df_futuro=fut_alto)["pred"].iloc[-1]
    b = i2.pronosticar(24, df_futuro=fut_bajo)["pred"].iloc[-1]
    assert abs(a - b) > 5.0, "IPP: el escenario debe mover el pronóstico tras recargar"
    print(f"  IPP escenarios OK: alto={a:.2f}  bajo={b:.2f}  spread={a-b:+.2f}")

    print(f"OK — modelos serializados y verificados. Meta: {meta_b.name}, {meta_i.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
