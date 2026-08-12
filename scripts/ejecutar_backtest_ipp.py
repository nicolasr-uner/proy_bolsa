"""Backtest rolling-origin mensual del IPP. Produce los pesos que usa producción.

POR QUÉ IMPORTA QUE ESTO SEA UN SCRIPT
--------------------------------------
`PronosticadorIPP.fit` lee `outputs/backtest/errores_ipp.parquet` para calibrar los pesos por
horizonte del ensemble (ver `_RUTA_BACKTEST_IPP` en `models/ipp/modelo_ipp.py`). Ese parquet se
versiona, de modo que Streamlit Cloud —que no puede correr un backtest— usa exactamente los
pesos calibrados en local. Si el parquet se genera a mano, la calibración de producción deja de
ser reproducible y nadie puede auditar de dónde salieron los pesos.

QUÉ MIDE
--------
Cada modelo se evalúa en multi-paso real con los drivers CONGELADOS en su último valor de
entrenamiento (`congelar_drivers_futuros`). Sin eso el backtest le entregaría al modelo los
valores futuros reales de TRM, Brent y los lags del propio IPP, y mediría una precisión que en
producción no existe.

Uso
---
    .venv\\Scripts\\python scripts/ejecutar_backtest_ipp.py
    .venv\\Scripts\\python scripts/ejecutar_backtest_ipp.py --rapido
    .venv\\Scripts\\python scripts/ejecutar_backtest_ipp.py --muestra ambas
    .venv\\Scripts\\python scripts/ejecutar_backtest_ipp.py --modelos ipp_arima_drift,ipp_bench_drift

Salidas (ambas versionadas):
    outputs/backtest/errores_ipp.parquet    una fila por (modelo, origen, horizonte)
    outputs/backtest/metricas_ipp.parquet   agregado por (modelo, horizonte)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from proybolsa.backtest.ipp_mensual import backtest_ipp, comparar_contra_benchmark  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("backtest_ipp")

PROCESSED = ROOT / "data" / "processed"
SALIDA = ROOT / "outputs" / "backtest"

# Modelos que van al parquet por defecto. Incluye los 4 componentes del ensemble por separado
# (que es de donde salen los pesos) y los benchmarks (que son la vara: ¿aporta algo sobre no
# hacer nada?).
MODELOS_DEFECTO = [
    "ipp_bench_rw", "ipp_bench_drift", "ipp_bench_drift12", "ipp_bench_theta",
    "ipp_arima_drift", "ipp_sarimax_actual", "ipp_vecm", "ipp_lgb_actual",
    "ipp_sarima_actual", "ipp_ensemble",
]

# Por debajo de esto, las métricas de un (modelo, horizonte) no son interpretables y se marcan.
# Un RMSE sobre 1 observación no es un RMSE.
MIN_VALIDAS = 10


def _cargar(muestra: str) -> pd.DataFrame:
    ruta = PROCESSED / "ipp_features_mensual.parquet"
    if not ruta.exists():
        logger.error("No existe %s. Corra scripts/construir_features.py", ruta)
        raise SystemExit(1)

    df = pd.read_parquet(ruta).sort_values("fecha").dropna(subset=["ipp"]).reset_index(drop=True)
    if muestra == "larga":
        logger.info("Muestra larga: %d meses (%s -> %s). Ojo: los modelos con drivers no son "
                    "estimables antes de 2015.", len(df), df.fecha.min().date(), df.fecha.max().date())
        return df

    con_drivers = df[df["brent_cop"].notna()].reset_index(drop=True)
    logger.info("Muestra con drivers: %d meses (%s -> %s)",
                len(con_drivers), con_drivers.fecha.min().date(), con_drivers.fecha.max().date())
    return con_drivers


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest rolling-origin mensual del IPP.")
    p.add_argument("--muestra", choices=["corta", "larga", "ambas"], default="corta",
                   help="corta (default) = solo meses con drivers macro (2015+).")
    p.add_argument("--modelos", default=None,
                   help="Lista separada por comas. Default: los 4 componentes + benchmarks.")
    p.add_argument("--horizontes", default="1,3,6,12,24")
    p.add_argument("--min-train", type=int, default=60, help="Meses mínimos de entrenamiento.")
    p.add_argument("--step", type=int, default=1, help="Meses que avanza cada origen.")
    p.add_argument("--modo-drivers", choices=["congelado", "perfecto"], default="congelado",
                   help="'perfecto' NO mide precisión alcanzable: sirve para cuantificar la "
                        "cota superior del valor de los drivers.")
    p.add_argument("--rapido", action="store_true",
                   help="step=3 y solo los componentes: para iterar, no para calibrar.")
    p.add_argument("--sin-guardar", action="store_true")
    args = p.parse_args()

    horizontes = tuple(int(h) for h in args.horizontes.split(","))
    modelos = args.modelos.split(",") if args.modelos else list(MODELOS_DEFECTO)
    step = 3 if args.rapido else args.step
    if args.rapido and not args.modelos:
        modelos = ["ipp_bench_drift", "ipp_arima_drift", "ipp_sarimax_actual"]

    muestras = ["corta", "larga"] if args.muestra == "ambas" else [args.muestra]

    errores, metricas = [], []
    t0 = time.perf_counter()
    for m in muestras:
        df = _cargar(m)
        err, met = backtest_ipp(df, modelos, horizontes=horizontes,
                                min_train=args.min_train, step=step,
                                modo_drivers=args.modo_drivers)
        if err.empty:
            logger.error("La muestra '%s' no produjo resultados", m)
            continue
        errores.append(err.assign(muestra=m))
        metricas.append(met.assign(muestra=m))

    if not errores:
        logger.error("Ningun backtest produjo resultados.")
        return 1

    err = pd.concat(errores, ignore_index=True)
    met = pd.concat(metricas, ignore_index=True)

    # Marca de interpretabilidad. No se borran las filas: se etiquetan, porque saber que un
    # modelo casi nunca pudo pronosticar es información valiosa por sí misma.
    met["n_suficiente"] = met["n_validas"] >= MIN_VALIDAS
    dudosas = met[~met["n_suficiente"]]
    if not dudosas.empty:
        logger.warning("%d combinaciones (modelo, horizonte) tienen menos de %d predicciones "
                       "validas: sus metricas NO son interpretables.", len(dudosas), MIN_VALIDAS)
        for m_, g in dudosas.groupby("modelo"):
            logger.warning("  %s: n_validas=%s de %s origenes",
                           m_, sorted(set(g["n_validas"])), sorted(set(g["n_predicciones"])))

    print()
    print("=== RMSE por horizonte (solo métricas interpretables) ===")
    vista = met[met["n_suficiente"]]
    if not vista.empty:
        print(vista.pivot_table(index="modelo", columns="horizonte", values="rmse")
              .round(2).to_string())

    print()
    print("=== Cobertura del CI 90% (nominal = 90) ===")
    if not vista.empty:
        print(vista.pivot_table(index="modelo", columns="horizonte",
                                values="cobertura_ci90_pct").round(1).to_string())

    print()
    print("=== Diebold-Mariano contra el mejor benchmark por horizonte ===")
    for clave in [c for c in modelos if not c.startswith("ipp_bench_")]:
        cmp = comparar_contra_benchmark(err, clave)
        if cmp.empty:
            continue
        print(f"\n{clave}:")
        print(cmp[["horizonte", "benchmark", "rmse_modelo", "rmse_benchmark",
                   "skill", "p_valor", "n"]].round(3).to_string(index=False))

    if args.sin_guardar:
        logger.info("--sin-guardar: no se escribio nada (%.0f s)", time.perf_counter() - t0)
        return 0

    SALIDA.mkdir(parents=True, exist_ok=True)
    err.to_parquet(SALIDA / "errores_ipp.parquet", index=False)
    met.to_parquet(SALIDA / "metricas_ipp.parquet", index=False)
    logger.info("Guardado en %s (%.0f s)", SALIDA.relative_to(ROOT), time.perf_counter() - t0)
    logger.info("Siguiente paso: re-serializar para que los pesos nuevos lleguen al .pkl")
    logger.info("  .venv\\Scripts\\python scripts/serializar_modelos.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
