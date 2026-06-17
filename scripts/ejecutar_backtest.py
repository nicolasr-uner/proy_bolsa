"""Backtesting rolling-origin del modelo de precio de bolsa.

Evalua el modelo con walk-forward validation sobre el historico disponible
y produce un reporte de métricas por horizonte.

El backtest usa refit_cada=4 (refitea cada 4 semanas) para ser manejable
en tiempo de cómputo; con refit_cada=1 es mas riguroso pero tarda ~30 min.

Uso:
    .venv/Scripts/python scripts/ejecutar_backtest.py
    .venv/Scripts/python scripts/ejecutar_backtest.py --riguroso     # refit_cada=1
    .venv/Scripts/python scripts/ejecutar_backtest.py --min-train 180
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from proybolsa.backtest.rolling_origin import RollingOriginBacktest, diebold_mariano
from proybolsa.models.bolsa import EnsembleNivel, PronosticadorBolsa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs" / "backtest"


# ---------------------------------------------------------------------------
# Wrappers fit/predict para RollingOriginBacktest
# ---------------------------------------------------------------------------

_COLS_PRECIO_LAG = [
    "precio_bolsa_mean_lag1d", "precio_bolsa_mean_lag7d", "precio_bolsa_mean_lag30d",
    "precio_bolsa_mean_roll7d_mean", "precio_bolsa_mean_roll30d_mean",
    "precio_bolsa_mean_roll7d_std",  "precio_bolsa_mean_roll30d_std",
]


def _fix_lags_precio(df_test: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza los lags de precio en df_test con los valores del ultimo dia de train.

    En produccion no conocemos los precios futuros; propagamos el ultimo estado conocido
    (random-walk assumption para los lags). Esto evita el data leak precio futuro -> feature.
    """
    df = df_test.copy()
    for col in _COLS_PRECIO_LAG:
        if col in df.columns and col in df_train.columns:
            df[col] = df_train[col].iloc[-1]
    return df


def _fit_ensemble(df_train: pd.DataFrame) -> dict:
    modelo = EnsembleNivel()
    modelo.fit(df_train, df_val=None)
    return {"modelo": modelo, "df_train": df_train}


def _predict_ensemble(estado: dict, df_test: pd.DataFrame, h: int) -> pd.DataFrame:
    modelo = estado["modelo"]
    df_train = estado["df_train"]
    # Usar features conocidos en el origen (sin data leak de precios futuros)
    df_input = _fix_lags_precio(df_test, df_train)
    return modelo.predict(df_input, horizon=h)


def _fit_naive(df_train: pd.DataFrame) -> dict:
    """Benchmark: ultimo valor observado (random walk)."""
    return {
        "ultimo": df_train["precio_bolsa_mean"].iloc[-1],
        "media_30d": df_train["precio_bolsa_mean"].iloc[-30:].mean(),
    }


def _predict_naive(modelo: dict, df_test: pd.DataFrame, h: int) -> pd.DataFrame:
    pred = modelo["ultimo"]
    std_est = df_test["precio_bolsa_mean_roll30d_std"].iloc[0] if "precio_bolsa_mean_roll30d_std" in df_test.columns else 60.0
    return pd.DataFrame({
        "pred": [pred] * h,
        "ci_lo90": [pred - 1.645 * std_est] * h,
        "ci_hi90": [pred + 1.645 * std_est] * h,
    })


def _fit_media_movil(df_train: pd.DataFrame) -> dict:
    """Benchmark: media movil 30 dias."""
    return {"pred": df_train["precio_bolsa_mean"].iloc[-30:].mean()}


def _predict_media_movil(modelo: dict, df_test: pd.DataFrame, h: int) -> pd.DataFrame:
    return pd.DataFrame({"pred": [modelo["pred"]] * h})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train", type=int, default=365, help="Dias minimos de entrenamiento")
    parser.add_argument("--step", type=int, default=14, help="Paso entre origenes (dias)")
    parser.add_argument("--horizonte-max", type=int, default=30)
    parser.add_argument("--riguroso", action="store_true", help="refit_cada=1 (lento pero correcto)")
    args = parser.parse_args()

    refit_cada = 1 if args.riguroso else 4
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando feature matrix...")
    df = pd.read_parquet(PROCESSED / "bolsa_features_diario.parquet")
    df = df.dropna(subset=["precio_bolsa_mean"]).reset_index(drop=True)
    logger.info("  %d dias disponibles (%.1f años)", len(df), len(df) / 365)

    bt = RollingOriginBacktest(
        horizonte_max=args.horizonte_max,
        step_dias=args.step,
        min_train_dias=args.min_train,
        horizontes_evaluar=[1, 7, 14, 30],
        refit_cada=refit_cada,
    )

    # --- Modelo ensemble ---
    logger.info("Evaluando ensemble (refit_cada=%d)...", refit_cada)
    err_ensemble, met_ensemble = bt.evaluar(
        df, "fecha", "precio_bolsa_mean",
        fn_fit=_fit_ensemble,
        fn_predict=_predict_ensemble,
    )
    met_ensemble["modelo"] = "ensemble"

    # --- Benchmark naive ---
    logger.info("Evaluando benchmark naive (ultimo valor)...")
    err_naive, met_naive = bt.evaluar(
        df, "fecha", "precio_bolsa_mean",
        fn_fit=_fit_naive,
        fn_predict=_predict_naive,
        verbose=False,
    )
    met_naive["modelo"] = "naive"

    # --- Benchmark media movil ---
    logger.info("Evaluando benchmark media movil 30d...")
    err_mm, met_mm = bt.evaluar(
        df, "fecha", "precio_bolsa_mean",
        fn_fit=_fit_media_movil,
        fn_predict=_predict_media_movil,
        verbose=False,
    )
    met_mm["modelo"] = "media_movil_30d"

    # --- Consolidar metricas ---
    metricas = pd.concat([met_ensemble, met_naive, met_mm], ignore_index=True)
    metricas = metricas.sort_values(["horizonte", "modelo"]).reset_index(drop=True)

    print("\n=== METRICAS POR HORIZONTE ===")
    for h in sorted(metricas["horizonte"].unique()):
        print(f"\n  Horizonte {h} dias:")
        sub = metricas[metricas["horizonte"] == h][
            ["modelo", "n_predicciones", "rmse", "mae", "mape_pct", "sesgo", "cobertura_ci90_pct"]
        ]
        print(sub.to_string(index=False))

    # --- Diebold-Mariano: ensemble vs naive ---
    print("\n=== TEST DIEBOLD-MARIANO (ensemble vs naive) ===")
    for h in [1, 7, 14, 30]:
        sub_e = err_ensemble[err_ensemble["horizonte"] == h].dropna(subset=["error"])
        sub_n = err_naive[err_naive["horizonte"] == h].dropna(subset=["error"])
        min_n = min(len(sub_e), len(sub_n))
        if min_n < 5:
            continue
        dm = diebold_mariano(
            (sub_e["error"] ** 2).iloc[:min_n].reset_index(drop=True),
            (sub_n["error"] ** 2).iloc[:min_n].reset_index(drop=True),
            h=h,
            alternativa="a_mejor",
        )
        print(f"  h={h:2d}: DM={dm['estadistico_dm']:+.2f}  p={dm['p_valor']:.4f}  -> {dm['conclusion']}")

    # --- Guardar ---
    err_ensemble["modelo"] = "ensemble"
    err_naive["modelo"] = "naive"
    err_mm["modelo"] = "media_movil_30d"
    errores_todo = pd.concat([err_ensemble, err_naive, err_mm], ignore_index=True)
    errores_todo.to_parquet(OUTPUTS / "errores_detalle.parquet", index=False)
    metricas.to_parquet(OUTPUTS / "metricas_resumen.parquet", index=False)
    logger.info("Resultados guardados en %s", OUTPUTS)


if __name__ == "__main__":
    main()
