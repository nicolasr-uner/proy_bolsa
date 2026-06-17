"""Ejecuta el modelo de precio de bolsa y guarda los pronosticos.

Carga la feature matrix, ajusta el modelo y genera pronosticos a tres horizontes:
  - Corto plazo:   7 dias  (diario + horario)
  - Tactico:      30 dias  (3 escenarios: seco, promedio, humedo)
  - Largo plazo: 365 dias  (3 escenarios)

Guarda los resultados en outputs/forecasts/.

Uso:
    .venv/Scripts/python scripts/ejecutar_modelo_bolsa.py
    .venv/Scripts/python scripts/ejecutar_modelo_bolsa.py --horizonte-largo 720
    .venv/Scripts/python scripts/ejecutar_modelo_bolsa.py --solo-resumen
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from proybolsa.models.bolsa import PronosticadorBolsa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs" / "forecasts"


def _guardar(df: pd.DataFrame, nombre: str) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ruta = OUTPUTS / nombre
    df.to_parquet(ruta, index=False)
    logger.info("  Guardado: %s (%d filas)", nombre, len(df))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizonte-largo", type=int, default=365)
    parser.add_argument("--solo-resumen", action="store_true")
    parser.add_argument("--oni", type=float, default=0.0, help="ONI asumido para el pronostico largo")
    args = parser.parse_args()

    # --- Cargar datos y ajustar modelo ---
    logger.info("Cargando feature matrix...")
    df_diario = pd.read_parquet(PROCESSED / "bolsa_features_diario.parquet")

    logger.info("Ajustando modelo (%d dias de historico)...", len(df_diario))
    modelo = PronosticadorBolsa()
    modelo.fit(df_diario)

    resumen = modelo.resumen_modelo()
    logger.info("Modelo listo. Pesos: SARIMAX=%.2f  LGB=%.2f  AIC=%s",
                resumen["w_sarimax"], resumen["w_lgb"],
                resumen.get("aic_sarimax", "N/A"))
    logger.info("Top features: %s", list(resumen.get("top_features_lgb", {}).keys()))

    if args.solo_resumen:
        for k, v in resumen.items():
            print(f"  {k}: {v}")
        return

    # --- Pronostico corto plazo (7 dias, horario) ---
    logger.info("Pronosticando 7 dias (corto plazo, horario)...")
    fc7 = modelo.pronosticar(7, devolver_horario=True)
    _guardar(fc7, "bolsa_corto_horario.parquet")
    logger.info("  Rango diario: %.1f - %.1f COP/kWh",
                fc7["pred_diaria"].min(), fc7["pred_diaria"].max())

    # --- Pronostico tactico (30 dias, 3 escenarios) ---
    logger.info("Pronosticando 30 dias (tactico, 3 escenarios)...")
    esc30 = modelo.pronosticar(30, escenarios_multiples=True, devolver_horario=False, oni_asumido=args.oni)
    for nombre, fc in esc30.items():
        fc["escenario"] = nombre
    fc30 = pd.concat(esc30.values())
    _guardar(fc30, "bolsa_tactico_escenarios.parquet")
    for nombre, fc in esc30.items():
        logger.info("  %s: media=%.1f  p10=%.1f  p90=%.1f",
                    nombre,
                    fc["pred_diaria"].mean(),
                    fc["ci_lo90"].mean(),
                    fc["ci_hi90"].mean())

    # --- Pronostico largo plazo ---
    logger.info("Pronosticando %d dias (largo plazo, 3 escenarios)...", args.horizonte_largo)
    esc_l = modelo.pronosticar(
        args.horizonte_largo,
        escenarios_multiples=True,
        devolver_horario=False,
        oni_asumido=args.oni,
    )
    for nombre, fc in esc_l.items():
        fc["escenario"] = nombre
    fc_l = pd.concat(esc_l.values())
    _guardar(fc_l, f"bolsa_largo_{args.horizonte_largo}d_escenarios.parquet")
    for nombre, fc in esc_l.items():
        logger.info("  %s: media 12m=%.1f COP/kWh", nombre, fc["pred_diaria"].mean())

    logger.info("Pronosticos guardados en %s", OUTPUTS)


if __name__ == "__main__":
    main()
