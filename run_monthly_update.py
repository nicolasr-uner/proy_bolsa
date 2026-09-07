"""Loop mensual de actualizacion y recalibracion de modelos.

Ejecutar una vez al mes, idealmente el primer dia habil del mes siguiente.

Flujo:
  1. Descarga datos nuevos de XM y macros
  2. Reconstruye feature matrices (bolsa + IPP)
  3. Carga el pronostico del mes anterior (si existe) y compara contra real
  4. Actualiza pesos del ensemble con la correccion de sesgo reciente
  5. Reajusta ambos modelos con el historico completo
  6. Genera pronosticos nuevos para los proximos 7d / 30d / 365d
  7. Registra todo en outputs/runs/{YYYY-MM}/

Uso:
    .venv/Scripts/python run_monthly_update.py
    .venv/Scripts/python run_monthly_update.py --solo-bolsa
    .venv/Scripts/python run_monthly_update.py --fecha-corte 2026-06-01
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

_PYTHON = sys.executable  # usa el python del venv activo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], desc: str) -> bool:
    logger.info("%s ...", desc)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("  FALLO: %s", res.stderr[-500:] if res.stderr else "sin stderr")
        return False
    logger.info("  OK")
    return True


def _cargar_ultimo_pronostico(tipo: str, run_dir: Path) -> pd.DataFrame | None:
    """Carga el pronostico del ciclo anterior para comparar con el real."""
    candidatos = sorted(OUTPUTS.glob(f"runs/*/pronostico_{tipo}_*.parquet"))
    if not candidatos:
        return None
    return pd.read_parquet(candidatos[-1])


def _calcular_error_ciclo(
    df_real: pd.DataFrame,
    df_pred: pd.DataFrame,
    col_fecha: str,
    col_real: str,
    col_pred: str,
) -> dict:
    """Compara predicción del ciclo anterior vs real del período recién observado."""
    merged = df_real.merge(df_pred, on=col_fecha, how="inner", suffixes=("_real", "_pred"))
    if merged.empty:
        return {}
    y_real = merged[col_real].values
    y_pred = merged[col_pred].values
    return {
        "n_obs": len(merged),
        "mae": float(np.mean(np.abs(y_real - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_real - y_pred) ** 2))),
        "sesgo": float(np.mean(y_pred - y_real)),
        "mape_pct": float(np.mean(np.abs((y_real - y_pred) / y_real.clip(1))) * 100),
    }


def _sesgo_ventana(errores_hist: list[dict], n_ultimos: int = 4) -> dict:
    """Calcula el sesgo medio de los últimos N ciclos por horizonte."""
    if not errores_hist:
        return {}
    recientes = errores_hist[-n_ultimos:]
    # Agrupa por horizonte y promedia el sesgo
    sesgos = {}
    for registro in recientes:
        h = registro.get("horizonte")
        s = registro.get("sesgo")
        if h is not None and s is not None:
            sesgos.setdefault(h, []).append(s)
    return {h: float(np.mean(vals)) for h, vals in sesgos.items()}


# ---------------------------------------------------------------------------
# Evaluacion del ciclo anterior
# ---------------------------------------------------------------------------

def _evaluar_ciclo_anterior(
    df_diario: pd.DataFrame,
    run_dir: Path,
) -> list[dict]:
    """Compara los pronosticos diarios guardados del ciclo anterior contra el real."""
    errores = []
    # Buscar archivos diarios de pronostico (guardados por ciclo_bolsa como *_diario_*)
    for nombre_patron, horizonte_nominal in [("corto_diario", 7), ("tactico", 30)]:
        fc_paths = sorted(OUTPUTS.glob(f"runs/*/pronostico_bolsa_{nombre_patron}_*.parquet"))
        if not fc_paths:
            continue
        df_fc = pd.read_parquet(fc_paths[-1])
        if "fecha" not in df_fc.columns or "pred_diaria" not in df_fc.columns:
            continue

        df_real_sub = df_diario[["fecha", "precio_bolsa_mean"]].copy()
        df_real_sub["fecha"] = pd.to_datetime(df_real_sub["fecha"]).dt.date
        df_fc["fecha"] = pd.to_datetime(df_fc["fecha"]).dt.date

        err = _calcular_error_ciclo(df_real_sub, df_fc, "fecha", "precio_bolsa_mean", "pred_diaria")
        if err:
            err["horizonte"] = horizonte_nominal
            err["nombre_fc"] = nombre_patron
            errores.append(err)
            logger.info(
                "  Error ciclo anterior h=%d: MAE=%.1f RMSE=%.1f Sesgo=%+.1f",
                horizonte_nominal, err["mae"], err["rmse"], err["sesgo"],
            )
    return errores


# ---------------------------------------------------------------------------
# Ciclo bolsa
# ---------------------------------------------------------------------------

def ciclo_bolsa(run_dir: Path, fecha_corte: str | None = None) -> None:
    from proybolsa.models.bolsa import PronosticadorBolsa

    logger.info("=== CICLO BOLSA ===")

    # 1. Cargar feature matrix
    df_diario = pd.read_parquet(PROCESSED / "bolsa_features_diario.parquet")
    if fecha_corte:
        df_diario = df_diario[df_diario["fecha"] <= pd.to_datetime(fecha_corte).date()]
    logger.info("  Historico: %d dias", len(df_diario))

    # 2. Evaluar ciclo anterior
    errores_ciclo = _evaluar_ciclo_anterior(df_diario, run_dir)

    # 3. Ajustar modelo
    logger.info("  Ajustando modelo...")
    modelo = PronosticadorBolsa()
    modelo.fit(df_diario)

    # 4. Actualizar sesgo si hay errores del ciclo anterior
    sesgo_reciente = _sesgo_ventana(errores_ciclo)
    if sesgo_reciente:
        modelo.modelo.actualizar_sesgo(sesgo_reciente)
        logger.info("  Sesgo actualizado: %s", sesgo_reciente)

    # 5. Generar pronosticos
    logger.info("  Generando pronosticos...")
    # Diario (para tracking de error mensual): devolver_horario=False
    fc7_diario = modelo.pronosticar(7, devolver_horario=False)
    # Horario: para consumo operativo
    fc7_horario = modelo.pronosticar(7, devolver_horario=True)
    fc30 = modelo.pronosticar(30, escenarios_multiples=True, devolver_horario=False)
    fc365 = modelo.pronosticar(365, escenarios_multiples=True, devolver_horario=False)

    # 6. Guardar
    run_dir.mkdir(parents=True, exist_ok=True)
    hoy = pd.Timestamp.today().strftime("%Y%m%d")

    # Diario (para evaluacion de error el proximo ciclo)
    fc7_diario.to_parquet(run_dir / f"pronostico_bolsa_corto_diario_{hoy}.parquet", index=False)
    # Horario (para uso operativo)
    fc7_horario.to_parquet(run_dir / f"pronostico_bolsa_corto_horario_{hoy}.parquet", index=False)

    for esc, df_esc in fc30.items():
        df_esc["escenario"] = esc
    pd.concat(fc30.values()).to_parquet(run_dir / f"pronostico_bolsa_tactico_{hoy}.parquet", index=False)
    for esc, df_esc in fc365.items():
        df_esc["escenario"] = esc
    pd.concat(fc365.values()).to_parquet(run_dir / f"pronostico_bolsa_largo_{hoy}.parquet", index=False)

    # 7. Guardar resumen del ciclo
    resumen = modelo.resumen_modelo()
    resumen["errores_ciclo_anterior"] = errores_ciclo
    resumen["sesgo_aplicado"] = sesgo_reciente
    resumen["fecha_run"] = hoy
    with open(run_dir / "resumen_bolsa.json", "w") as f:
        json.dump(resumen, f, indent=2, default=str)

    logger.info(
        "  Bolsa: corto=%.0f COP/kWh  seco_30d=%.0f  promedio_30d=%.0f  humedo_30d=%.0f",
        fc7_diario["pred_diaria"].mean(),
        fc30["seco"]["pred_diaria"].mean(),
        fc30["promedio"]["pred_diaria"].mean(),
        fc30["humedo"]["pred_diaria"].mean(),
    )


# ---------------------------------------------------------------------------
# Ciclo IPP
# ---------------------------------------------------------------------------

def ciclo_ipp(run_dir: Path, fecha_corte: str | None = None) -> None:
    from proybolsa.models.ipp import PronosticadorIPP

    logger.info("=== CICLO IPP ===")

    ipp_path = PROCESSED / "ipp_features_mensual.parquet"
    if not ipp_path.exists():
        logger.warning("  ipp_features_mensual.parquet no encontrado. Saltando ciclo IPP.")
        return

    df_ipp = pd.read_parquet(ipp_path)
    if "ipp" not in df_ipp.columns:
        logger.warning("  Sin columna 'ipp'. Cargar datos DANE y volver a ejecutar.")
        return

    if fecha_corte:
        df_ipp = df_ipp[df_ipp["fecha"] <= pd.to_datetime(fecha_corte)]
    logger.info("  Historico IPP: %d meses", len(df_ipp))

    logger.info("  Ajustando modelo IPP...")
    modelo = PronosticadorIPP()
    modelo.fit(df_ipp)

    fc12 = modelo.pronosticar(12)
    fc24 = modelo.pronosticar(24)

    run_dir.mkdir(parents=True, exist_ok=True)
    hoy = pd.Timestamp.today().strftime("%Y%m%d")
    fc12.to_parquet(run_dir / f"pronostico_ipp_12m_{hoy}.parquet", index=False)
    fc24.to_parquet(run_dir / f"pronostico_ipp_24m_{hoy}.parquet", index=False)

    resumen = modelo.resumen_modelo()
    resumen["fecha_run"] = hoy
    with open(run_dir / "resumen_ipp.json", "w") as f:
        json.dump(resumen, f, indent=2, default=str)

    logger.info(
        "  IPP: media 12m=%.2f  CI p10=%.2f  CI p90=%.2f",
        fc12["pred"].mean(),
        fc12["ci_lo90"].mean(),
        fc12["ci_hi90"].mean(),
    )

    # Torneo de modelos: puntear lo vencido y registrar el panel de este origen.
    try:
        from proybolsa.torneo.ciclo import correr_ciclo_torneo
        from proybolsa.models.ipp.modelo_ipp import (
            _RUTA_BACKTEST_IPP,
            _cargar_errores_backtest,
        )

        correr_ciclo_torneo(
            df_ipp,
            dir_salida=OUTPUTS / "torneo",
            fecha_run=hoy,
            errores_backtest=_cargar_errores_backtest(_RUTA_BACKTEST_IPP),
        )
        logger.info("  Torneo IPP actualizado en outputs/torneo/")
    except Exception:
        logger.exception("  El torneo IPP fallo; el ciclo principal continua")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Loop mensual de actualizacion de modelos.")
    parser.add_argument("--solo-bolsa", action="store_true")
    parser.add_argument("--solo-ipp", action="store_true")
    parser.add_argument("--fecha-corte", default=None, help="YYYY-MM-DD (para simulacion historica)")
    parser.add_argument("--sin-descarga", action="store_true", help="Saltar descarga de datos nuevos")
    parser.add_argument(
        "--permitir-datos-viejos", action="store_true",
        help="No abortar si la descarga falla o deja series fuera de tolerancia de frescura. "
             "Necesario para la simulacion historica con --fecha-corte.",
    )
    parser.add_argument("--resync-dias", type=int, default=90,
                        help="Cola extra a re-descargar (default 90, para recuperar atrasos).")
    args = parser.parse_args()

    periodo = pd.Timestamp.today().strftime("%Y-%m")
    run_dir = OUTPUTS / "runs" / periodo
    logger.info("Ciclo mensual: %s -> %s", periodo, run_dir)

    # Paso 1: descargar datos nuevos
    #
    # Este paso era un no-op. Se invocaba `descarga_historico.py` SIN `--forzar`, y su
    # `_ya_existe()` hacía skip binario cuando el parquet existía. Como los 12 parquets
    # existían, no se descargaba nada y el paso registraba "OK": así los datos quedaron
    # congelados dos meses mientras el ciclo se declaraba exitoso cada vez.
    #
    # Ahora la descarga es incremental (con la cola de revisión de cada serie) y un fallo
    # o un atraso ABORTA el ciclo, en vez de degradarse a un warning. Pronosticar sobre
    # datos rancios es peor que no pronosticar: el resultado parece válido.
    if not args.sin_descarga:
        cmd = [_PYTHON, "scripts/descarga_historico.py",
               "--modo", "incremental", "--resync-dias", str(args.resync_dias)]
        if not args.permitir_datos_viejos:
            cmd.append("--fail-si-atrasado")
        ok = _run(cmd, "Descargando datos nuevos")
        if not ok and not args.permitir_datos_viejos:
            logger.error(
                "La descarga falló o dejó series atrasadas. Se aborta el ciclo: un pronóstico "
                "sobre datos viejos es indistinguible de uno bueno. "
                "Use --permitir-datos-viejos si es intencional."
            )
            return

        ok_ipp = _run([_PYTHON, "scripts/actualizar_ipp.py"], "Actualizando IPP (anexo DANE)")
        if not ok_ipp:
            # No aborta: el IPP tiene su propio ciclo de publicación y el modelo de bolsa no
            # depende de él. Pero tiene que verse.
            logger.warning("No se pudo actualizar el IPP; se sigue con la serie vigente.")

    # Paso 2: reconstruir feature matrices
    ok = _run([_PYTHON, "scripts/construir_features.py"], "Construyendo feature matrices")
    if not ok:
        logger.error("Error construyendo features. Abortando.")
        return

    # Paso 3: ciclos de modelos
    if not args.solo_ipp:
        ciclo_bolsa(run_dir, args.fecha_corte)

    if not args.solo_bolsa:
        ciclo_ipp(run_dir, args.fecha_corte)

    logger.info("Ciclo mensual completado. Resultados en %s", run_dir)


if __name__ == "__main__":
    main()
