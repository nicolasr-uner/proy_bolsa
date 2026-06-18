"""Construye los feature matrices para los modelos de bolsa e IPP.

Lee los datos crudos de data/raw/, aplica el feature engineering y guarda
los DataFrames procesados en data/processed/:

  bolsa_features_horario.parquet    — serie horaria con todos los drivers
  bolsa_features_diario.parquet     — serie diaria (nivel) con drivers
  ipp_features_mensual.parquet      — serie mensual con drivers macro
  perfil_horario.parquet            — perfil 24h por tipo_dia y mes

Uso:
    .venv/Scripts/python scripts/construir_features.py
    .venv/Scripts/python scripts/construir_features.py --sin-perfil
    .venv/Scripts/python scripts/construir_features.py --solo-ipp
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from proybolsa.features.calendar import agregar_features_calendario, agregar_features_fecha
from proybolsa.features.enso import agregar_features_enso, dummy_enso
from proybolsa.features.hourly_profile import estimar_perfil_horario, guardar_perfil
from proybolsa.features.hydro import (
    agregar_ratio_vertimientos,
    clasificar_regimen_hidro,
    escalar_fracciones,
)
from proybolsa.features.lags import (
    agregar_lags_diarios,
    agregar_lags_horarios,
    agregar_lags_mensuales,
    agregar_rolling,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW_XM = ROOT / "data" / "raw" / "xm"
RAW_MACRO = ROOT / "data" / "raw" / "macro"
PROCESSED = ROOT / "data" / "processed"


def _leer(ruta: Path, nombre: str) -> pd.DataFrame | None:
    if not ruta.exists():
        logger.warning("  No encontrado: %s (saltar)", ruta.relative_to(ROOT))
        return None
    df = pd.read_parquet(ruta)
    logger.info("  Leido: %s (%d filas)", nombre, len(df))
    return df


def _guardar(df: pd.DataFrame, nombre: str) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ruta = PROCESSED / nombre
    df.to_parquet(ruta, index=False)
    logger.info("  Guardado: %s (%d filas, %d columnas)", nombre, len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Helpers de resampleo mensual
# ---------------------------------------------------------------------------

def _diario_a_mensual(df: pd.DataFrame, col_fecha: str, cols_valor: list[str]) -> pd.DataFrame:
    """Agrega un DataFrame diario a mensual tomando el promedio del mes."""
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df["fecha_mes"] = df[col_fecha].dt.to_period("M").dt.to_timestamp()
    return df.groupby("fecha_mes")[cols_valor].mean().reset_index().rename(columns={"fecha_mes": col_fecha})


# ---------------------------------------------------------------------------
# Feature matrix para precio de bolsa (horario + diario)
# ---------------------------------------------------------------------------

def construir_features_bolsa(incluir_perfil: bool = True) -> None:
    logger.info("=== Construyendo features BOLSA ===")

    # --- Cargar datos crudos ---
    df_precio = _leer(RAW_XM / "precio_bolsa_horario.parquet", "precio_bolsa_horario")
    df_escasez = _leer(RAW_XM / "precio_escasez.parquet", "precio_escasez")
    df_aportes = _leer(RAW_XM / "aportes_diarios.parquet", "aportes_diarios")
    df_embalses = _leer(RAW_XM / "embalses_diarios.parquet", "embalses_diarios")
    df_vert = _leer(RAW_XM / "vertimientos_diarios.parquet", "vertimientos_diarios")
    df_demanda = _leer(RAW_XM / "demanda_diaria.parquet", "demanda_diaria")
    df_oni = _leer(RAW_MACRO / "oni_mensual.parquet", "oni_mensual")

    if df_precio is None:
        logger.error("precio_bolsa_horario.parquet no disponible. Ejecutar descarga_historico.py primero.")
        return

    # --- Feature calendario (horario) ---
    logger.info("  Aplicando features de calendario...")
    df_precio = agregar_features_calendario(df_precio, col_ts="timestamp")

    # --- Lags horarios del precio ---
    logger.info("  Calculando lags horarios...")
    df_precio = agregar_lags_horarios(df_precio, "precio_bolsa", lags_horas=[24, 168])

    # --- Hidrologia: merge diario al horario ---
    logger.info("  Construyendo features hidrologicas...")
    df_precio["_fecha"] = df_precio["timestamp"].dt.date

    # Embalses + aportes juntos
    hidro = None
    if df_aportes is not None and df_embalses is not None:
        hidro = df_aportes.merge(df_embalses, on="fecha", how="outer")
        hidro = escalar_fracciones(hidro)
        if df_vert is not None:
            hidro = hidro.merge(df_vert, on="fecha", how="left")
        if df_demanda is not None:
            hidro = agregar_ratio_vertimientos(hidro, df_demanda)
        hidro = clasificar_regimen_hidro(hidro)
        # Lags de variables hidrologicas (1, 7, 30 dias)
        for col in ("aportes_pct", "volumen_util_pct"):
            if col in hidro.columns:
                hidro = agregar_lags_diarios(hidro, col, lags_dias=[1, 7, 30], col_fecha="fecha")
        # Merge al horario sin duplicar columnas calendario
        _CAL_COLS = ["dia_semana", "mes", "anio", "semana_iso", "tipo_dia", "es_festivo", "es_fin_semana"]
        hidro_sin_cal = hidro.drop(columns=[c for c in _CAL_COLS if c in hidro.columns], errors="ignore")
        df_precio = df_precio.merge(
            hidro_sin_cal.rename(columns={"fecha": "_fecha"}),
            on="_fecha",
            how="left",
        )
        # Version con calendario para df_diario
        hidro = agregar_features_fecha(hidro)

    # Precio de escasez (diario -> merge al horario)
    if df_escasez is not None:
        df_precio = df_precio.merge(
            df_escasez.rename(columns={"fecha": "_fecha"}),
            on="_fecha",
            how="left",
        )

    # --- ENSO ---
    if df_oni is not None:
        logger.info("  Agregando features ENSO...")
        df_precio = agregar_features_enso(df_precio, df_oni, rezago_meses=2, col_fecha="_fecha")
        df_precio = dummy_enso(df_precio)

    df_precio = df_precio.drop(columns=["_fecha"], errors="ignore")
    _guardar(df_precio, "bolsa_features_horario.parquet")

    # --- Serie diaria (para el modelo de nivel) ---
    logger.info("  Construyendo serie diaria...")
    df_diario = (
        df_precio.groupby(df_precio["timestamp"].dt.date)["precio_bolsa"]
        .agg(precio_bolsa_mean="mean", precio_bolsa_std="std", precio_bolsa_max="max")
        .reset_index()
        .rename(columns={"timestamp": "fecha"})
    )
    if hidro is not None:
        df_diario = df_diario.merge(hidro, on="fecha", how="left", suffixes=("", "_hidro"))
    # Lags del precio diario
    df_diario = agregar_lags_diarios(df_diario, "precio_bolsa_mean", lags_dias=[1, 7, 30])
    df_diario = agregar_rolling(df_diario, "precio_bolsa_mean", ventanas_dias=[7, 30])
    _guardar(df_diario, "bolsa_features_diario.parquet")

    # --- Perfil horario ---
    if incluir_perfil and "tipo_dia" in df_precio.columns and "hora" in df_precio.columns:
        logger.info("  Estimando perfil horario...")
        perfil = estimar_perfil_horario(df_precio, min_dias_por_celda=10)
        guardar_perfil(perfil, str(PROCESSED / "perfil_horario.parquet"))
        _guardar(perfil, "perfil_horario.parquet")
        logger.info("  Perfil: %d celdas (tipo_dia x mes x hora)", len(perfil))


# ---------------------------------------------------------------------------
# Feature matrix para IPP (mensual)
# ---------------------------------------------------------------------------

def construir_features_ipp() -> None:
    logger.info("=== Construyendo features IPP ===")

    # Intentar cargar IPP (puede no existir si no hay carga manual)
    df_ipp = _leer(RAW_MACRO / "ipp_mensual.parquet", "ipp_mensual")
    if df_ipp is None:
        from proybolsa.ingest.macros import load_ipp_local
        try:
            df_ipp = load_ipp_local()
            logger.info("  IPP cargado desde archivo manual (%d filas)", len(df_ipp))
        except FileNotFoundError:
            logger.warning(
                "  IPP no disponible. Cargar manualmente en data/raw/macro/ipp_manual.csv "
                "y volver a ejecutar. Solo se guardara el feature matrix de drivers macro."
            )

    # Macro drivers
    df_trm = _leer(RAW_MACRO / "trm_diaria.parquet", "trm_diaria")
    df_brent = _leer(RAW_MACRO / "brent_diario.parquet", "brent_diario")
    df_ppi = _leer(RAW_MACRO / "ppi_usa_mensual.parquet", "ppi_usa_mensual")
    df_oni = _leer(RAW_MACRO / "oni_mensual.parquet", "oni_mensual")

    # Agregar macro a mensual
    dfs_mensuales = []
    if df_trm is not None:
        trm_m = _diario_a_mensual(df_trm, "fecha", ["trm"])
        dfs_mensuales.append(trm_m)
    if df_brent is not None:
        brent_m = _diario_a_mensual(df_brent, "fecha", ["brent"])
        dfs_mensuales.append(brent_m)
    if df_ppi is not None:
        dfs_mensuales.append(df_ppi.rename(columns={"fecha": "fecha"}))

    # Construir feature matrix macro mensual
    if not dfs_mensuales:
        logger.warning("  Sin datos macro disponibles. Abortar feature matrix IPP.")
        return

    # Normalizar todas las fechas a datetime antes de mergear
    for i, d in enumerate(dfs_mensuales):
        dfs_mensuales[i] = d.assign(fecha=pd.to_datetime(d["fecha"]))

    macro = dfs_mensuales[0]
    for d in dfs_mensuales[1:]:
        macro = macro.merge(d, on="fecha", how="outer")
    macro = macro.sort_values("fecha").reset_index(drop=True)

    # Agregar mes (cyclical encoding)
    macro["mes"] = pd.to_datetime(macro["fecha"]).dt.month
    macro["cos_mes"] = macro["mes"].apply(lambda m: __import__("math").cos(2 * 3.14159 * m / 12))
    macro["sin_mes"] = macro["mes"].apply(lambda m: __import__("math").sin(2 * 3.14159 * m / 12))

    # Lags de drivers originales (mantenidos para compatibilidad con datos hist.)
    for col in ("trm", "brent", "ppi_usa"):
        if col in macro.columns:
            macro = agregar_lags_mensuales(macro, col, lags_meses=[1, 2, 3], col_fecha="fecha")

    # Drivers ortogonales (ver docs/estudio_drivers_ipp.md)
    # brent_cop = Brent_USD × TRM: precio del petróleo en pesos colombianos
    # Elimina la multicolinealidad TRM~PPI_USA=0.80 / Brent~PPI_USA=0.82
    if "brent" in macro.columns and "trm" in macro.columns:
        macro["brent_cop"] = macro["brent"] * macro["trm"]
        macro = agregar_lags_mensuales(macro, "brent_cop", lags_meses=[1, 2, 3], col_fecha="fecha")

    # Variaciones anuales: ortogonales al nivel; mejor correlación con ipp_log_dif
    macro = macro.sort_values("fecha").reset_index(drop=True)
    if "trm" in macro.columns:
        macro["trm_yoy"] = macro["trm"].pct_change(12) * 100
        macro = agregar_lags_mensuales(macro, "trm_yoy", lags_meses=[1], col_fecha="fecha")
    if "brent" in macro.columns:
        macro["brent_yoy"] = macro["brent"].pct_change(12) * 100
        macro = agregar_lags_mensuales(macro, "brent_yoy", lags_meses=[1], col_fecha="fecha")

    # ENSO con rezago 2 meses
    if df_oni is not None:
        macro = agregar_features_enso(macro, df_oni, rezago_meses=2, col_fecha="fecha")
        macro = dummy_enso(macro)

    # Si hay IPP, mergear y calcular lags del target
    if df_ipp is not None:
        df_ipp["fecha"] = pd.to_datetime(df_ipp["fecha"])
        macro["fecha"] = pd.to_datetime(macro["fecha"])
        macro = macro.merge(df_ipp, on="fecha", how="left")
        macro = agregar_lags_mensuales(macro, "ipp", lags_meses=[1, 2, 3, 12], col_fecha="fecha")
        # Variacion mensual log
        macro["ipp_log_dif"] = macro["ipp"].apply(__import__("numpy").log).diff()

    _guardar(macro, "ipp_features_mensual.parquet")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Construye feature matrices para los modelos.")
    parser.add_argument("--sin-perfil", action="store_true", help="No calcular el perfil horario.")
    parser.add_argument("--solo-ipp", action="store_true", help="Solo construir features IPP.")
    parser.add_argument("--solo-bolsa", action="store_true", help="Solo construir features bolsa.")
    args = parser.parse_args()

    if not args.solo_ipp:
        construir_features_bolsa(incluir_perfil=not args.sin_perfil)
    if not args.solo_bolsa:
        construir_features_ipp()

    logger.info("Feature matrices listos en %s", PROCESSED)


if __name__ == "__main__":
    main()
