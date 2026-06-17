"""Features de ENSO (El Nino / La Nina) para los modelos.

El indice ONI (Oceanic Nino Index) clasifica el regimen ENSO:
  El Nino  : ONI >= +0.5 por al menos 5 meses consecutivos
  La Nina  : ONI <= -0.5 por al menos 5 meses consecutivos
  Neutro   : entre -0.5 y +0.5

Para modelos de proyeccion a corto y medio plazo usamos el ONI sin el
criterio de persistencia (5 meses) — simplemente el umbral de +/-0.5
sobre el valor del mes. Esto es suficiente para capturar el regimen
hidrologico que afecta el precio de bolsa colombiano.

La transmision ENSO -> precio de bolsa se da con rezago de 2-3 meses
(el ENSO afecta los aportes hidricos que luego cambian el nivel de embalses).
"""

from __future__ import annotations

import pandas as pd

# Umbrales ONI estandar de NOAA
_ONI_NINO = 0.5
_ONI_NINA = -0.5


def clasificar_enso(oni: float) -> str:
    """Clasifica el regimen ENSO segun el valor del ONI."""
    if oni >= _ONI_NINO:
        return "el_nino"
    if oni <= _ONI_NINA:
        return "la_nina"
    return "neutro"


def agregar_features_enso(
    df: pd.DataFrame,
    df_oni: pd.DataFrame,
    rezago_meses: int = 2,
    col_fecha: str = "fecha",
) -> pd.DataFrame:
    """Agrega el regimen ENSO al DataFrame, con rezago de N meses.

    El rezago tipico ENSO -> hidrologia Colombia es 2-3 meses. El default de 2
    captura el efecto sobre el nivel de embalses con lag adecuado.

    Parametros
    ----------
    df          : DataFrame objetivo con columna de fecha diaria
    df_oni      : DataFrame de ONI mensual (columnas: fecha, oni)
    rezago_meses: meses de rezago para la variable ENSO
    col_fecha   : nombre de la columna de fecha en df

    Agrega columnas: 'oni_lag', 'enso_regime'
    """
    df = df.copy()
    df_oni = df_oni.copy()

    # Aplicar rezago desplazando las fechas del ONI hacia adelante
    df_oni["fecha"] = pd.to_datetime(df_oni["fecha"]) + pd.DateOffset(months=rezago_meses)
    df_oni["fecha"] = df_oni["fecha"].dt.date

    # Crear indice mensual en df para hacer join
    df["_fecha_mes"] = pd.to_datetime(df[col_fecha]).dt.to_period("M").dt.to_timestamp().dt.date

    # Join por mes
    oni_map = df_oni.set_index("fecha")["oni"].to_dict()
    df["oni_lag"] = df["_fecha_mes"].map(oni_map)
    df["enso_regime"] = df["oni_lag"].apply(
        lambda v: clasificar_enso(v) if pd.notna(v) else "neutro"
    )
    df = df.drop(columns=["_fecha_mes"])
    return df


def dummy_enso(df: pd.DataFrame, col: str = "enso_regime") -> pd.DataFrame:
    """Convierte el regimen ENSO a variables dummy (0/1).

    Agrega: enso_el_nino (bool), enso_la_nina (bool). Neutro = ambas cero.
    """
    df = df.copy()
    df["enso_el_nino"] = (df[col] == "el_nino").astype(int)
    df["enso_la_nina"] = (df[col] == "la_nina").astype(int)
    return df
