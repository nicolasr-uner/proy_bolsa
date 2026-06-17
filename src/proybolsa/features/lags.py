"""Features de rezago y estadisticas rodantes para series de tiempo.

Encapsula la logica de creacion de lags, rolling stats y variables de
tendencia para los modelos SARIMAX y LightGBM.

Convencion: los lags de N dias/horas se nombran como {col}_lag{N}{unidad},
donde unidad es 'h' para horas, 'd' para dias, 'm' para meses.
"""

from __future__ import annotations

import pandas as pd


def agregar_lags_diarios(
    df: pd.DataFrame,
    col: str,
    lags_dias: list[int] | None = None,
    col_fecha: str = "fecha",
) -> pd.DataFrame:
    """Agrega lags diarios a una serie con columna de fecha.

    Ordena por fecha antes de calcular, para garantizar que el lag es correcto
    aunque el DataFrame llegue desordenado.

    Parametros
    ----------
    df        : DataFrame con la serie objetivo
    col       : nombre de la columna a la que se calculan los lags
    lags_dias : lista de dias de rezago (default: [1, 7, 30])
    col_fecha : columna de fecha

    Agrega columnas: {col}_lag1d, {col}_lag7d, {col}_lag30d (segun lags_dias).
    """
    if lags_dias is None:
        lags_dias = [1, 7, 30]
    df = df.copy().sort_values(col_fecha).reset_index(drop=True)
    for d in lags_dias:
        df[f"{col}_lag{d}d"] = df[col].shift(d)
    return df


def agregar_lags_horarios(
    df: pd.DataFrame,
    col: str,
    lags_horas: list[int] | None = None,
    col_ts: str = "timestamp",
) -> pd.DataFrame:
    """Agrega lags horarios a una serie temporal horaria.

    Parametros
    ----------
    df         : DataFrame con una fila por hora
    col        : columna objetivo
    lags_horas : lista de horas de rezago (default: [1, 24, 168])
                 168 = 7 dias en horas
    col_ts     : columna de timestamp

    Agrega columnas: {col}_lag1h, {col}_lag24h, {col}_lag168h.
    """
    if lags_horas is None:
        lags_horas = [1, 24, 168]
    df = df.copy().sort_values(col_ts).reset_index(drop=True)
    for h in lags_horas:
        df[f"{col}_lag{h}h"] = df[col].shift(h)
    return df


def agregar_rolling(
    df: pd.DataFrame,
    col: str,
    ventanas_dias: list[int] | None = None,
    col_fecha: str = "fecha",
) -> pd.DataFrame:
    """Agrega media y desviacion estandar rodante a una serie diaria.

    Parametros
    ----------
    df             : DataFrame diario
    col            : columna objetivo
    ventanas_dias  : ventanas en dias (default: [7, 30])

    Agrega columnas: {col}_roll{W}d_mean, {col}_roll{W}d_std.
    """
    if ventanas_dias is None:
        ventanas_dias = [7, 30]
    df = df.copy().sort_values(col_fecha).reset_index(drop=True)
    for w in ventanas_dias:
        df[f"{col}_roll{w}d_mean"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_roll{w}d_std"] = df[col].rolling(w, min_periods=1).std()
    return df


def agregar_lags_mensuales(
    df: pd.DataFrame,
    col: str,
    lags_meses: list[int] | None = None,
    col_fecha: str = "fecha",
) -> pd.DataFrame:
    """Agrega lags mensuales a una serie con frecuencia mensual.

    Asume que df tiene una fila por mes (periodo mensual).

    Parametros
    ----------
    lags_meses : lista de meses (default: [1, 2, 3, 12])
    """
    if lags_meses is None:
        lags_meses = [1, 2, 3, 12]
    df = df.copy().sort_values(col_fecha).reset_index(drop=True)
    for m in lags_meses:
        df[f"{col}_lag{m}m"] = df[col].shift(m)
    return df
