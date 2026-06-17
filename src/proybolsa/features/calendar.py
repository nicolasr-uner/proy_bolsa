"""Features de calendario para el mercado electrico colombiano.

Clasifica cada timestamp/fecha en tipo de dia (habil, sabado, domingo_festivo)
y agrega features de periodo (hora, dia_semana, mes, etc.) usados en los modelos.

El tipo de dia determina el perfil horario de consumo y precio:
  habil          — lunes-viernes, no festivo
  sabado         — sabado, no festivo
  domingo_festivo — domingo O festivo nacional (independientemente del dia de semana)
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from typing import Literal

import pandas as pd

try:
    import holidays as hlib
    _HAS_HOLIDAYS = True
except ImportError:
    _HAS_HOLIDAYS = False


DayType = Literal["habil", "sabado", "domingo_festivo"]


@lru_cache(maxsize=8)
def festivos_colombia(anio: int) -> frozenset[dt.date]:
    """Festivos nacionales de Colombia para un ano dado.

    Usa el paquete `holidays` (Colombia subdivision CO).
    Si no esta instalado, devuelve el conjunto vacio y lanza un aviso.
    """
    if not _HAS_HOLIDAYS:
        import warnings
        warnings.warn(
            "El paquete 'holidays' no esta instalado. "
            "Los dias festivos no se reconoceran. "
            "Instalar con: pip install 'holidays>=0.46'",
            stacklevel=2,
        )
        return frozenset()
    return frozenset(hlib.Colombia(years=anio).keys())


def tipo_dia(fecha: dt.date) -> DayType:
    """Devuelve el tipo de dia para una fecha colombiana.

    Returns: 'habil', 'sabado' o 'domingo_festivo'.
    """
    dow = fecha.weekday()  # 0=lun ... 6=dom
    es_festivo = fecha in festivos_colombia(fecha.year)
    if es_festivo or dow == 6:
        return "domingo_festivo"
    if dow == 5:
        return "sabado"
    return "habil"


def agregar_features_calendario(df: pd.DataFrame, col_ts: str = "timestamp") -> pd.DataFrame:
    """Agrega features de calendario a un DataFrame con columna de timestamp horario.

    Agrega:
      hora          int (0-23)
      dia_semana    int (0=lunes, 6=domingo)
      mes           int (1-12)
      anio          int
      semana_iso    int (1-53)
      tipo_dia      str ('habil', 'sabado', 'domingo_festivo')
      es_festivo    bool
      es_fin_semana bool

    Parametros
    ----------
    df      : DataFrame con la columna de timestamp
    col_ts  : nombre de la columna con los timestamps (default 'timestamp')
    """
    df = df.copy()
    ts = pd.to_datetime(df[col_ts])

    df["hora"] = ts.dt.hour
    df["dia_semana"] = ts.dt.dayofweek
    df["mes"] = ts.dt.month
    df["anio"] = ts.dt.year
    df["semana_iso"] = ts.dt.isocalendar().week.astype(int)

    fechas = ts.dt.date
    df["tipo_dia"] = fechas.apply(tipo_dia)
    df["es_festivo"] = fechas.apply(lambda d: d in festivos_colombia(d.year))
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6]) | df["es_festivo"]
    return df


def agregar_features_fecha(df: pd.DataFrame, col_fecha: str = "fecha") -> pd.DataFrame:
    """Version para DataFrames con columna de fecha diaria (no timestamp horario).

    Agrega: dia_semana, mes, anio, semana_iso, tipo_dia, es_festivo, es_fin_semana.
    """
    df = df.copy()
    fechas = pd.to_datetime(df[col_fecha]).dt.date
    df["dia_semana"] = pd.to_datetime(df[col_fecha]).dt.dayofweek
    df["mes"] = pd.to_datetime(df[col_fecha]).dt.month
    df["anio"] = pd.to_datetime(df[col_fecha]).dt.year
    df["semana_iso"] = pd.to_datetime(df[col_fecha]).dt.isocalendar().week.astype(int)
    df["tipo_dia"] = fechas.apply(tipo_dia)
    df["es_festivo"] = fechas.apply(lambda d: d in festivos_colombia(d.year))
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6]) | df["es_festivo"]
    return df
