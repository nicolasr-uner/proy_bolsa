"""Perfil horario del precio de bolsa (y demanda) por tipo de dia y mes.

La descomposicion del precio horario es:
    precio_horario(dia, hora) = nivel_diario(dia) * perfil(hora | tipo_dia, mes)

El perfil es la forma normalizada (factor multiplicador sobre el promedio diario).
Un perfil de 1.0 en la hora H significa que esa hora tiene el precio promedio del dia.

Ventajas frente a modelar 24 series acopladas:
  - El nivel diario lo explican los fundamentales (hidrologia, demanda, ENSO)
  - El perfil es mucho mas estable que el nivel absoluto
  - Se puede actualizar el perfil con datos recientes sin reentrenar todo el modelo
  - Para horizontes largos (PPAs) el perfil historico es suficiente

Tipos de dia (ver features/calendar.py):
  habil             — lun-vie, no festivo
  sabado            — sabado, no festivo
  domingo_festivo   — domingo o festivo
"""

from __future__ import annotations

import pandas as pd


def estimar_perfil_horario(
    df_horario: pd.DataFrame,
    col_precio: str = "precio_bolsa",
    col_ts: str = "timestamp",
    min_dias_por_celda: int = 20,
) -> pd.DataFrame:
    """Estima el perfil horario normalizado por (tipo_dia, mes, hora).

    Requiere que df_horario ya tenga las columnas 'tipo_dia', 'mes', 'hora'
    (agregar con features.calendar.agregar_features_calendario).

    Algoritmo:
      1. Calcular el precio promedio diario por timestamp.
      2. Dividir el precio de cada hora por el promedio de ese dia -> factor_hora.
      3. Promediar factor_hora por (tipo_dia, mes, hora) -> perfil.

    El resultado se suaviza a nivel de mes para evitar overfitting en meses
    con pocos datos en los extremos del historico.

    Returns
    -------
    DataFrame con columnas: tipo_dia, mes, hora, perfil_medio, perfil_p10, perfil_p90.
    Donde perfil_medio es el multiplicador esperado (cercano a 1.0 en promedio),
    y p10/p90 forman una banda para construir intervalos de precio.
    """
    required = {"tipo_dia", "mes", "hora", col_precio, col_ts}
    missing = required - set(df_horario.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en df_horario: {missing}")

    df = df_horario.copy()
    df["_fecha"] = pd.to_datetime(df[col_ts]).dt.date

    # Promedio diario del precio
    precio_diario = (
        df.groupby("_fecha")[col_precio]
        .mean()
        .rename("precio_diario_mean")
        .reset_index()
    )
    df = df.merge(precio_diario, on="_fecha", how="left")

    # Factor horario: precio_hora / precio_diario
    df["factor_hora"] = df[col_precio] / df["precio_diario_mean"].replace(0, float("nan"))

    # Agrupar por (tipo_dia, mes, hora)
    perfil = (
        df.groupby(["tipo_dia", "mes", "hora"])["factor_hora"]
        .agg(
            perfil_medio="mean",
            perfil_p10=lambda x: x.quantile(0.10),
            perfil_p90=lambda x: x.quantile(0.90),
            _n_obs="count",
        )
        .reset_index()
    )
    perfil = perfil.rename(columns={"_n_obs": "n_obs"})

    # Advertir si hay celdas con pocas observaciones
    n_pocas = (perfil["n_obs"] < min_dias_por_celda).sum()
    if n_pocas > 0:
        import warnings
        warnings.warn(
            f"{n_pocas} celdas del perfil tienen < {min_dias_por_celda} observaciones. "
            "Aumentar el historico o reducir min_dias_por_celda.",
            stacklevel=2,
        )

    return perfil.sort_values(["tipo_dia", "mes", "hora"]).reset_index(drop=True)


def aplicar_perfil_horario(
    forecast_diario: pd.DataFrame,
    perfil: pd.DataFrame,
    col_nivel_diario: str = "precio_bolsa_pred",
    col_fecha: str = "fecha",
) -> pd.DataFrame:
    """Expande un pronostico diario a 24 horas usando el perfil horario.

    Parametros
    ----------
    forecast_diario : DataFrame con columnas: fecha, tipo_dia, mes, col_nivel_diario
    perfil          : DataFrame con columnas: tipo_dia, mes, hora, perfil_medio
    col_nivel_diario: nombre de la columna con el nivel diario pronosticado

    Returns
    -------
    DataFrame con una fila por hora, columnas: timestamp, precio_bolsa_pred_horario.
    """
    required_f = {col_fecha, "tipo_dia", "mes", col_nivel_diario}
    missing_f = required_f - set(forecast_diario.columns)
    if missing_f:
        raise ValueError(f"Columnas faltantes en forecast_diario: {missing_f}")

    # Merge con perfil
    expanded = forecast_diario.merge(perfil[["tipo_dia", "mes", "hora", "perfil_medio"]], on=["tipo_dia", "mes"], how="left")

    # Construir timestamp
    expanded["timestamp"] = (
        pd.to_datetime(expanded[col_fecha])
        + pd.to_timedelta(expanded["hora"], unit="h")
    )
    expanded["precio_bolsa_pred_horario"] = expanded[col_nivel_diario] * expanded["perfil_medio"]

    return (
        expanded[["timestamp", "precio_bolsa_pred_horario"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def guardar_perfil(perfil: pd.DataFrame, ruta: str = "data/processed/perfil_horario.parquet") -> None:
    """Guarda el perfil estimado en Parquet para reutilizacion."""
    from pathlib import Path
    p = Path(ruta)
    p.parent.mkdir(parents=True, exist_ok=True)
    perfil.to_parquet(p, index=False)


def cargar_perfil(ruta: str = "data/processed/perfil_horario.parquet") -> pd.DataFrame:
    """Carga el perfil desde Parquet."""
    return pd.read_parquet(ruta)
