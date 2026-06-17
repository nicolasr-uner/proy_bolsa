"""Features hidrologicas derivadas para el modelo de precio de bolsa.

Las metricas crudas de XM vienen como fracciones (0-1); aqui se escalan,
se calculan ratios derivados y se clasifica el regimen hidrologico.

Referencia con modelo R:
  v_max_t  (vol max tecnico) -> capacidad_util_energia  -> proporcion respecto capacidad
  moi      (aportes)         -> aportes_pct (x100)      -> % de la media historica
  mos      (generacion)      -> no usada directamente en features; se usa demanda
  vertimientos               -> vertimientos             -> ratio sobre demanda diaria
"""

from __future__ import annotations

import pandas as pd

# Umbrales para clasificar regimen hidrologico (Colombia, sistema SIN)
# Basados en percentiles historicos tipicos
_UMBRAL_SECO_APORTES = 0.70      # < 70% de la media: condicion seca
_UMBRAL_HUMEDO_APORTES = 1.30    # > 130% de la media: condicion humeda
_UMBRAL_SECO_EMBALSE = 0.40      # < 40% de la capacidad: riesgo alto
_UMBRAL_HUMEDO_EMBALSE = 0.70    # > 70% de la capacidad: condicion comoda


def escalar_fracciones(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de fraccion (0-1) de XM a porcentaje (0-100).

    Columnas afectadas: aportes_pct, volumen_util_pct (si existen en el DataFrame).
    """
    df = df.copy()
    for col in ("aportes_pct", "volumen_util_pct"):
        if col in df.columns:
            df[col] = df[col] * 100.0
    return df


def agregar_ratio_volumen(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el ratio volumen_util / capacidad_util como verificacion alternativa.

    Solo se calcula si existen 'volumen_util_energia' y 'capacidad_util_energia'.
    Deberia concordar con 'volumen_util_pct' (ya escalado).
    """
    df = df.copy()
    if "volumen_util_energia" in df.columns and "capacidad_util_energia" in df.columns:
        df["volumen_util_ratio_calc"] = (
            df["volumen_util_energia"] / df["capacidad_util_energia"].replace(0, float("nan"))
        )
    return df


def agregar_ratio_vertimientos(
    df_hidro: pd.DataFrame, df_demanda: pd.DataFrame
) -> pd.DataFrame:
    """Calcula vertimientos como fraccion de la demanda diaria.

    Requiere que ambos DataFrames tengan la columna 'fecha'.
    """
    merged = df_hidro.merge(df_demanda[["fecha", "demanda_sin"]], on="fecha", how="left")
    merged["vertimientos_ratio"] = (
        merged["vertimientos"] / merged["demanda_sin"].replace(0, float("nan"))
    )
    return merged.drop(columns=["demanda_sin"])


def clasificar_regimen_hidro(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la columna 'regimen_hidro' con valores 'seco', 'normal', 'humedo'.

    Requiere columnas 'aportes_pct' (0-100) y 'volumen_util_pct' (0-100).
    El regimen se determina por ambas condiciones en conjunto (si ambas apuntan
    al mismo extremo, ese regimen prevalece).
    """
    df = df.copy()
    if "aportes_pct" not in df.columns:
        return df

    ap = df["aportes_pct"] / 100.0  # volver a fraccion para comparar con umbrales
    seco_ap = ap < _UMBRAL_SECO_APORTES
    humedo_ap = ap > _UMBRAL_HUMEDO_APORTES

    if "volumen_util_pct" in df.columns:
        vp = df["volumen_util_pct"] / 100.0
        seco_emb = vp < _UMBRAL_SECO_EMBALSE
        humedo_emb = vp > _UMBRAL_HUMEDO_EMBALSE
        seco = seco_ap | seco_emb
        humedo = humedo_ap & humedo_emb
    else:
        seco = seco_ap
        humedo = humedo_ap

    df["regimen_hidro"] = "normal"
    df.loc[seco, "regimen_hidro"] = "seco"
    df.loc[humedo, "regimen_hidro"] = "humedo"
    return df
