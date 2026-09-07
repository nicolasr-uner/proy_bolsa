"""Scorer prequencial: resuelve pronosticos vencidos contra el IPP real y agrega el leaderboard.

`resolver` empareja cada fila del registro cuya fecha_objetivo ya tiene IPP real con ese valor
y calcula el error. `agregar_leaderboard` resume por (modelo, horizonte) con skill vs. naive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def resolver(registro_df: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por pronostico vencido: y_pred, y_real, error.

    actuals: DataFrame con columnas fecha (mensual, primer dia) e ipp (real).
    Solo se resuelven filas con y_pred no-nulo y cuya fecha_objetivo existe en actuals.
    """
    reales = actuals[["fecha", "ipp"]].rename(columns={"fecha": "fecha_objetivo", "ipp": "y_real"})
    df = registro_df.dropna(subset=["y_pred"]).merge(reales, on="fecha_objetivo", how="inner")
    df["error"] = df["y_pred"] - df["y_real"]
    df["fecha_resuelto"] = pd.Timestamp.now().normalize()
    return df[["origen_fecha", "modelo", "horizonte", "fecha_objetivo",
               "y_pred", "y_real", "error", "fecha_resuelto"]].reset_index(drop=True)


def agregar_leaderboard(resueltos_df: pd.DataFrame,
                        naive: str = "ipp_bench_drift") -> pd.DataFrame:
    """Agrega por (modelo, horizonte): n, rmse, mae, skill vs. naive, ultima fecha."""
    if resueltos_df.empty:
        return pd.DataFrame(columns=["modelo", "horizonte", "n_resueltos",
                                     "rmse", "mae", "skill_vs_naive", "ultima_fecha"])

    def _agg(g):
        return pd.Series({
            "n_resueltos": len(g),
            "rmse": float(np.sqrt((g["error"] ** 2).mean())),
            "mae": float(g["error"].abs().mean()),
            "ultima_fecha": g["fecha_objetivo"].max(),
        })

    lb = resueltos_df.groupby(["modelo", "horizonte"]).apply(
        _agg, include_groups=False
    ).reset_index()

    rmse_naive = lb[lb["modelo"] == naive].set_index("horizonte")["rmse"]
    lb["skill_vs_naive"] = lb.apply(
        lambda r: (1 - r["rmse"] / rmse_naive[r["horizonte"]])
        if r["horizonte"] in rmse_naive.index and rmse_naive[r["horizonte"]] > 0 else np.nan,
        axis=1,
    )
    return lb
