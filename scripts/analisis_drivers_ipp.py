"""Analisis empirico de drivers del IPP Oferta Interna.

Computa:
- Correlaciones entre drivers actuales y IPP
- VIF de los drivers actuales (diagnostico de multicolinealidad)
- Elasticidades long-run (log-log OLS)
- Propuesta de features ortogonales y sus correlaciones con IPP

Salida: docs/estudio_drivers_ipp.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import inv

PARQUET = "data/processed/ipp_features_mensual.parquet"


def vif(X: np.ndarray, names: list[str]) -> dict[str, float]:
    R = np.corrcoef(X.T)
    vifs = np.diag(inv(R))
    return dict(zip(names, vifs))


def ols_loglog(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict[str, float]:
    """Regresion log-log simple (sin lags) para elasticidades long-run."""
    sub = df[[y_col] + x_cols].dropna()
    log_y = np.log(sub[y_col].values)
    X_raw = sub[x_cols].values
    # Solo log-transformar columnas positivas
    X = np.column_stack([
        np.log(X_raw[:, i]) if (X_raw[:, i] > 0).all() else X_raw[:, i]
        for i in range(X_raw.shape[1])
    ])
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_aug, log_y, rcond=None)
    return dict(zip(["const"] + x_cols, beta))


def main():
    df = pd.read_parquet(PARQUET)
    df_c = df.dropna(subset=["ipp", "trm", "brent", "ppi_usa"]).copy()

    print(f"Filas con datos completos: {len(df_c)}")
    print(f"Rango: {df_c['fecha'].min()} – {df_c['fecha'].max()}")
    print()

    # -----------------------------------------------------------------------
    # 1. Correlaciones entre drivers actuales
    # -----------------------------------------------------------------------
    drivers = ["trm", "brent", "ppi_usa"]
    print("=== Correlaciones entre drivers actuales ===")
    print(df_c[drivers].corr().round(4).to_string())
    print()

    # -----------------------------------------------------------------------
    # 2. VIF
    # -----------------------------------------------------------------------
    X_drv = df_c[drivers].values
    X_drv_std = (X_drv - X_drv.mean(0)) / X_drv.std(0)
    vifs = vif(X_drv_std, drivers)
    print("=== VIF de drivers actuales ===")
    for name, v in vifs.items():
        flag = " *** PROBLEMATICO" if v > 5 else ""
        print(f"  VIF {name}: {v:.2f}{flag}")
    print()

    # -----------------------------------------------------------------------
    # 3. Correlaciones con IPP (nivel y log-dif)
    # -----------------------------------------------------------------------
    print("=== Correlacion (nivel) con IPP ===")
    for d in drivers + ["oni_lag", "ipp_lag1m"]:
        if d in df_c.columns:
            print(f"  {d}: {df_c[d].corr(df_c['ipp']):.4f}")
    print()

    print("=== Correlacion con ipp_log_dif (cambio mensual) ===")
    dif = df_c.dropna(subset=["ipp_log_dif"])
    for d in drivers + ["oni_lag", "ipp_lag1m"]:
        if d in dif.columns:
            print(f"  {d}: {dif[d].corr(dif['ipp_log_dif']):.4f}")
    print()

    # -----------------------------------------------------------------------
    # 4. Elasticidades log-log OLS
    # -----------------------------------------------------------------------
    elast = ols_loglog(df_c, "ipp", drivers)
    print("=== Elasticidades log-log OLS (IPP ~ TRM + Brent + PPI_USA) ===")
    for k, v in elast.items():
        print(f"  beta_{k}: {v:+.4f}")
    print()

    # -----------------------------------------------------------------------
    # 5. Feature engineering: variables ortogonales propuestas
    # -----------------------------------------------------------------------
    # A) brent_cop = Brent x TRM  (precio en pesos del petróleo)
    df_c["brent_cop"] = df_c["brent"] * df_c["trm"]

    # B) real_trm = TRM / PPI_USA * 100  (tipo de cambio real simplificado)
    df_c["real_trm"] = df_c["trm"] / df_c["ppi_usa"] * 100

    # C) trm_yoy = variacion anual de TRM (captura depreciacion, no nivel)
    df_c = df_c.sort_values("fecha").reset_index(drop=True)
    df_c["trm_yoy"] = df_c["trm"].pct_change(12) * 100

    # D) brent_yoy = variacion anual de Brent
    df_c["brent_yoy"] = df_c["brent"].pct_change(12) * 100

    nuevas = ["brent_cop", "real_trm", "trm_yoy", "brent_yoy"]
    print("=== Correlaciones entre nuevas variables propuestas ===")
    df_n = df_c.dropna(subset=nuevas)
    print(df_n[nuevas].corr().round(4).to_string())
    print()

    print("=== Correlacion de nuevas variables con IPP (nivel) ===")
    for v in nuevas:
        print(f"  {v}: {df_n[v].corr(df_n['ipp']):.4f}")
    print()

    print("=== Correlacion de nuevas variables con ipp_log_dif ===")
    df_nd = df_n.dropna(subset=["ipp_log_dif"])
    for v in nuevas:
        print(f"  {v}: {df_nd[v].corr(df_nd['ipp_log_dif']):.4f}")
    print()

    # VIF de set propuesto: brent_cop + real_trm + oni_lag
    prop = ["brent_cop", "real_trm"]
    df_prop = df_n.dropna(subset=prop)
    X_prop = df_prop[prop].values
    X_prop_std = (X_prop - X_prop.mean(0)) / X_prop.std(0)
    vifs_prop = vif(X_prop_std, prop)
    print("=== VIF del set propuesto (brent_cop, real_trm) ===")
    for name, v in vifs_prop.items():
        flag = " *** PROBLEMATICO" if v > 5 else ""
        print(f"  VIF {name}: {v:.2f}{flag}")
    print()

    # Elasticidades con set propuesto
    elast2 = ols_loglog(df_prop, "ipp", ["brent_cop", "real_trm"])
    print("=== Elasticidades log-log OLS (IPP ~ brent_cop + real_trm) ===")
    for k, v2 in elast2.items():
        print(f"  beta_{k}: {v2:+.4f}")
    print()

    print("Analisis completado.")


if __name__ == "__main__":
    main()
