"""Dashboard de pronósticos — Precio de Bolsa e IPP Colombia.

Ejecutar:
    streamlit run dashboard/app.py

Requiere streamlit y plotly instalados en el entorno activo.
Los datos provienen de outputs/runs/{ultimo_ciclo}/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "outputs" / "runs"
PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Configuracion de la pagina
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Proyecciones Energía Colombia",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _ultimo_run() -> Path | None:
    runs = sorted(RUNS.glob("*/"))
    return runs[-1] if runs else None


@st.cache_data(ttl=300)
def _cargar_df(patron: str) -> pd.DataFrame | None:
    run = _ultimo_run()
    if run is None:
        return None
    archivos = sorted(run.glob(patron))
    if not archivos:
        return None
    return pd.read_parquet(archivos[-1])


@st.cache_data(ttl=300)
def _cargar_historico_diario() -> pd.DataFrame | None:
    p = PROCESSED / "bolsa_features_diario.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)[["fecha", "precio_bolsa_mean"]].dropna()
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.tail(180)  # ultimos 6 meses para el grafico


@st.cache_data(ttl=300)
def _cargar_historico_ipp() -> pd.DataFrame | None:
    p = PROCESSED / "ipp_features_mensual.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "ipp" not in df.columns:
        return None
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df[["fecha", "ipp"]].dropna().tail(36)


@st.cache_data(ttl=300)
def _cargar_resumen() -> dict:
    run = _ultimo_run()
    if run is None:
        return {}
    p = run / "resumen_bolsa.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def _cargar_perfil() -> pd.DataFrame | None:
    p = PROCESSED / "perfil_horario.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _color_escenario(esc: str) -> str:
    return {"seco": "#E05252", "promedio": "#2E86AB", "humedo": "#44BBA4"}.get(esc, "#888")


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

st.title("⚡ Proyecciones de Energía — Colombia")

run_actual = _ultimo_run()
if run_actual:
    st.caption(f"Último ciclo: **{run_actual.name}**")
else:
    st.warning("No se encontraron datos. Ejecuta `run_monthly_update.py` primero.")
    st.stop()

# Tabs principales
tab_bolsa, tab_horario, tab_escenarios, tab_ipp, tab_precision = st.tabs([
    "Precio Bolsa (corto)",
    "Perfil Horario",
    "Escenarios Hidrológicos",
    "IPP",
    "Seguimiento de Precisión",
])


# ---------------------------------------------------------------------------
# Tab 1: Pronóstico de bolsa corto plazo
# ---------------------------------------------------------------------------

with tab_bolsa:
    df_hist = _cargar_historico_diario()
    df_fc = _cargar_df("pronostico_bolsa_corto_diario_*.parquet")

    if df_fc is None:
        st.info("Sin pronostico corto disponible.")
    else:
        df_fc["fecha"] = pd.to_datetime(df_fc["fecha"])

        fig = go.Figure()

        if df_hist is not None:
            fig.add_trace(go.Scatter(
                x=df_hist["fecha"], y=df_hist["precio_bolsa_mean"],
                mode="lines", name="Histórico",
                line={"color": "#444", "width": 1.5},
            ))

        fig.add_trace(go.Scatter(
            x=pd.concat([df_fc["fecha"], df_fc["fecha"][::-1]]),
            y=pd.concat([df_fc["ci_hi90"], df_fc["ci_lo90"][::-1]]),
            fill="toself", fillcolor="rgba(46,134,171,0.18)",
            line={"color": "rgba(0,0,0,0)"}, showlegend=True,
            name="IC 90%",
        ))
        fig.add_trace(go.Scatter(
            x=df_fc["fecha"], y=df_fc["pred_diaria"],
            mode="lines+markers", name="Pronóstico",
            line={"color": "#2E86AB", "width": 2.5},
            marker={"size": 6},
        ))

        fig.update_layout(
            title="Precio de Bolsa Diario — Próximos 7 días",
            xaxis_title="Fecha",
            yaxis_title="COP/kWh",
            hovermode="x unified",
            legend={"orientation": "h", "y": -0.15},
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Promedio 7d", f"{df_fc['pred_diaria'].mean():.0f} COP/kWh")
        col2.metric("Min", f"{df_fc['pred_diaria'].min():.0f}")
        col3.metric("Max", f"{df_fc['pred_diaria'].max():.0f}")

    resumen = _cargar_resumen()
    if resumen:
        with st.expander("Diagnóstico del modelo"):
            col1, col2 = st.columns(2)
            col1.metric("Peso SARIMAX", f"{resumen.get('w_sarimax', 0):.1%}")
            col1.metric("Peso LGB", f"{resumen.get('w_lgb', 0):.1%}")
            col1.metric("AIC SARIMAX", resumen.get("aic_sarimax", "—"))
            col2.metric("Días de entrenamiento", resumen.get("n_train", "—"))
            col2.metric("Celdas del perfil", resumen.get("perfil_celdas", "—"))
            top = resumen.get("top_features_lgb", {})
            if top:
                st.caption("**Top features LGB (gain)**")
                st.dataframe(
                    pd.Series(top).rename("gain").sort_values(ascending=False).reset_index().rename(columns={"index": "feature"}),
                    use_container_width=True, hide_index=True,
                )


# ---------------------------------------------------------------------------
# Tab 2: Perfil horario
# ---------------------------------------------------------------------------

with tab_horario:
    df_perf = _cargar_perfil()
    df_fc_h = _cargar_df("pronostico_bolsa_corto_horario_*.parquet")

    if df_perf is None:
        st.info("Sin perfil horario disponible.")
    else:
        # Mapa de calor: tipo_dia x hora -> factor
        tipo_dias = sorted(df_perf["tipo_dia"].unique())
        pivot = df_perf.pivot_table(values="factor", index="tipo_dia", columns="hora", aggfunc="mean")

        fig_heatmap = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="RdYlGn",
            text=[[f"{v:.2f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar={"title": "Factor"},
        ))
        fig_heatmap.update_layout(
            title="Perfil Horario (factor multiplicador sobre nivel diario)",
            xaxis_title="Hora del día",
            yaxis_title="Tipo de día",
            height=280,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    if df_fc_h is not None:
        df_fc_h["timestamp"] = pd.to_datetime(df_fc_h["timestamp"])
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=df_fc_h["timestamp"], y=df_fc_h["ci_hi90_horario"],
            fill=None, mode="lines", line={"width": 0}, showlegend=False,
        ))
        fig_h.add_trace(go.Scatter(
            x=df_fc_h["timestamp"], y=df_fc_h["ci_lo90_horario"],
            fill="tonexty", fillcolor="rgba(46,134,171,0.18)",
            mode="lines", line={"width": 0}, name="IC 90% horario",
        ))
        fig_h.add_trace(go.Scatter(
            x=df_fc_h["timestamp"], y=df_fc_h["pred_horaria"],
            mode="lines", name="Pronóstico horario",
            line={"color": "#2E86AB", "width": 1.5},
        ))
        fig_h.update_layout(
            title="Precio Horario — Próximas 168 horas",
            xaxis_title="Timestamp",
            yaxis_title="COP/kWh",
            height=360, hovermode="x unified",
            legend={"orientation": "h", "y": -0.15},
        )
        st.plotly_chart(fig_h, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Escenarios hidrológicos (30d)
# ---------------------------------------------------------------------------

with tab_escenarios:
    df_tac = _cargar_df("pronostico_bolsa_tactico_*.parquet")

    if df_tac is None:
        st.info("Sin pronostico táctico disponible.")
    else:
        df_tac["fecha"] = pd.to_datetime(df_tac["fecha"])
        fig_esc = go.Figure()

        for esc in ["seco", "promedio", "humedo"]:
            sub = df_tac[df_tac["escenario"] == esc]
            color = _color_escenario(esc)
            fig_esc.add_trace(go.Scatter(
                x=sub["fecha"], y=sub["pred_diaria"],
                mode="lines", name=esc.capitalize(),
                line={"color": color, "width": 2},
            ))
            fig_esc.add_trace(go.Scatter(
                x=pd.concat([sub["fecha"], sub["fecha"][::-1]]),
                y=pd.concat([sub["ci_hi90"], sub["ci_lo90"][::-1]]),
                fill="toself",
                fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if "rgb" in color
                          else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
                line={"color": "rgba(0,0,0,0)"},
                showlegend=False,
            ))

        fig_esc.update_layout(
            title="Escenarios Hidrológicos — Próximos 30 días",
            xaxis_title="Fecha",
            yaxis_title="COP/kWh",
            hovermode="x unified",
            height=420,
            legend={"orientation": "h", "y": -0.15},
        )
        st.plotly_chart(fig_esc, use_container_width=True)

        # Tabla resumen
        resumen_esc = df_tac.groupby("escenario")["pred_diaria"].agg(
            Media="mean", Min="min", Max="max"
        ).round(0).reset_index()
        resumen_esc.columns = ["Escenario", "Promedio (COP/kWh)", "Min", "Max"]
        st.dataframe(resumen_esc, use_container_width=True, hide_index=True)

    # Largo plazo (365d)
    df_largo = _cargar_df("pronostico_bolsa_largo_*.parquet")
    if df_largo is not None:
        df_largo["fecha"] = pd.to_datetime(df_largo["fecha"])
        fig_largo = go.Figure()
        for esc in ["seco", "promedio", "humedo"]:
            sub = df_largo[df_largo["escenario"] == esc]
            sub_agg = sub.resample("ME", on="fecha")["pred_diaria"].mean().reset_index()
            fig_largo.add_trace(go.Scatter(
                x=sub_agg["fecha"], y=sub_agg["pred_diaria"],
                mode="lines+markers", name=esc.capitalize(),
                line={"color": _color_escenario(esc)},
            ))
        fig_largo.update_layout(
            title="Proyección a 12 meses — Media mensual por escenario",
            xaxis_title="Mes", yaxis_title="COP/kWh",
            hovermode="x unified", height=360,
            legend={"orientation": "h", "y": -0.15},
        )
        st.plotly_chart(fig_largo, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: IPP
# ---------------------------------------------------------------------------

with tab_ipp:
    df_hist_ipp = _cargar_historico_ipp()
    df_fc_ipp = _cargar_df("pronostico_ipp_12m_*.parquet")

    if df_fc_ipp is None and df_hist_ipp is None:
        st.info("Sin datos IPP. Cargar `data/raw/macro/ipp_manual.csv` y ejecutar `construir_features.py`.")
    else:
        fig_ipp = go.Figure()

        if df_hist_ipp is not None:
            fig_ipp.add_trace(go.Scatter(
                x=df_hist_ipp["fecha"], y=df_hist_ipp["ipp"],
                mode="lines", name="Histórico IPP",
                line={"color": "#444", "width": 1.5},
            ))

        if df_fc_ipp is not None:
            df_fc_ipp["fecha"] = pd.to_datetime(df_fc_ipp["fecha"])
            fig_ipp.add_trace(go.Scatter(
                x=pd.concat([df_fc_ipp["fecha"], df_fc_ipp["fecha"][::-1]]),
                y=pd.concat([df_fc_ipp["ci_hi90"], df_fc_ipp["ci_lo90"][::-1]]),
                fill="toself", fillcolor="rgba(68,187,164,0.18)",
                line={"color": "rgba(0,0,0,0)"}, showlegend=True, name="IC 90%",
            ))
            fig_ipp.add_trace(go.Scatter(
                x=df_fc_ipp["fecha"], y=df_fc_ipp["pred"],
                mode="lines+markers", name="Pronóstico IPP",
                line={"color": "#44BBA4", "width": 2.5},
                marker={"size": 6},
            ))

        fig_ipp.update_layout(
            title="IPP Colombia — Histórico y Pronóstico 12 meses",
            xaxis_title="Fecha", yaxis_title="Índice (base dic-2014=100)",
            hovermode="x unified", height=420,
            legend={"orientation": "h", "y": -0.15},
        )
        st.plotly_chart(fig_ipp, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5: Seguimiento de precisión
# ---------------------------------------------------------------------------

with tab_precision:
    resumen = _cargar_resumen()
    errores = resumen.get("errores_ciclo_anterior", [])

    if not errores:
        st.info("Sin errores registrados aún — aparecerán a partir del segundo ciclo mensual.")
        st.markdown("""
        **¿Cómo funciona el seguimiento?**

        Cada vez que `run_monthly_update.py` se ejecuta:
        1. Compara los pronósticos del mes anterior contra los precios realizados
        2. Calcula MAE, RMSE y sesgo por horizonte
        3. Actualiza los pesos del ensemble para compensar el sesgo reciente
        4. Registra el error en `outputs/runs/{YYYY-MM}/resumen_bolsa.json`
        """)
    else:
        df_err = pd.DataFrame(errores)
        st.subheader("Error del ciclo anterior")
        st.dataframe(
            df_err[["horizonte", "n_obs", "mae", "rmse", "sesgo", "mape_pct"]].rename(columns={
                "horizonte": "Horizonte (d)",
                "n_obs": "N obs",
                "mae": "MAE (COP/kWh)",
                "rmse": "RMSE (COP/kWh)",
                "sesgo": "Sesgo (pred−real)",
                "mape_pct": "MAPE %",
            }).round(1),
            use_container_width=True, hide_index=True,
        )

    # Historial de todos los ciclos
    all_resumenes = []
    for run_path in sorted(RUNS.glob("*/resumen_bolsa.json")):
        try:
            with open(run_path) as f:
                r = json.load(f)
            r["periodo"] = run_path.parent.name
            all_resumenes.append(r)
        except Exception:
            pass

    if len(all_resumenes) > 1:
        st.subheader("Evolución de pesos del ensemble")
        df_pesos = pd.DataFrame([
            {"periodo": r["periodo"], "SARIMAX": r.get("w_sarimax", 0), "LGB": r.get("w_lgb", 0)}
            for r in all_resumenes
        ])
        fig_pesos = go.Figure()
        for modelo in ["SARIMAX", "LGB"]:
            fig_pesos.add_trace(go.Scatter(
                x=df_pesos["periodo"], y=df_pesos[modelo],
                mode="lines+markers", name=modelo,
            ))
        fig_pesos.update_layout(
            title="Pesos del ensemble por ciclo mensual",
            yaxis_title="Peso", hovermode="x unified", height=300,
            legend={"orientation": "h", "y": -0.15},
        )
        st.plotly_chart(fig_pesos, use_container_width=True)

    st.subheader("Referencia: Backtest rolling-origin (62 orígenes)")
    st.markdown("""
    | Horizonte | RMSE ensemble | RMSE naive | Sesgo |
    |-----------|:---:|:---:|:---:|
    | 1d  | 352 | 271 | +62  |
    | 7d  | 359 | 271 | +107 |
    | 14d | 496 | 321 | +130 |
    | 30d | 844 | 348 | +262 |

    > **Nota:** El sesgo positivo refleja la transición del régimen El Niño 2023-2024
    > (precio ~800–2000 COP/kWh) al período post-Niño 2025-2026 (~465 COP/kWh).
    > El loop mensual lo corrige gradualmente via `actualizar_sesgo()`.
    """)
