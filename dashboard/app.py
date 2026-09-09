"""Dashboard de proyecciones — Precio de Bolsa e IPP Colombia.

Ejecutar:
    streamlit run dashboard/app.py     (desde la raiz del proyecto)

Requiere streamlit y plotly instalados en el entorno activo.
Los modelos se ajustan una vez por sesion desde los parquets de features.
Los pronósticos historicos provienen de outputs/runs/{ultimo_ciclo}/.
"""

from __future__ import annotations

import io
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

ROOT = Path(__file__).parent.parent
# Necesario en Streamlit Community Cloud donde no hay editable install de proybolsa
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
RUNS = ROOT / "outputs" / "runs"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"

st.set_page_config(
    page_title="Proyecciones Energía Colombia",
    page_icon="⚡",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers: datos estáticos
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
    return df.tail(180)


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
# Helpers: modelos en vivo (cacheados por sesion — se ajustan una vez ~8s Bolsa / <1s IPP)
# ---------------------------------------------------------------------------

@st.cache_resource
def _modelo_bolsa():
    """Carga PronosticadorBolsa serializado (rápido) o lo ajusta en vivo (fallback).

    En Cloud el `.fit()` de SARIMAX bloquea el CPU del tier gratuito; por eso se
    prefiere el .pkl pre-ajustado (ver scripts/serializar_modelos.py). Si no existe
    (desarrollo local sin .pkl), se ajusta en vivo.
    """
    pkl = MODELS / "bolsa.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    p_feat = PROCESSED / "bolsa_features_diario.parquet"
    if not p_feat.exists():
        return None
    from proybolsa.models.bolsa.pronostico import PronosticadorBolsa
    df = pd.read_parquet(p_feat)
    modelo = PronosticadorBolsa()
    modelo.fit(df, ruta_perfil=str(PROCESSED / "perfil_horario.parquet"))
    return modelo


@st.cache_resource
def _modelo_ipp():
    """Carga PronosticadorIPP serializado (rápido) o lo ajusta en vivo (fallback).

    El `.fit()` del IPP (SARIMAX + Johansen/VECM) es lo que bloquea el tier gratuito
    de Streamlit Cloud; se prefiere el .pkl pre-ajustado. El `.pronosticar()` de los
    escenarios es barato y sigue corriendo en vivo.
    """
    pkl = MODELS / "ipp.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    p_feat = PROCESSED / "ipp_features_mensual.parquet"
    if not p_feat.exists():
        return None
    df = pd.read_parquet(p_feat).dropna(subset=["ipp"])
    if len(df) < 10:
        return None
    from proybolsa.models.ipp.modelo_ipp import PronosticadorIPP
    modelo = PronosticadorIPP()
    modelo.fit(df)
    return modelo


# ---------------------------------------------------------------------------
# Helper: Excel export
# ---------------------------------------------------------------------------

def _escenarios_a_excel(escenarios: list[dict], modelo_label: str) -> bytes:
    """Convierte lista de escenarios guardados a bytes de Excel."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Hoja 1: resumen de supuestos
        rows = []
        for i, esc in enumerate(escenarios):
            row = {"#": i + 1, "Nombre": esc["nombre"]}
            row.update({f"Param: {k}": v for k, v in esc.get("params", {}).items()})
            rows.append(row)
        pd.DataFrame(rows).to_excel(writer, sheet_name="Supuestos", index=False)

        # Hoja por escenario
        for esc in escenarios:
            nombre_hoja = esc["nombre"][:31]  # Excel limit
            fc = esc["fc"].copy()
            if "fecha" in fc.columns:
                fc["fecha"] = pd.to_datetime(fc["fecha"]).dt.strftime("%Y-%m-%d")
            fc.to_excel(writer, sheet_name=nombre_hoja, index=False)

        # Hoja comparativa
        if len(escenarios) > 1 and all("fc" in e for e in escenarios):
            frames = []
            for esc in escenarios:
                fc = esc["fc"][["fecha", "pred"]].copy() if "pred" in esc["fc"].columns else esc["fc"][["fecha", "pred_diaria"]].copy()
                fc.columns = ["fecha", esc["nombre"]]
                frames.append(fc.set_index("fecha"))
            comp = frames[0].join(frames[1:], how="outer")
            comp.reset_index().to_excel(writer, sheet_name="Comparacion", index=False)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Estado de sesion para escenarios guardados
# ---------------------------------------------------------------------------

if "escenarios_bolsa" not in st.session_state:
    st.session_state.escenarios_bolsa = []
if "escenarios_ipp" not in st.session_state:
    st.session_state.escenarios_ipp = []


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

st.title("⚡ Proyecciones de Energía — Colombia")

run_actual = _ultimo_run()
if run_actual:
    st.caption(f"Último ciclo mensual: **{run_actual.name}**  |  Modelos ajustados en vivo desde features parquet")
else:
    st.info("No se encontraron resultados de ciclos mensuales. Los modelos se ajustan directamente desde los parquets.")

tab_bolsa, tab_horario, tab_esc_bolsa, tab_ipp, tab_precision = st.tabs([
    "Precio Bolsa (corto)",
    "Perfil Horario",
    "Escenarios Bolsa",
    "IPP y Escenarios",
    "Seguimiento de Precision",
])


# ---------------------------------------------------------------------------
# Tab 1: Pronóstico de bolsa corto plazo (del ciclo mensual)
# ---------------------------------------------------------------------------

with tab_bolsa:
    df_hist = _cargar_historico_diario()
    df_fc = _cargar_df("pronostico_bolsa_corto_diario_*.parquet")

    if df_fc is None:
        st.info("Sin pronostico corto del ciclo mensual. Ejecuta `run_monthly_update.py`.")
    else:
        df_fc["fecha"] = pd.to_datetime(df_fc["fecha"])
        fig = go.Figure()
        if df_hist is not None:
            fig.add_trace(go.Scatter(
                x=df_hist["fecha"], y=df_hist["precio_bolsa_mean"],
                mode="lines", name="Historico", line={"color": "#444", "width": 1.5},
            ))
        fig.add_trace(go.Scatter(
            x=pd.concat([df_fc["fecha"], df_fc["fecha"][::-1]]),
            y=pd.concat([df_fc["ci_hi90"], df_fc["ci_lo90"][::-1]]),
            fill="toself", fillcolor="rgba(46,134,171,0.18)",
            line={"color": "rgba(0,0,0,0)"}, showlegend=True, name="IC 90%",
        ))
        fig.add_trace(go.Scatter(
            x=df_fc["fecha"], y=df_fc["pred_diaria"],
            mode="lines+markers", name="Pronostico",
            line={"color": "#2E86AB", "width": 2.5}, marker={"size": 6},
        ))
        fig.update_layout(
            title="Precio de Bolsa Diario - Proximos 7 dias",
            xaxis_title="Fecha", yaxis_title="COP/kWh",
            hovermode="x unified", legend={"orientation": "h", "y": -0.15}, height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Promedio 7d", f"{df_fc['pred_diaria'].mean():.0f} COP/kWh")
        col2.metric("Min", f"{df_fc['pred_diaria'].min():.0f}")
        col3.metric("Max", f"{df_fc['pred_diaria'].max():.0f}")

    resumen = _cargar_resumen()
    if resumen:
        with st.expander("Diagnostico del modelo"):
            col1, col2 = st.columns(2)
            col1.metric("Peso SARIMAX", f"{resumen.get('w_sarimax', 0):.1%}")
            col1.metric("Peso LGB", f"{resumen.get('w_lgb', 0):.1%}")
            col1.metric("AIC SARIMAX", resumen.get("aic_sarimax", ""))
            col2.metric("Dias entrenamiento", resumen.get("n_train", ""))
            col2.metric("Celdas del perfil", resumen.get("perfil_celdas", ""))
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

    if df_perf is not None:
        col_factor = "perfil_medio" if "perfil_medio" in df_perf.columns else (
            "factor" if "factor" in df_perf.columns else None
        )
        if col_factor:
            pivot = df_perf.pivot_table(values=col_factor, index="tipo_dia", columns="hora", aggfunc="mean")
            fig_heatmap = go.Figure(go.Heatmap(
                z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                colorscale="RdYlGn",
                text=[[f"{v:.2f}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                colorbar={"title": "Factor"},
            ))
            fig_heatmap.update_layout(
                title="Perfil Horario (factor multiplicador sobre nivel diario)",
                xaxis_title="Hora del dia", yaxis_title="Tipo de dia", height=280,
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Columna de factor no encontrada en perfil.")
    else:
        st.info("Sin perfil horario disponible.")

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
            mode="lines", name="Pronostico horario",
            line={"color": "#2E86AB", "width": 1.5},
        ))
        fig_h.update_layout(
            title="Precio Horario - Proximas 168 horas",
            xaxis_title="Timestamp", yaxis_title="COP/kWh",
            height=360, hovermode="x unified",
            legend={"orientation": "h", "y": -0.15},
        )
        st.plotly_chart(fig_h, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Escenarios Bolsa (INTERACTIVO)
# ---------------------------------------------------------------------------

with tab_esc_bolsa:
    st.subheader("Constructor de escenarios — Precio de Bolsa")
    st.caption(
        "Define los supuestos hidrologicos y macroeconómicos. "
        "El modelo se recalcula en <1 segundo sobre los datos mas recientes."
    )

    modelo_b = _modelo_bolsa()

    if modelo_b is None:
        st.warning("No se pudo ajustar el modelo de bolsa. Verifica que `bolsa_features_diario.parquet` exista.")
    else:
        # --- Panel de supuestos ---
        with st.form("form_bolsa"):
            c1, c2, c3 = st.columns(3)

            with c1:
                aportes = st.slider(
                    "Aportes hídricos (% del promedio histórico)", 40, 160, 100, step=5,
                    help="100 = normal, <80 = seco, >120 = húmedo",
                )
                volumen = st.slider(
                    "Volumen útil embalses (% de capacidad)", 20, 90, 60, step=5,
                )

            with c2:
                oni = st.slider(
                    "ONI asumido (ENSO)", -2.0, 2.5, 0.0, step=0.1,
                    help=">0.5 = El Niño (precios altos), <-0.5 = La Niña (precios bajos)",
                )
                escasez = st.slider(
                    "Precio de escasez (COP/kWh)", 800, 1200, 906, step=10,
                )

            with c3:
                horizonte_b = st.selectbox("Horizonte de pronóstico", [7, 30, 365], index=1,
                                           format_func=lambda h: f"{h} días")
                nombre_esc_b = st.text_input("Nombre del escenario", value="Escenario 1")

            submitted_b = st.form_submit_button("Calcular escenario", type="primary")

        if submitted_b:
            with st.spinner("Calculando..."):
                fc_b = modelo_b.pronosticar(
                    horizonte_b,
                    aportes_pct=float(aportes),
                    volumen_util_pct=float(volumen),
                    oni_asumido=float(oni),
                    precio_escasez=float(escasez),
                    devolver_horario=False,
                )
                fc_b["fecha"] = pd.to_datetime(fc_b["fecha"])
            st.session_state["_ultimo_fc_bolsa"] = fc_b
            st.session_state["_ultimo_params_bolsa"] = {
                "aportes_pct": aportes, "volumen_util_pct": volumen,
                "oni": oni, "escasez": escasez, "horizonte_dias": horizonte_b,
            }

        fc_b_actual = st.session_state.get("_ultimo_fc_bolsa")

        if fc_b_actual is not None:
            fig_b = go.Figure()

            # Histórico
            df_hist = _cargar_historico_diario()
            if df_hist is not None:
                fig_b.add_trace(go.Scatter(
                    x=df_hist["fecha"], y=df_hist["precio_bolsa_mean"],
                    mode="lines", name="Historico", line={"color": "#888", "width": 1.5},
                ))

            # Escenarios guardados (superpuestos)
            colores_saved = ["#E05252", "#44BBA4", "#F4A261", "#A663CC", "#2D6A4F"]
            for i, esc_g in enumerate(st.session_state.escenarios_bolsa):
                c = colores_saved[i % len(colores_saved)]
                fig_b.add_trace(go.Scatter(
                    x=esc_g["fc"]["fecha"], y=esc_g["fc"]["pred_diaria"],
                    mode="lines", name=esc_g["nombre"],
                    line={"color": c, "width": 1.5, "dash": "dot"},
                ))

            # Escenario actual
            fig_b.add_trace(go.Scatter(
                x=pd.concat([fc_b_actual["fecha"], fc_b_actual["fecha"][::-1]]),
                y=pd.concat([fc_b_actual["ci_hi90"], fc_b_actual["ci_lo90"][::-1]]),
                fill="toself", fillcolor="rgba(46,134,171,0.18)",
                line={"color": "rgba(0,0,0,0)"}, showlegend=True, name="IC 90%",
            ))
            fig_b.add_trace(go.Scatter(
                x=fc_b_actual["fecha"], y=fc_b_actual["pred_diaria"],
                mode="lines+markers", name="Actual",
                line={"color": "#2E86AB", "width": 2.5}, marker={"size": 5},
            ))
            fig_b.update_layout(
                title=f"Escenario de Bolsa — {horizonte_b} dias",
                xaxis_title="Fecha", yaxis_title="COP/kWh",
                hovermode="x unified", height=400,
                legend={"orientation": "h", "y": -0.18},
            )
            st.plotly_chart(fig_b, use_container_width=True)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Promedio", f"{fc_b_actual['pred_diaria'].mean():.0f} COP/kWh")
            col_m2.metric("Min", f"{fc_b_actual['pred_diaria'].min():.0f}")
            col_m3.metric("Max", f"{fc_b_actual['pred_diaria'].max():.0f}")
            col_m4.metric("IC ancho (prom.)", f"{(fc_b_actual['ci_hi90'] - fc_b_actual['ci_lo90']).mean():.0f}")

        # --- Guardar y comparar ---
        st.divider()
        col_g1, col_g2, col_g3 = st.columns([2, 1, 1])

        with col_g1:
            if st.button("Guardar escenario actual", disabled=fc_b_actual is None):
                params_b = st.session_state.get("_ultimo_params_bolsa", {})
                st.session_state.escenarios_bolsa.append({
                    "nombre": nombre_esc_b,
                    "params": params_b,
                    "fc": fc_b_actual.copy(),
                })
                st.success(f"Escenario '{nombre_esc_b}' guardado.")
                st.rerun()

        with col_g2:
            if st.button("Limpiar comparacion", disabled=not st.session_state.escenarios_bolsa):
                st.session_state.escenarios_bolsa = []
                st.rerun()

        with col_g3:
            if st.session_state.escenarios_bolsa:
                xlsx_bytes = _escenarios_a_excel(st.session_state.escenarios_bolsa, "bolsa")
                st.download_button(
                    "Exportar a Excel",
                    data=xlsx_bytes,
                    file_name="escenarios_bolsa.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        if st.session_state.escenarios_bolsa:
            st.subheader("Escenarios guardados")
            rows_g = []
            for esc_g in st.session_state.escenarios_bolsa:
                p = esc_g.get("params", {})
                rows_g.append({
                    "Nombre": esc_g["nombre"],
                    "Aportes (%)": p.get("aportes_pct", ""),
                    "Volumen (%)": p.get("volumen_util_pct", ""),
                    "ONI": p.get("oni", ""),
                    "Escasez": p.get("escasez", ""),
                    "Horizonte (d)": p.get("horizonte_dias", ""),
                    "Prom. COP/kWh": f"{esc_g['fc']['pred_diaria'].mean():.0f}",
                })
            st.dataframe(pd.DataFrame(rows_g), use_container_width=True, hide_index=True)

        # --- Escenarios hidrologicos estaticos (del ciclo mensual) ---
        st.divider()
        st.subheader("Escenarios hidrológicos del ciclo mensual (30 días)")
        df_tac = _cargar_df("pronostico_bolsa_tactico_*.parquet")
        if df_tac is not None:
            df_tac["fecha"] = pd.to_datetime(df_tac["fecha"])
            fig_esc = go.Figure()
            for esc in ["seco", "promedio", "humedo"]:
                sub = df_tac[df_tac["escenario"] == esc] if "escenario" in df_tac.columns else df_tac
                if len(sub) == 0:
                    continue
                color = _color_escenario(esc)
                fig_esc.add_trace(go.Scatter(
                    x=sub["fecha"], y=sub["pred_diaria"],
                    mode="lines", name=esc.capitalize(),
                    line={"color": color, "width": 2},
                ))
            fig_esc.update_layout(
                title="Escenarios Hidrologicos — 30 dias (ciclo mensual)",
                xaxis_title="Fecha", yaxis_title="COP/kWh",
                hovermode="x unified", height=360,
                legend={"orientation": "h", "y": -0.15},
            )
            st.plotly_chart(fig_esc, use_container_width=True)
        else:
            st.info("Ejecutar `run_monthly_update.py` para ver pronósticos del ciclo mensual.")


# ---------------------------------------------------------------------------
# Tab 4: IPP (histórico + constructor interactivo de escenarios)
# ---------------------------------------------------------------------------

with tab_ipp:
    # --- Histórico + pronóstico del ciclo mensual ---
    st.subheader("IPP Colombia — Histórico y Pronóstico Base")
    df_hist_ipp = _cargar_historico_ipp()
    df_fc_ipp = _cargar_df("pronostico_ipp_12m_*.parquet")

    fig_ipp = go.Figure()
    if df_hist_ipp is not None:
        fig_ipp.add_trace(go.Scatter(
            x=df_hist_ipp["fecha"], y=df_hist_ipp["ipp"],
            mode="lines", name="Historico IPP",
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
            mode="lines+markers", name="Pronostico base IPP",
            line={"color": "#44BBA4", "width": 2.5}, marker={"size": 6},
        ))
    if df_hist_ipp is None and df_fc_ipp is None:
        st.info("Sin datos IPP. Cargar `data/raw/macro/ipp_manual.csv` y ejecutar `construir_features.py`.")
    else:
        fig_ipp.update_layout(
            title="IPP Colombia — Historico y Pronostico 12 meses (base)",
            xaxis_title="Fecha", yaxis_title="Indice (base dic-2014=100)",
            hovermode="x unified", height=380,
            legend={"orientation": "h", "y": -0.15},
        )
        st.plotly_chart(fig_ipp, use_container_width=True)

    # --- Constructor de escenarios IPP ---
    st.divider()
    st.subheader("Constructor de escenarios — IPP")
    st.caption(
        "Define trayectorias de TRM y Brent. "
        "El efecto sobre el IPP es modesto en el corto plazo (IPP es autoregresivo); "
        "el impacto se acumula gradualmente en horizontes de 12-24 meses."
    )

    modelo_i = _modelo_ipp()

    if modelo_i is None:
        st.warning("No se pudo ajustar el modelo IPP. Verifica que `ipp_features_mensual.parquet` exista y tenga datos de IPP.")
    else:
        with st.form("form_ipp"):
            c1, c2, c3 = st.columns(3)

            with c1:
                trm_var = st.slider(
                    "Variacion TRM anual (%)",
                    -20, 50, 0, step=2,
                    help="Positivo = depreciacion del peso (TRM sube). Afecta costo de insumos importados.",
                )
                brent_var = st.slider(
                    "Variacion Brent anual (%)",
                    -30, 50, 0, step=2,
                    help="Variacion anual del precio del petroleo en USD/barril.",
                )

            with c2:
                oni_i = st.slider(
                    "ONI asumido (ENSO)", -2.0, 2.5, 0.0, step=0.1,
                    help=">0.5 = El Niño activo",
                )
                horizonte_i = st.selectbox(
                    "Horizonte de pronostico", [12, 24], index=0,
                    format_func=lambda h: f"{h} meses",
                )

            with c3:
                nombre_esc_i = st.text_input("Nombre del escenario", value="Escenario IPP 1", key="nombre_ipp")

            submitted_i = st.form_submit_button("Calcular escenario IPP", type="primary")

        if submitted_i:
            with st.spinner("Calculando IPP..."):
                fut = modelo_i.construir_futuro_drivers(
                    horizonte_i,
                    trm_var_anual=trm_var / 100,
                    brent_var_anual=brent_var / 100,
                    oni=float(oni_i),
                )
                # modo="escenario": reparte el peso entre los componentes que SI responden a
                # los drivers. El pronostico oficial (modo precision) usa pesos calibrados por
                # backtest que dejan al VECM en el piso, y el VECM es el que aporta casi toda
                # la sensibilidad a TRM/Brent. Sin este modo, mover los sliders +-20% cambia el
                # resultado 1.2 puntos en vez de 14.8, y el tab no comunica nada.
                fc_i = modelo_i.pronosticar(horizonte_i, df_futuro=fut, modo="escenario")
                fc_i["fecha"] = pd.to_datetime(fc_i["fecha"])
            st.session_state["_ultimo_fc_ipp"] = fc_i
            st.session_state["_ultimo_params_ipp"] = {
                "trm_var_anual_pct": trm_var, "brent_var_anual_pct": brent_var,
                "oni": oni_i, "horizonte_meses": horizonte_i,
            }

        fc_i_actual = st.session_state.get("_ultimo_fc_ipp")

        if fc_i_actual is not None:
            st.info(
                "**Esto es un ejercicio de sensibilidad, no el pronóstico oficial.** El "
                "escenario reparte el peso entre los componentes que responden a TRM y Brent, "
                "para que los sliders comuniquen una elasticidad. Por eso su nivel base "
                "difiere del pronóstico de la gráfica de arriba, que usa los pesos calibrados "
                "por backtest. Úselo para leer *cuánto se movería*, no *cuánto va a valer*."
            )
            fig_i = go.Figure()

            # Histórico
            if df_hist_ipp is not None:
                fig_i.add_trace(go.Scatter(
                    x=df_hist_ipp["fecha"], y=df_hist_ipp["ipp"],
                    mode="lines", name="Historico", line={"color": "#888", "width": 1.5},
                ))

            # Escenarios guardados (superpuestos)
            colores_i = ["#E05252", "#F4A261", "#A663CC", "#2D6A4F", "#E76F51"]
            for idx_g, esc_g in enumerate(st.session_state.escenarios_ipp):
                c = colores_i[idx_g % len(colores_i)]
                fig_i.add_trace(go.Scatter(
                    x=esc_g["fc"]["fecha"], y=esc_g["fc"]["pred"],
                    mode="lines", name=esc_g["nombre"],
                    line={"color": c, "width": 1.5, "dash": "dot"},
                ))

            # Escenario actual
            fig_i.add_trace(go.Scatter(
                x=pd.concat([fc_i_actual["fecha"], fc_i_actual["fecha"][::-1]]),
                y=pd.concat([fc_i_actual["ci_hi90"], fc_i_actual["ci_lo90"][::-1]]),
                fill="toself", fillcolor="rgba(68,187,164,0.20)",
                line={"color": "rgba(0,0,0,0)"}, showlegend=True, name="IC 90%",
            ))
            fig_i.add_trace(go.Scatter(
                x=fc_i_actual["fecha"], y=fc_i_actual["pred"],
                mode="lines+markers", name="Escenario actual",
                line={"color": "#44BBA4", "width": 2.5}, marker={"size": 6},
            ))
            fig_i.update_layout(
                title="IPP — Escenario interactivo",
                xaxis_title="Fecha", yaxis_title="Indice (base dic-2014=100)",
                hovermode="x unified", height=400,
                legend={"orientation": "h", "y": -0.18},
            )
            st.plotly_chart(fig_i, use_container_width=True)

            col_m1, col_m2, col_m3 = st.columns(3)
            ultimo = fc_i_actual["pred"].iloc[-1]
            primero = fc_i_actual["pred"].iloc[0]
            col_m1.metric("IPP mes 1", f"{primero:.2f}")
            col_m2.metric(f"IPP mes {horizonte_i}", f"{ultimo:.2f}")
            col_m3.metric("Variacion acumulada", f"{(ultimo/primero - 1)*100:+.1f}%")

            # Tabla detallada
            with st.expander("Ver tabla de pronostico"):
                tbl = fc_i_actual[["fecha", "pred", "ci_lo90", "ci_hi90"]].copy()
                tbl.columns = ["Fecha", "Pred. IPP", "IC Lo 90%", "IC Hi 90%"]
                tbl = tbl.set_index("Fecha").round(2)
                st.dataframe(tbl, use_container_width=True)

        # --- Guardar y comparar ---
        st.divider()
        col_g1, col_g2, col_g3 = st.columns([2, 1, 1])

        with col_g1:
            if st.button("Guardar escenario IPP", disabled=fc_i_actual is None, key="btn_guardar_ipp"):
                params_i = st.session_state.get("_ultimo_params_ipp", {})
                st.session_state.escenarios_ipp.append({
                    "nombre": nombre_esc_i,
                    "params": params_i,
                    "fc": fc_i_actual.copy(),
                })
                st.success(f"Escenario IPP '{nombre_esc_i}' guardado.")
                st.rerun()

        with col_g2:
            if st.button("Limpiar comparacion IPP", disabled=not st.session_state.escenarios_ipp, key="btn_limpiar_ipp"):
                st.session_state.escenarios_ipp = []
                st.rerun()

        with col_g3:
            if st.session_state.escenarios_ipp:
                xlsx_i_bytes = _escenarios_a_excel(st.session_state.escenarios_ipp, "ipp")
                st.download_button(
                    "Exportar a Excel",
                    data=xlsx_i_bytes,
                    file_name="escenarios_ipp.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="btn_export_ipp",
                )

        if st.session_state.escenarios_ipp:
            st.subheader("Escenarios IPP guardados")
            rows_i = []
            for esc_g in st.session_state.escenarios_ipp:
                p = esc_g.get("params", {})
                rows_i.append({
                    "Nombre": esc_g["nombre"],
                    "TRM var. anual (%)": p.get("trm_var_anual_pct", ""),
                    "Brent var. anual (%)": p.get("brent_var_anual_pct", ""),
                    "ONI": p.get("oni", ""),
                    "Horizonte (m)": p.get("horizonte_meses", ""),
                    f"IPP ultimo mes": f"{esc_g['fc']['pred'].iloc[-1]:.2f}",
                })
            st.dataframe(pd.DataFrame(rows_i), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 5: Seguimiento de precisión
# ---------------------------------------------------------------------------

with tab_precision:
    resumen = _cargar_resumen()
    errores = resumen.get("errores_ciclo_anterior", [])

    if not errores:
        st.info("Sin errores registrados aun — apareceran a partir del segundo ciclo mensual.")
        st.markdown("""
        **Como funciona el seguimiento:**

        Cada vez que `run_monthly_update.py` se ejecuta:
        1. Compara los pronosticos del mes anterior contra los precios realizados
        2. Calcula MAE, RMSE y sesgo por horizonte
        3. Actualiza los pesos del ensemble para compensar el sesgo reciente
        4. Registra el error en `outputs/runs/{YYYY-MM}/resumen_bolsa.json`
        """)
    else:
        df_err = pd.DataFrame(errores)
        st.subheader("Error del ciclo anterior")
        cols_disp = [c for c in ["horizonte", "n_obs", "mae", "rmse", "sesgo", "mape_pct"] if c in df_err.columns]
        rename_map = {"horizonte": "Horizonte (d)", "n_obs": "N obs", "mae": "MAE",
                      "rmse": "RMSE", "sesgo": "Sesgo", "mape_pct": "MAPE %"}
        st.dataframe(
            df_err[cols_disp].rename(columns=rename_map).round(1),
            use_container_width=True, hide_index=True,
        )

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
        st.subheader("Evolucion de pesos del ensemble")
        df_pesos = pd.DataFrame([
            {"periodo": r["periodo"], "SARIMAX": r.get("w_sarimax", 0), "LGB": r.get("w_lgb", 0)}
            for r in all_resumenes
        ])
        fig_pesos = go.Figure()
        for modelo_lbl in ["SARIMAX", "LGB"]:
            fig_pesos.add_trace(go.Scatter(
                x=df_pesos["periodo"], y=df_pesos[modelo_lbl],
                mode="lines+markers", name=modelo_lbl,
            ))
        fig_pesos.update_layout(
            title="Pesos del ensemble por ciclo mensual",
            yaxis_title="Peso", hovermode="x unified", height=300,
            legend={"orientation": "h", "y": -0.15},
        )
        st.plotly_chart(fig_pesos, use_container_width=True)

    st.subheader("Referencia: Backtest rolling-origin (62 origenes)")
    st.markdown("""
    | Horizonte | RMSE ensemble | RMSE naive | Sesgo |
    |-----------|:---:|:---:|:---:|
    | 1d  | 352 | 271 | +62  |
    | 7d  | 359 | 271 | +107 |
    | 14d | 496 | 321 | +130 |
    | 30d | 844 | 348 | +262 |

    > **Nota:** El sesgo positivo refleja la transicion del regimen El Nino 2023-2024
    > (precio ~800-2000 COP/kWh) al periodo post-Nino 2025-2026 (~465 COP/kWh).
    > El loop mensual lo corrige gradualmente via `actualizar_sesgo()`.
    """)

    # -----------------------------------------------------------------------
    # Torneo de modelos IPP: leaderboard (historico backtest + track record real)
    # -----------------------------------------------------------------------
    from pathlib import Path as _Path

    _OUT_TORNEO = _Path(__file__).resolve().parent.parent / "outputs"

    st.markdown("---")
    st.subheader("Leaderboard de modelos IPP")
    st.caption(
        "Historico = backtest rolling-origin. Track record real = pronosticos publicados "
        "cada mes, puntuados cuando el IPP real llego (n = cuantos ya vencieron). "
        "Skill vs. naive (drift) positivo = le gana a 'no hacer casi nada'."
    )

    _lb_path = _OUT_TORNEO / "torneo" / "leaderboard_ipp.parquet"
    _lb = pd.read_parquet(_lb_path) if _lb_path.exists() else pd.DataFrame()
    if _lb.empty:
        st.info(
            "Aun no hay pronosticos IPP vencidos: el track record real empieza a acumularse "
            "con los proximos ciclos mensuales. Debajo, el historico del backtest."
        )
    else:
        st.markdown("**Skill vs. naive (drift) — track record real** (positivo = le gana)")
        st.dataframe(
            _lb.pivot_table(index="modelo", columns="horizonte", values="skill_vs_naive").round(3),
            use_container_width=True,
        )
        st.markdown("**Evidencia acumulada (n_resueltos)**")
        st.dataframe(
            _lb.pivot_table(index="modelo", columns="horizonte", values="n_resueltos"),
            use_container_width=True,
        )

    _bt_path = _OUT_TORNEO / "backtest" / "metricas_ipp.parquet"
    _bt = pd.read_parquet(_bt_path) if _bt_path.exists() else pd.DataFrame()
    if not _bt.empty:
        st.markdown("**Historico (backtest rolling-origin) — RMSE por horizonte**")
        _vista = _bt.copy()
        if "modo_drivers" in _vista.columns:
            _vista = _vista[_vista["modo_drivers"] == "congelado"]
        if "n_suficiente" in _vista.columns:
            _vista = _vista[_vista["n_suficiente"]]
        st.dataframe(
            _vista.pivot_table(index="modelo", columns="horizonte", values="rmse").round(2),
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # Torneo: evolucion del skill en el tiempo (spec §7)
    # -----------------------------------------------------------------------
    from proybolsa.torneo.evaluacion import skill_acumulado as _skill_acum

    st.markdown("---")
    st.subheader("Evolucion del skill en el tiempo")
    st.caption(
        "Skill acumulado vs. naive (drift) a medida que cada pronostico vence. Positivo = le "
        "gana al naive. Se construye desde outputs/torneo/resueltos_ipp.parquet."
    )
    _res_path = _OUT_TORNEO / "torneo" / "resueltos_ipp.parquet"
    _res = pd.read_parquet(_res_path) if _res_path.exists() else pd.DataFrame()
    if _res.empty or "ipp_bench_drift" not in set(_res.get("modelo", [])):
        st.info(
            "Aun no hay pronosticos IPP vencidos suficientes para la evolucion del skill. "
            "Aparecera a medida que los ciclos mensuales acumulen resultados."
        )
    else:
        _hs = sorted(int(h) for h in _res["horizonte"].unique())
        _h_sel = st.selectbox("Horizonte (meses)", _hs,
                              index=_hs.index(12) if 12 in _hs else 0)
        _sk = _skill_acum(_res, horizonte=_h_sel, naive="ipp_bench_drift")
        if _sk.empty:
            st.info("Sin datos suficientes para ese horizonte todavia.")
        else:
            _figsk = go.Figure()
            for _mod in sorted(_sk["modelo"].unique()):
                _g = _sk[_sk["modelo"] == _mod]
                _figsk.add_trace(go.Scatter(
                    x=_g["fecha_objetivo"], y=_g["skill"],
                    mode="lines+markers", name=_mod,
                ))
            _figsk.add_hline(y=0, line_dash="dash", line_color="gray",
                             annotation_text="naive (drift)")
            _figsk.update_layout(
                title=f"Skill acumulado vs naive — horizonte {_h_sel}m",
                yaxis_title="skill (1 - RMSE/RMSE_naive)", xaxis_title="fecha objetivo",
                hovermode="x unified", height=380,
                legend={"orientation": "h", "y": -0.25},
            )
            st.plotly_chart(_figsk, use_container_width=True)
