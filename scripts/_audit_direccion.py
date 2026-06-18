"""Verificacion de direccion de escenarios para la auditoria."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd

# === Bolsa: seco > promedio > humedo ===
df_b = pd.read_parquet("data/processed/bolsa_features_diario.parquet")
from proybolsa.models.bolsa.pronostico import PronosticadorBolsa
p = PronosticadorBolsa()
p.fit(df_b, ruta_perfil="data/processed/perfil_horario.parquet")
esc = p.pronosticar(30, escenarios_multiples=True, devolver_horario=False)
m_s = esc["seco"]["pred_diaria"].mean()
m_p = esc["promedio"]["pred_diaria"].mean()
m_h = esc["humedo"]["pred_diaria"].mean()
ok_bolsa = m_s >= m_p >= m_h
print("=== Bolsa escenarios hidrologicos ===")
print(f"  Seco: {m_s:.1f}  Promedio: {m_p:.1f}  Humedo: {m_h:.1f}")
print(f"  Seco >= Promedio >= Humedo: {ok_bolsa}")
print()

# Aportes bajos => precio mayor
fc_bajo = p.pronosticar(14, aportes_pct=50.0, volumen_util_pct=25.0, devolver_horario=False)
fc_alto = p.pronosticar(14, aportes_pct=140.0, volumen_util_pct=80.0, devolver_horario=False)
bajo_m = fc_bajo["pred_diaria"].mean()
alto_m = fc_alto["pred_diaria"].mean()
ok_override = bajo_m > alto_m
print("=== Bolsa override hidrologia ===")
print(f"  Aportes 50%: {bajo_m:.1f}  Aportes 140%: {alto_m:.1f}")
print(f"  Bajo aportes => precio mayor: {ok_override}")
print()

# === IPP: TRM+20%+Brent+20% => IPP > base > TRM-10%+Brent-15% ===
df_i = pd.read_parquet("data/processed/ipp_features_mensual.parquet").dropna(subset=["ipp"])
from proybolsa.models.ipp.modelo_ipp import PronosticadorIPP
pi = PronosticadorIPP()
pi.fit(df_i)
fc_base = pi.pronosticar(12)
fut_a = pi.construir_futuro_drivers(12, trm_var_anual=0.20, brent_var_anual=0.20)
fc_alto_ipp = pi.pronosticar(12, df_futuro=fut_a)
fut_b = pi.construir_futuro_drivers(12, trm_var_anual=-0.10, brent_var_anual=-0.15)
fc_bajo_ipp = pi.pronosticar(12, df_futuro=fut_b)
base_m = fc_base["pred"].mean()
alto_ipp = fc_alto_ipp["pred"].mean()
bajo_ipp = fc_bajo_ipp["pred"].mean()
ok_ipp = alto_ipp > base_m > bajo_ipp
print("=== IPP escenarios ===")
print(f"  Bajo: {bajo_ipp:.2f}  Base: {base_m:.2f}  Alto: {alto_ipp:.2f}")
print(f"  Bajo < Base < Alto: {ok_ipp}")
print()

# === Excel export (verificar que openpyxl funciona) ===
import io, openpyxl
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    fc_base[["fecha","pred","ci_lo90","ci_hi90"]].to_excel(w, sheet_name="IPP_base", index=False)
n_bytes = len(buf.getvalue())
ok_excel = n_bytes > 5000
print(f"=== Excel export ===")
print(f"  Bytes generados: {n_bytes}  OK: {ok_excel}")
print()

all_ok = ok_bolsa and ok_override and ok_ipp and ok_excel
print(f"=== RESULTADO AUDITORIA ===  {'PASS' if all_ok else 'FAIL'}")
