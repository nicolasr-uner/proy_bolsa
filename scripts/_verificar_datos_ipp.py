"""Verificacion de integridad de la serie IPP Oferta Interna."""
import pandas as pd
import numpy as np

# === 1. Serie cruda ipp_manual.csv ===
csv = pd.read_csv("data/raw/macro/ipp_manual.csv", parse_dates=["fecha"])
print("=== ipp_manual.csv ===")
print(f"  Filas: {len(csv)}")
print(f"  Rango: {csv.fecha.min().date()} a {csv.fecha.max().date()}")
dic14 = csv[csv.fecha == "2014-12-01"]["ipp"]
val_dic14 = float(dic14.values[0]) if len(dic14) else None
print(f"  dic-2014 = {val_dic14}")
ok_base = val_dic14 is not None and abs(val_dic14 - 100.0) < 0.05
print(f"  Base dic-2014=100: {'OK' if ok_base else 'FALLA'}")
print("  Ultimos 6 meses:")
print(csv.tail(6)[["fecha", "ipp"]].to_string(index=False))

# === 2. Saltos bruscos (>15 puntos en un mes) ===
jumps = csv["ipp"].diff().abs()
bad = csv[jumps > 15]
print(f"\n  Saltos > 15 puntos: {len(bad)}")
if len(bad):
    print(bad[["fecha", "ipp"]].to_string(index=False))

# === 3. Parquet de features ===
df = pd.read_parquet("data/processed/ipp_features_mensual.parquet")
df_nona = df.dropna(subset=["ipp"])
print("\n=== ipp_features_mensual.parquet ===")
print(f"  Filas: {len(df_nona)}  Columnas: {len(df.columns)}")
new_cols = ["brent_cop", "brent_cop_lag1m", "trm_yoy", "brent_yoy"]
for c in new_cols:
    print(f"  {c}: {'presente' if c in df.columns else 'FALTA'}")
print(f"  ppi_usa_lag1m (eliminado): {'aun presente' if 'ppi_usa_lag1m' in df.columns else 'correctamente ausente'}")
ult_fecha = df_nona["fecha"].iloc[-1]
ult_ipp = float(df_nona["ipp"].iloc[-1])
print(f"  Ultimo observado: {ult_fecha} -> ipp={ult_ipp:.2f}")

# === 4. Coherencia de escenarios ===
import warnings; warnings.filterwarnings("ignore")
from proybolsa.models.ipp.modelo_ipp import PronosticadorIPP

pi = PronosticadorIPP()
pi.fit(df_nona)

fc_base = pi.pronosticar(12)
fut_alto = pi.construir_futuro_drivers(12, trm_var_anual=0.20, brent_var_anual=0.20)
fc_alto  = pi.pronosticar(12, df_futuro=fut_alto)
fut_bajo = pi.construir_futuro_drivers(12, trm_var_anual=-0.10, brent_var_anual=-0.15)
fc_bajo  = pi.pronosticar(12, df_futuro=fut_bajo)

m_base = fc_base["pred"].mean()
m_alto = fc_alto["pred"].mean()
m_bajo = fc_bajo["pred"].mean()
ok_dir = m_alto > m_base > m_bajo

print("\n=== Coherencia de escenarios IPP ===")
print(f"  Base:  {m_base:.2f}")
print(f"  Alto (TRM+20%, Brent+20%): {m_alto:.2f}")
print(f"  Bajo  (TRM-10%, Brent-15%): {m_bajo:.2f}")
print(f"  Bajo < Base < Alto: {'OK' if ok_dir else 'FALLA'}")

# === 5. Bolsa: seco > promedio > humedo ===
from proybolsa.models.bolsa.pronostico import PronosticadorBolsa
df_b = pd.read_parquet("data/processed/bolsa_features_diario.parquet")
pb = PronosticadorBolsa()
pb.fit(df_b, ruta_perfil="data/processed/perfil_horario.parquet")
escs = pb.pronosticar(30, escenarios_multiples=True, devolver_horario=False)
m_s = escs["seco"]["pred_diaria"].mean()
m_p = escs["promedio"]["pred_diaria"].mean()
m_h = escs["humedo"]["pred_diaria"].mean()
ok_bolsa = m_s >= m_p >= m_h
print("\n=== Coherencia de escenarios Bolsa ===")
print(f"  Seco: {m_s:.1f}  Promedio: {m_p:.1f}  Humedo: {m_h:.1f}")
print(f"  Seco >= Promedio >= Humedo: {'OK' if ok_bolsa else 'FALLA'}")

# === Resumen ===
all_ok = ok_base and ok_dir and ok_bolsa
print(f"\n=== RESULTADO FINAL: {'PASS' if all_ok else 'FAIL'} ===")
