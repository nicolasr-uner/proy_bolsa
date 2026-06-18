"""Verifica que los escenarios del IPP tengan la dirección económica correcta.

Prueba: TRM +20% Y Brent +20% ambos deben resultar en IPP proyectado mayor
que el escenario base (sin cambios). Esto confirma que brent_cop funciona.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from proybolsa.models.ipp.modelo_ipp import PronosticadorIPP

df = pd.read_parquet("data/processed/ipp_features_mensual.parquet")
df = df.dropna(subset=["ipp"])

p = PronosticadorIPP()
p.fit(df)

HORIZONTE = 12

fc_base = p.pronosticar(HORIZONTE)
fut_alto = p.construir_futuro_drivers(HORIZONTE, trm_var_anual=0.20, brent_var_anual=0.20)
fc_alto  = p.pronosticar(HORIZONTE, df_futuro=fut_alto)
fut_bajo = p.construir_futuro_drivers(HORIZONTE, trm_var_anual=-0.10, brent_var_anual=-0.15)
fc_bajo  = p.pronosticar(HORIZONTE, df_futuro=fut_bajo)

m_base = fc_base["pred"].mean()
m_alto = fc_alto["pred"].mean()
m_bajo = fc_bajo["pred"].mean()

print(f"IPP base (sin cambio):          {m_base:.2f}")
print(f"IPP alto (TRM+20%, Brent+20%):  {m_alto:.2f}")
print(f"IPP bajo (TRM-10%, Brent-15%):  {m_bajo:.2f}")
print()
ok_alto = m_alto > m_base
ok_bajo = m_bajo < m_base
print(f"TRM+20%/Brent+20% => IPP sube: {'✓ CORRECTO' if ok_alto else '✗ INCORRECTO'}")
print(f"TRM-10%/Brent-15% => IPP baja: {'✓ CORRECTO' if ok_bajo else '✗ INCORRECTO'}")
print()
print(f"Resumen: {p.resumen_modelo()}")
