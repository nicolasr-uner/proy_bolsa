"""La calibracion de pesos del ensemble de bolsa usa lags de precio CONGELADOS (protocolo de
despliegue), no los reales del futuro, para no premiar al LGB por persistencia con el precio
verdadero (artefacto 98/2)."""
from __future__ import annotations
import numpy as np
from proybolsa.models.bolsa.nivel_diario import EnsembleNivel, COLS_LAG_PRECIO
from tests.test_modelo_bolsa import _df_diario_sintetico


def test_calibracion_bolsa_invariante_a_lags_futuros_reales():
    df = _df_diario_sintetico(n=500)
    n_val = 60
    tr, val = df.iloc[:-n_val].reset_index(drop=True), df.iloc[-n_val:].reset_index(drop=True)

    m1 = EnsembleNivel().fit(tr.copy(), df_val=val.copy())
    # corromper los lags de precio REALES del futuro en val
    val2 = val.copy()
    for c in COLS_LAG_PRECIO:
        if c in val2.columns:
            val2[c] = val2[c] * 3.0
    m2 = EnsembleNivel().fit(tr.copy(), df_val=val2)

    # Los pesos deben ser IDENTICOS: la calibracion no mira los lags reales del futuro.
    assert abs(m1.w_lgb - m2.w_lgb) < 1e-9, (
        f"la calibracion filtro los lags reales del futuro: w_lgb {m1.w_lgb} vs {m2.w_lgb}")
