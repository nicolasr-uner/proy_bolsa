"""Conectores de ingesta de datos.

XM (mercado electrico colombiano):
    fetch_precio_bolsa_horario, fetch_precio_escasez,
    fetch_aportes_diarios, fetch_embalses_diarios, fetch_vertimientos_diarios,
    fetch_demanda_diaria, fetch_demanda_horaria, fetch_demanda_upme,
    fetch_metric_range, horas_anchas_a_largo

Macro (TRM, Brent, PPI, ONI, IPP):
    fetch_trm, fetch_brent, fetch_ppi_usa, fetch_oni, fetch_ipp
"""

from proybolsa.ingest.macros import (
    fetch_brent,
    fetch_ipp,
    fetch_oni,
    fetch_ppi_usa,
    fetch_trm,
)
from proybolsa.ingest.xm import (
    fetch_aportes_diarios,
    fetch_demanda_diaria,
    fetch_demanda_horaria,
    fetch_demanda_upme,
    fetch_embalses_diarios,
    fetch_metric_range,
    fetch_precio_bolsa_horario,
    fetch_precio_escasez,
    fetch_vertimientos_diarios,
    horas_anchas_a_largo,
)

__all__ = [
    # XM
    "fetch_precio_bolsa_horario",
    "fetch_precio_escasez",
    "fetch_aportes_diarios",
    "fetch_embalses_diarios",
    "fetch_vertimientos_diarios",
    "fetch_demanda_diaria",
    "fetch_demanda_horaria",
    "fetch_demanda_upme",
    "fetch_metric_range",
    "horas_anchas_a_largo",
    # Macro
    "fetch_trm",
    "fetch_brent",
    "fetch_ppi_usa",
    "fetch_oni",
    "fetch_ipp",
]
