"""Feature engineering para los modelos de bolsa e IPP.

Calendar:
    tipo_dia, agregar_features_calendario, agregar_features_fecha

Hydrology:
    escalar_fracciones, clasificar_regimen_hidro, agregar_ratio_vertimientos

ENSO:
    clasificar_enso, agregar_features_enso, dummy_enso

Lags:
    agregar_lags_diarios, agregar_lags_horarios, agregar_rolling, agregar_lags_mensuales

Hourly profile:
    estimar_perfil_horario, aplicar_perfil_horario, guardar_perfil, cargar_perfil
"""

from proybolsa.features.calendar import (
    agregar_features_calendario,
    agregar_features_fecha,
    festivos_colombia,
    tipo_dia,
)
from proybolsa.features.enso import (
    agregar_features_enso,
    clasificar_enso,
    dummy_enso,
)
from proybolsa.features.hourly_profile import (
    aplicar_perfil_horario,
    cargar_perfil,
    estimar_perfil_horario,
    guardar_perfil,
)
from proybolsa.features.hydro import (
    agregar_ratio_vertimientos,
    clasificar_regimen_hidro,
    escalar_fracciones,
)
from proybolsa.features.lags import (
    agregar_lags_diarios,
    agregar_lags_horarios,
    agregar_lags_mensuales,
    agregar_rolling,
)

__all__ = [
    # Calendar
    "tipo_dia", "festivos_colombia",
    "agregar_features_calendario", "agregar_features_fecha",
    # Hydro
    "escalar_fracciones", "clasificar_regimen_hidro", "agregar_ratio_vertimientos",
    # ENSO
    "clasificar_enso", "agregar_features_enso", "dummy_enso",
    # Lags
    "agregar_lags_diarios", "agregar_lags_horarios",
    "agregar_rolling", "agregar_lags_mensuales",
    # Hourly profile
    "estimar_perfil_horario", "aplicar_perfil_horario",
    "guardar_perfil", "cargar_perfil",
]
