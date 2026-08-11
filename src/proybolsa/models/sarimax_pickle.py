"""Protocolo de pickle compacto para los wrappers de `statsmodels.SARIMAX`.

Un objeto `SARIMAXResults` arrastra las matrices del filtro de Kalman de toda la muestra.
En el modelo de bolsa eso son ~21 MB de los 22.8 MB que pesa `outputs/models/bolsa.pkl`;
los parámetros estimados pesan ~50 KB. Reconstruir el resultado con `.filter(params)` al
deserializar produce forecasts y bandas **idénticas** y cuesta ~0.15 s.

Por qué importa: los `.pkl` se versionan en git para que Streamlit Community Cloud no tenga
que ajustar los modelos (el `.fit()` de SARIMAX tumba el CPU del tier gratuito). Sin este
protocolo, cada ciclo mensual añade 22 MB de blob binario al historial (~290 MB/año) y
automatizar el commit desde CI es inviable.

Contrato para la clase que lo use:
  - `fit()` debe dejar `self._endog`, `self._exog` (o None), `self._order`,
    `self._seasonal_order` y `self._result`.
  - Los pickles antiguos (que traían el `SARIMAXResults` completo) siguen cargando: si el
    estado incluye `_result`, se respeta tal cual. Es lo que evita tumbar la app pública
    durante el despliegue en que coexisten el `.pkl` viejo y el código nuevo.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

# Se sube cuando cambia la forma del estado serializado.
VERSION_PICKLE = 2

# Los tres wrappers ajustan con estos kwargs; deben coincidir en la reconstrucción o los
# parámetros no corresponden al mismo modelo.
_KWARGS_SARIMAX = {"enforce_stationarity": False, "enforce_invertibility": False}


class PickleCompactoSARIMAX:
    """Mixin que serializa `(endog, exog, order, params)` en vez de `SARIMAXResults`."""

    def __getstate__(self) -> dict:
        estado = {k: v for k, v in self.__dict__.items() if k != "_result"}
        estado["_version_pickle"] = VERSION_PICKLE
        res = self.__dict__.get("_result")
        estado["_params"] = np.asarray(res.params) if res is not None else None
        return estado

    def __setstate__(self, estado: dict) -> None:
        estado = dict(estado)
        params = estado.pop("_params", None)
        estado.pop("_version_pickle", None)
        legacy = estado.pop("_result", None)
        self.__dict__.update(estado)

        if legacy is not None:
            # Pickle v1: traía el SARIMAXResults completo. Se respeta para que un .pkl
            # anterior siga funcionando con este código.
            self._result = legacy
            return

        self._result = None
        if params is None or self.__dict__.get("_endog") is None:
            return
        try:
            self._result = self._reconstruir_result(params)
        except Exception as exc:  # pragma: no cover - depende de la version de statsmodels
            logger.error(
                "%s: no se pudo reconstruir el SARIMAXResults desde los parámetros (%s). "
                "El modelo quedó sin ajustar; re-ejecute scripts/serializar_modelos.py",
                type(self).__name__, exc,
            )

    def _reconstruir_result(self, params):
        """Reconstruye el resultado ajustado sin re-optimizar (solo corre el filtro)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                self._endog,
                exog=self.__dict__.get("_exog"),
                order=self._order,
                seasonal_order=self._seasonal_order,
                **_KWARGS_SARIMAX,
            )
            return model.filter(params)
