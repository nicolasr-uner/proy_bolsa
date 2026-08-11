"""Capa HTTP compartida para la ingesta: timeouts, reintentos y una sola sesión.

Antes de este módulo, cada fetcher hacía su propio `requests.get`, con timeouts desiguales
y sin reintentos:

  - `_fred_serie` usaba `pd.read_csv(url)`, es decir urllib **sin timeout**. Un FRED colgado
    bloqueaba el ciclo mensual indefinidamente.
  - `pydataxm` hace `requests.post` sin timeout: lo mismo por el lado de XM.
  - TRM y ONI tenían `timeout=30`, BanRep `timeout=15`: tres criterios distintos.
  - Ningún fetcher reintentaba, así que un 503 transitorio de una fuente abortaba la corrida.
  - Sin `Session`, cada llamada renegociaba TLS.

Los timeouts son tuplas `(connect, read)`: separar los dos importa porque un servidor que
acepta la conexión y luego se queda callado es el caso que cuelga, y es distinto de un host
inalcanzable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# (connect, read). El read es generoso porque algunas fuentes (Socrata paginado, anexos del
# DANE) tardan en responder el primer byte.
TIMEOUT_DEFECTO: tuple[float, float] = (10.0, 60.0)
# Para descargas de archivo: el read cubre la transferencia completa.
TIMEOUT_DESCARGA: tuple[float, float] = (10.0, 180.0)

USER_AGENT = "proybolsa/0.1 (ingesta de series publicas; +https://github.com/nicolasr-uner/proy_bolsa)"

# 429 y 5xx son transitorios y vale reintentar. Un 404 NO está aquí a propósito: en la
# resolución del anexo del DANE un 404 es información (ese mes no se publicó), no un fallo.
_STATUS_REINTENTABLES = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class CabecerasRemotas:
    """Resultado de un HEAD: lo mínimo para decidir si vale la pena descargar."""
    status: int
    content_length: int | None
    last_modified: str | None
    content_type: str | None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@lru_cache(maxsize=1)
def sesion() -> requests.Session:
    """Sesión única con reintentos y backoff exponencial.

    `backoff_factor=1.5` da esperas de ~0, 3, 6, 12 s. Con `total=4` el peor caso son ~21 s
    por request, aceptable para un ciclo que corre una vez al mes o una vez a la semana.
    """
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    reintentos = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.5,
        status_forcelist=_STATUS_REINTENTABLES,
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=reintentos, pool_connections=4, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def head(url: str, *, timeout: tuple[float, float] = TIMEOUT_DEFECTO,
         permitir_redirects: bool = True) -> CabecerasRemotas:
    """HEAD sin lanzar por status: el llamador decide qué significa cada código.

    Cuidado al usarlo: un 404 del DANE devuelve igualmente `Last-Modified` (la fecha de la
    página de error), así que la existencia del recurso se decide por `status`, nunca por la
    presencia de esa cabecera.
    """
    r = sesion().head(url, timeout=timeout, allow_redirects=permitir_redirects)
    cl = r.headers.get("Content-Length")
    return CabecerasRemotas(
        status=r.status_code,
        content_length=int(cl) if cl and cl.isdigit() else None,
        last_modified=r.headers.get("Last-Modified"),
        content_type=r.headers.get("Content-Type"),
    )


def get_text(url: str, *, params: dict | None = None,
             timeout: tuple[float, float] = TIMEOUT_DEFECTO) -> str:
    r = sesion().get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.text


def get_json(url: str, *, params: dict | None = None, headers: dict | None = None,
             timeout: tuple[float, float] = TIMEOUT_DEFECTO) -> Any:
    r = sesion().get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_bytes(url: str, *, timeout: tuple[float, float] = TIMEOUT_DESCARGA) -> bytes:
    r = sesion().get(url, timeout=timeout)
    r.raise_for_status()
    contenido = r.content
    logger.debug("GET %s -> %d bytes", url, len(contenido))
    return contenido
