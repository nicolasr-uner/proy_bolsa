"""Registro de las series crudas: un único lugar donde vive la verdad de cada fuente.

Antes, la información de cada serie estaba repartida entre `scripts/descarga_historico.py`
(qué fetcher, qué ruta), `scripts/construir_features.py` (qué columnas) y `.gitignore` (qué se
versiona), sin nada que las mantuviera de acuerdo.

Lo que este módulo añade y no existía: **la ventana de revisión**. La ingesta anterior era
"existe el archivo -> no lo toques" (`_ya_existe`), lo que en la práctica congeló los datos
dos meses. Pero descargar solo lo posterior a la última fecha tampoco alcanza, porque varias
fuentes **revisan datos ya publicados**:

  - XM liquida el mercado en varias pasadas (TX2/TX3), así que días viejos cambian de valor.
  - FRED rellena festivos y el BLS revisa el PPI dos meses hacia atrás.
  - Socrata publica la TRM por rangos de vigencia, y el último rango se extiende.
  - NOAA recalcula el ONI sobre la base ERSSTv5.

Por eso la regla es `(última_fecha − ventana_revisión, ayer)`: se re-baja una cola y se deja
que el upsert sobreescriba lo que cambió, en vez de asumir que el pasado es inmutable.

Los fetchers se referencian por nombre y se importan de forma tardía: `proybolsa.ingest.xm`
arrastra `pydataxm`, que a su vez descarga el catálogo de 193 métricas al instanciarse.
Importar este registro no debe costar eso.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable, Literal

# Arranque por defecto cuando un parquet no existe. 3 años cubre el mínimo de entrenamiento
# de bolsa (730 días) con margen.
INICIO_DEFECTO = dt.date(2023, 1, 1)


@dataclass(frozen=True)
class SerieSpec:
    """Cómo se descarga, dónde vive y cuánta cola se re-baja de una serie."""
    nombre: str
    ruta_rel: str                      # relativa a data/raw/
    col_tiempo: str                    # 'timestamp' (horaria) o 'fecha'
    freq: Literal["H", "D", "M"]
    ventana_revision_dias: int         # cola que SIEMPRE se re-descarga y sobreescribe
    tolerancia_frescura_dias: int      # más atraso que esto = falla el gate de frescura
    fetcher_ref: tuple[str, str]       # (módulo, función); import tardío
    inicio_default: dt.date = INICIO_DEFECTO
    kwargs_fetcher: dict = field(default_factory=dict)
    # Series que se re-publican completas (proyecciones, índices retropolados): no tiene
    # sentido pedir un rango, se reemplaza todo.
    reemplazo_total: bool = False

    def fetcher(self) -> Callable:
        modulo, funcion = self.fetcher_ref
        return getattr(import_module(modulo), funcion)


_XM = "proybolsa.ingest.xm"
_MACRO = "proybolsa.ingest.macros"

SERIES: dict[str, SerieSpec] = {
    # --- XM ---------------------------------------------------------------------------
    # 15 días de cola en las diarias/horarias: cubre la re-liquidación semanal de XM.
    "precio_bolsa_horario": SerieSpec(
        nombre="precio_bolsa_horario", ruta_rel="xm/precio_bolsa_horario.parquet",
        col_tiempo="timestamp", freq="H", ventana_revision_dias=15,
        tolerancia_frescura_dias=5, fetcher_ref=(_XM, "fetch_precio_bolsa_horario"),
    ),
    "precio_escasez": SerieSpec(
        nombre="precio_escasez", ruta_rel="xm/precio_escasez.parquet",
        col_tiempo="fecha", freq="D", ventana_revision_dias=45,
        tolerancia_frescura_dias=40, fetcher_ref=(_XM, "fetch_precio_escasez"),
    ),
    "aportes_diarios": SerieSpec(
        nombre="aportes_diarios", ruta_rel="xm/aportes_diarios.parquet",
        col_tiempo="fecha", freq="D", ventana_revision_dias=15,
        tolerancia_frescura_dias=5, fetcher_ref=(_XM, "fetch_aportes_diarios"),
    ),
    "embalses_diarios": SerieSpec(
        nombre="embalses_diarios", ruta_rel="xm/embalses_diarios.parquet",
        col_tiempo="fecha", freq="D", ventana_revision_dias=15,
        tolerancia_frescura_dias=5, fetcher_ref=(_XM, "fetch_embalses_diarios"),
    ),
    "vertimientos_diarios": SerieSpec(
        nombre="vertimientos_diarios", ruta_rel="xm/vertimientos_diarios.parquet",
        col_tiempo="fecha", freq="D", ventana_revision_dias=15,
        tolerancia_frescura_dias=8, fetcher_ref=(_XM, "fetch_vertimientos_diarios"),
    ),
    "demanda_diaria": SerieSpec(
        nombre="demanda_diaria", ruta_rel="xm/demanda_diaria.parquet",
        col_tiempo="fecha", freq="D", ventana_revision_dias=15,
        tolerancia_frescura_dias=5, fetcher_ref=(_XM, "fetch_demanda_diaria"),
    ),
    # Proyección de demanda de la UPME: se re-publica completa, no es una serie observada.
    "demanda_upme_medio": SerieSpec(
        nombre="demanda_upme_medio", ruta_rel="xm/demanda_upme_medio.parquet",
        col_tiempo="fecha", freq="M", ventana_revision_dias=0,
        tolerancia_frescura_dias=400, fetcher_ref=(_XM, "fetch_demanda_upme"),
        kwargs_fetcher={"scenario": "Medio"}, reemplazo_total=True,
    ),

    # --- Macro ------------------------------------------------------------------------
    # TRM: Socrata publica por rangos de vigencia y el último se extiende -> 5 días de cola.
    "trm_diaria": SerieSpec(
        nombre="trm_diaria", ruta_rel="macro/trm_diaria.parquet",
        col_tiempo="fecha", freq="D", ventana_revision_dias=5,
        tolerancia_frescura_dias=5, fetcher_ref=(_MACRO, "fetch_trm"),
    ),
    # Brent: FRED rellena festivos a posteriori.
    "brent_diario": SerieSpec(
        nombre="brent_diario", ruta_rel="macro/brent_diario.parquet",
        col_tiempo="fecha", freq="D", ventana_revision_dias=10,
        tolerancia_frescura_dias=8, fetcher_ref=(_MACRO, "fetch_brent"),
    ),
    # PPI USA: el BLS revisa ~2 meses hacia atrás (medido: 4 meses cambiaron de valor en la
    # primera corrida incremental). La tolerancia es de 100 días, no 70: el BLS publica el mes
    # M a mediados de M+1, así que un rezago de ~70 días es el estado NORMAL y una tolerancia
    # más ajustada dispara falsas alarmas todos los meses.
    "ppi_usa_mensual": SerieSpec(
        nombre="ppi_usa_mensual", ruta_rel="macro/ppi_usa_mensual.parquet",
        col_tiempo="fecha", freq="M", ventana_revision_dias=95,
        tolerancia_frescura_dias=100, fetcher_ref=(_MACRO, "fetch_ppi_usa"),
    ),
    # ONI: el CPC recalcula la serie sobre ERSSTv5 -> cola larga.
    "oni_mensual": SerieSpec(
        nombre="oni_mensual", ruta_rel="macro/oni_mensual.parquet",
        col_tiempo="fecha", freq="M", ventana_revision_dias=130,
        tolerancia_frescura_dias=95, fetcher_ref=(_MACRO, "fetch_oni"),
    ),
}

# El IPP no está aquí: su adquisición no es "un rango de fechas" sino "el anexo publicado más
# reciente", que trae la historia completa. Vive en `ingest/dane_ipp.py` y lo orquesta
# `scripts/actualizar_ipp.py`.

GRUPOS: dict[str, tuple[str, ...]] = {
    "xm": tuple(n for n, s in SERIES.items() if s.ruta_rel.startswith("xm/")),
    "macro": tuple(n for n, s in SERIES.items() if s.ruta_rel.startswith("macro/")),
}


def resolver(nombres: str | None = None) -> list[SerieSpec]:
    """Resuelve una selección tipo 'xm', 'macro' o 'a,b,c' a specs concretas."""
    if not nombres:
        return list(SERIES.values())
    pedidos: list[str] = []
    for token in nombres.split(","):
        token = token.strip()
        if not token:
            continue
        if token in GRUPOS:
            pedidos.extend(GRUPOS[token])
        elif token in SERIES:
            pedidos.append(token)
        else:
            raise KeyError(
                f"serie desconocida: {token!r}. Opciones: {sorted(SERIES)} o grupos {sorted(GRUPOS)}"
            )
    vistos: set[str] = set()
    return [SERIES[n] for n in pedidos if not (n in vistos or vistos.add(n))]
