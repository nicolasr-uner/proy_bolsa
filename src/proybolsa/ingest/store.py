"""Persistencia incremental de las series crudas: append + dedup + escritura atómica.

Lo que reemplaza: `scripts/descarga_historico.py` tenía `_guardar_parquet` (un
`to_parquet` que reescribía el archivo entero) y `_ya_existe` (si el archivo existe, no
hagas nada). La combinación no dejaba opción intermedia: o se re-bajaban 3 años completos
con `--forzar`, o no se bajaba nada. Como el ciclo mensual invocaba el script **sin**
`--forzar`, el paso de descarga era un no-op que registraba "OK" mientras los datos
envejecían.

Tres decisiones que vale explicar:

1. **`keep="last"` en el dedup**: cuando una fecha viene en el archivo viejo y en la descarga
   nueva, gana la nueva. Es lo que permite que una revisión de XM o del BLS entre sin lógica
   extra.
2. **Escritura atómica** (`.tmp` + `os.replace`): un proceso muerto a mitad de un
   `to_parquet` deja un parquet truncado. En CI eso se commitea y el repo queda roto.
3. **Se reporta lo revisado, no solo lo nuevo**: si un valor histórico cambia, hay que poder
   verlo. Un dato que se mueve en silencio es indistinguible de un bug de ingesta.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from proybolsa.ingest.series import SERIES, SerieSpec

logger = logging.getLogger(__name__)

Modo = Literal["incremental", "completo"]


@dataclass
class ResultadoUpsert:
    serie: str
    filas_antes: int
    filas_despues: int
    filas_nuevas: int
    filas_revisadas: int
    rango_antes: tuple | None
    rango_despues: tuple | None
    revisiones: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def avanzo(self) -> bool:
        return self.filas_nuevas > 0 or self.filas_revisadas > 0


def _normalizar_tiempo(s: pd.Series) -> pd.Series:
    """A datetime64 para poder comparar y ordenar sin importar cómo vino del parquet.

    Los parquets del repo son inconsistentes a propósito de nadie: las series horarias traen
    `timestamp` como datetime64 y las diarias `fecha` como object con `datetime.date`.
    """
    return pd.to_datetime(s, errors="coerce")


def _restaurar_dtype_tiempo(df: pd.DataFrame, col: str, como_date: bool) -> pd.DataFrame:
    """Devuelve la columna de tiempo al dtype que tenía el parquet original.

    Importa: `construir_features.py` y el resto del pipeline ya conviven con esa mezcla, así
    que cambiarla aquí rompería consumidores por un motivo cosmético.
    """
    if como_date:
        df[col] = pd.to_datetime(df[col]).dt.date
    return df


def _es_date_puro(s: pd.Series) -> bool:
    """¿La columna guarda `datetime.date` en vez de `datetime64`/`Timestamp`?

    `pd.Timestamp` hereda de `datetime.datetime`, que a su vez hereda de `datetime.date`, así
    que hay que excluir explícitamente el caso datetime o todo parece `date`.
    """
    muestra = s.dropna()
    if muestra.empty:
        return False
    v = muestra.iloc[0]
    return isinstance(v, dt.date) and not isinstance(v, dt.datetime)


def ruta_de(spec: SerieSpec, raiz: Path) -> Path:
    return raiz / spec.ruta_rel


def ultima_fecha(spec: SerieSpec, raiz: Path) -> dt.date | None:
    """Última fecha presente en el parquet, o None si no existe/está vacío."""
    ruta = ruta_de(spec, raiz)
    if not ruta.exists():
        return None
    try:
        df = pd.read_parquet(ruta, columns=[spec.col_tiempo])
    except Exception as exc:
        logger.warning("%s: no se pudo leer %s (%s)", spec.nombre, ruta.name, exc)
        return None
    if df.empty:
        return None
    return _normalizar_tiempo(df[spec.col_tiempo]).max().date()


def rango_a_descargar(spec: SerieSpec, raiz: Path, hoy: dt.date | None = None, *,
                      modo: Modo = "incremental",
                      resync_dias: int = 0) -> tuple[dt.date, dt.date] | None:
    """Rango a pedirle a la fuente. None si no hay nada por hacer.

    `resync_dias` amplía la cola por encima de la ventana de revisión de la serie, para
    recuperar un atraso puntual sin re-bajar la historia completa.
    """
    hoy = hoy or dt.date.today()
    ayer = hoy - dt.timedelta(days=1)

    if modo == "completo" or spec.reemplazo_total:
        return (spec.inicio_default, ayer)

    ultima = ultima_fecha(spec, raiz)
    if ultima is None:
        return (spec.inicio_default, ayer)

    cola = spec.ventana_revision_dias + max(0, resync_dias)
    desde = ultima - dt.timedelta(days=cola)
    if desde > ayer:
        return None
    return (max(desde, spec.inicio_default), ayer)


def upsert_parquet(df_nuevo: pd.DataFrame, spec: SerieSpec, raiz: Path) -> ResultadoUpsert:
    """Fusiona `df_nuevo` con el parquet existente y lo reescribe de forma atómica."""
    ruta = ruta_de(spec, raiz)
    col = spec.col_tiempo

    if col not in df_nuevo.columns:
        raise ValueError(
            f"{spec.nombre}: la descarga no trae la columna de tiempo {col!r}. "
            f"Columnas: {list(df_nuevo.columns)}"
        )

    nuevo = df_nuevo.copy()
    previo = None

    # El dtype de la columna de tiempo lo manda el parquet existente; si no hay archivo, lo
    # manda el fetcher. Nunca este módulo: cambiar la convención en la primera escritura la
    # cambiaría para siempre, porque las siguientes la leen del archivo ya escrito.
    como_date = _es_date_puro(nuevo[col])

    if ruta.exists():
        previo = pd.read_parquet(ruta)
        como_date = _es_date_puro(previo[col])
        previo[col] = _normalizar_tiempo(previo[col])

    nuevo[col] = _normalizar_tiempo(nuevo[col])
    nuevo = nuevo.dropna(subset=[col])

    filas_antes = 0 if previo is None else len(previo)
    rango_antes = None
    if previo is not None and not previo.empty:
        rango_antes = (previo[col].min().date(), previo[col].max().date())

    # --- Contar nuevas y revisadas antes de fusionar ---
    filas_nuevas = len(nuevo)
    filas_revisadas = 0
    revisiones = pd.DataFrame()

    if previo is not None and not previo.empty:
        claves_previas = set(previo[col])
        es_nueva = ~nuevo[col].isin(claves_previas)
        filas_nuevas = int(es_nueva.sum())

        solapadas = nuevo[~es_nueva]
        if not solapadas.empty:
            cols_valor = [c for c in nuevo.columns if c != col and c in previo.columns]
            if cols_valor:
                comp = (previo[[col] + cols_valor]
                        .merge(solapadas[[col] + cols_valor], on=col, suffixes=("_ant", "_new")))
                difiere = pd.Series(False, index=comp.index)
                for c in cols_valor:
                    a, b = comp[f"{c}_ant"], comp[f"{c}_new"]
                    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                        # Tolerancia relativa: el redondeo de ida y vuelta a parquet no es
                        # una revisión.
                        d = (a - b).abs() > (1e-9 + 1e-6 * b.abs())
                    else:
                        d = a.astype(str) != b.astype(str)
                    difiere |= d.fillna(False)
                filas_revisadas = int(difiere.sum())
                if filas_revisadas:
                    revisiones = comp.loc[difiere, [col] + [
                        f"{c}{suf}" for c in cols_valor for suf in ("_ant", "_new")
                    ]]

    # --- Fusionar: el dato nuevo gana ---
    combinado = nuevo if previo is None else pd.concat([previo, nuevo], ignore_index=True)
    combinado = (combinado
                 .drop_duplicates(subset=[col], keep="last")
                 .sort_values(col)
                 .reset_index(drop=True))

    rango_despues = ((combinado[col].min().date(), combinado[col].max().date())
                     if not combinado.empty else None)
    combinado = _restaurar_dtype_tiempo(combinado, col, como_date)

    # --- Escritura atómica ---
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".parquet.tmp")
    combinado.to_parquet(tmp, index=False)
    os.replace(tmp, ruta)

    res = ResultadoUpsert(
        serie=spec.nombre,
        filas_antes=filas_antes,
        filas_despues=len(combinado),
        filas_nuevas=filas_nuevas,
        filas_revisadas=filas_revisadas,
        rango_antes=rango_antes,
        rango_despues=rango_despues,
        revisiones=revisiones,
    )
    logger.info("  %s: %d -> %d filas (+%d nuevas, ~%d revisadas)%s",
                spec.nombre, filas_antes, len(combinado), filas_nuevas, filas_revisadas,
                f", hasta {rango_despues[1]}" if rango_despues else "")
    return res


def estado_series(raiz: Path, hoy: dt.date | None = None) -> pd.DataFrame:
    """Tabla de frescura por serie. Alimenta outputs/estado_datos.json y el aviso del dashboard.

    Esto es lo que faltaba para que una congelación de datos fuera visible: la corrida de
    junio "funcionó" durante dos meses sin que nada dijera que las series no avanzaban.
    """
    hoy = hoy or dt.date.today()
    filas = []
    for spec in SERIES.values():
        ruta = ruta_de(spec, raiz)
        ultima = ultima_fecha(spec, raiz)
        atraso = (hoy - ultima).days if ultima else None
        filas.append({
            "serie": spec.nombre,
            "ruta": spec.ruta_rel,
            "existe": ruta.exists(),
            "ultima_fecha": ultima.isoformat() if ultima else None,
            "dias_atraso": atraso,
            "tolerancia_dias": spec.tolerancia_frescura_dias,
            "frescura_ok": bool(ultima is not None and atraso <= spec.tolerancia_frescura_dias),
        })
    return pd.DataFrame(filas)
