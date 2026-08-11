"""Adquisición y parseo del IPP de Colombia desde el anexo histórico del DANE.

**El IPP se puede automatizar.** `docs/fuentes.md` concluyó "carga manual definitiva", pero
ese veredicto venía de que `banrep.gov.co` está detrás de un CAPTCHA de Radware y de que el
dataset de Socrata no existe. El anexo del DANE es otra cosa: se sirve como archivo estático
en un patrón de URL estable, sin token, sin CAPTCHA y sin JavaScript.

Verificado el 2026-08-11:

    https://www.dane.gov.co/files/operaciones/IPP/anex-IPP-historicos-{mmm}{yyyy}.xlsx

    jul2026  -> 200, 65.449 bytes, Last-Modified 2026-08-06
    ago2026  -> 404 (aún no publicado: el dato de julio salió el 6 de agosto)
    los 12 meses de 2025 -> 200
    sept2025 -> 404  (la abreviatura es SIEMPRE de 3 letras)

Trampa a tener presente: una respuesta 404 del DANE **también** trae la cabecera
`Last-Modified` (con la fecha del momento, de la página de error). La existencia del recurso
se decide por el status code y por nada más.

El anexo trae la serie histórica **completa** (1999-06 en adelante) en cada publicación. Eso
resuelve gratis la política de revisiones que documenta `config/variables_ipp.yaml` ("DANE
publica un dato PROVISIONAL y lo revisa el mes siguiente"): se reemplaza el archivo entero y
se registra qué meses cambiaron, en vez de tener que parchear observaciones sueltas.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from proybolsa.ingest.http import TIMEOUT_DESCARGA, get_bytes, get_text, head

logger = logging.getLogger(__name__)

URL_ANEXO = ("https://www.dane.gov.co/files/operaciones/IPP/"
             "anex-IPP-historicos-{mmm}{anio}.xlsx")

URL_PORTAL_IPP = ("https://www.dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/"
                  "indice-de-precios-del-productor-ipp/ipp-historicos")

# Abreviaturas de 3 letras, en minúscula. NO es 'sept' (verificado: da 404).
MESES_ABBR = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")

MESES_ES = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12,
}

# Series disponibles en el Excel DANE. `objetivo` es el texto de cabecera normalizado
# (sin acentos, sin marcas de nota al pie); `fallback_idx` es el indice de columna
# conocido (0-based) si la deteccion por cabecera falla.
SERIES = {
    "oferta_interna": {"objetivo": "oferta interna",      "fallback_idx": 6, "etiqueta": "Oferta Interna"},
    "prod_nacional":  {"objetivo": "produccion nacional", "fallback_idx": 2, "etiqueta": "Produccion Nacional"},
}

# --- Umbrales de validación, calibrados contra la serie real (324 meses, 1999-06 a 2026-05) ---
# El salto mensual más grande de la historia es +3.89% (ene-2022, pico inflacionario), así que
# 5% deja ~29% de margen sin dejar pasar una columna equivocada (los agregados del Excel
# difieren en decenas de puntos, no en décimas).
MAX_DLOG_MENSUAL = 0.05
# El mínimo real de la serie es 49.29 (1999-06). Ojo: el esquema pandera IPP_MENSUAL exige
# ge(50) y rechazaría el primer mes de la historia.
IPP_MIN_PLAUSIBLE = 40.0
IPP_MAX_PLAUSIBLE = 500.0
MIN_MESES = 300
BASE_DIC_2014 = 100.0
TOL_BASE = 0.05


class ValidacionIPP(ValueError):
    """La serie descargada no pasó las compuertas: no se sobreescribe nada."""


@dataclass(frozen=True)
class AnexoIPP:
    """Un anexo publicado, identificado por el mes de la publicación."""
    url: str
    anio: int
    mes: int
    last_modified: str | None = None
    content_length: int | None = None

    @property
    def periodo(self) -> str:
        return f"{MESES_ABBR[self.mes - 1]}{self.anio}"


def url_anexo(anio: int, mes: int) -> str:
    if not 1 <= mes <= 12:
        raise ValueError(f"mes fuera de rango: {mes}")
    return URL_ANEXO.format(mmm=MESES_ABBR[mes - 1], anio=anio)


def _mes_anterior(anio: int, mes: int) -> tuple[int, int]:
    return (anio - 1, 12) if mes == 1 else (anio, mes - 1)


def resolver_anexo_mas_reciente(hoy: dt.date | None = None, *,
                                max_retroceso: int = 6) -> AnexoIPP:
    """Encuentra el anexo publicado más reciente probando hacia atrás con HEAD.

    Se empieza en el mes corriente y se retrocede hasta el primer 200. Normalmente acierta al
    primer o segundo intento: el DANE publica el dato del mes M a comienzos de M+1, así que
    durante los primeros días de un mes el anexo de ese mes todavía no existe.

    Si los `max_retroceso` meses dan 404, se cae al listado del portal antes de rendirse: eso
    cubre el caso de que el DANE cambie el patrón de nombre.
    """
    hoy = hoy or dt.date.today()
    anio, mes = hoy.year, hoy.month

    intentos: list[str] = []
    for _ in range(max_retroceso):
        url = url_anexo(anio, mes)
        try:
            cab = head(url)
        except Exception as exc:
            logger.warning("HEAD %s falló (%s); se sigue retrocediendo", url, exc)
            intentos.append(f"{MESES_ABBR[mes-1]}{anio}: error {exc}")
            anio, mes = _mes_anterior(anio, mes)
            continue

        if cab.ok:
            logger.info("Anexo IPP encontrado: %s (%s bytes, Last-Modified %s)",
                        url, cab.content_length, cab.last_modified)
            return AnexoIPP(url=url, anio=anio, mes=mes,
                            last_modified=cab.last_modified,
                            content_length=cab.content_length)

        intentos.append(f"{MESES_ABBR[mes-1]}{anio}: HTTP {cab.status}")
        anio, mes = _mes_anterior(anio, mes)

    logger.warning("El patrón de URL no resolvió en %d meses (%s). Probando el portal.",
                   max_retroceso, "; ".join(intentos))
    candidatos = listar_anexos_portal()
    if candidatos:
        mejor = candidatos[0]
        logger.info("Anexo IPP resuelto vía portal: %s", mejor.url)
        return mejor

    raise RuntimeError(
        "No se pudo resolver el anexo del IPP.\n"
        f"  Patrón probado: {URL_ANEXO}\n"
        f"  Intentos: {'; '.join(intentos)}\n"
        f"  El listado de {URL_PORTAL_IPP} tampoco devolvió candidatos.\n"
        "Ruta manual: descargar el .xlsx del portal y correr\n"
        "  .venv\\Scripts\\python scripts/parsear_ipp_dane.py <archivo.xlsx>"
    )


def listar_anexos_portal() -> list[AnexoIPP]:
    """FALLBACK: raspa los enlaces del portal del IPP y los ordena de más nuevo a más viejo.

    No es la ruta normal — solo corre si el patrón de URL deja de funcionar.
    """
    patron = re.compile(r"anex-IPP-historicos-([a-z]{3})(\d{4})\.xlsx", re.IGNORECASE)
    try:
        html = get_text(URL_PORTAL_IPP)
    except Exception as exc:
        logger.error("No se pudo leer el portal del IPP: %s", exc)
        return []

    vistos: set[tuple[int, int]] = set()
    encontrados: list[AnexoIPP] = []
    for mmm, anio_txt in patron.findall(html):
        mmm = mmm.lower()
        if mmm not in MESES_ABBR:
            continue
        anio, mes = int(anio_txt), MESES_ABBR.index(mmm) + 1
        if (anio, mes) in vistos:
            continue
        vistos.add((anio, mes))
        encontrados.append(AnexoIPP(url=url_anexo(anio, mes), anio=anio, mes=mes))

    encontrados.sort(key=lambda a: (a.anio, a.mes), reverse=True)
    logger.info("El portal listó %d anexos; el más reciente es %s",
                len(encontrados), encontrados[0].periodo if encontrados else "-")
    return encontrados


def descargar_anexo(anexo: AnexoIPP, destino: Path) -> Path:
    """Descarga el .xlsx a `destino`. Devuelve la ruta escrita."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    contenido = get_bytes(anexo.url, timeout=TIMEOUT_DESCARGA)
    if len(contenido) < 10_000:
        raise ValidacionIPP(
            f"El anexo {anexo.periodo} pesa solo {len(contenido)} bytes: no parece un Excel "
            "(los anexos reales rondan los 65 KB). Posible pagina de error."
        )
    destino.write_bytes(contenido)
    logger.info("Descargado %s -> %s (%d bytes)", anexo.url, destino, len(contenido))
    return destino


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Parseo del Excel (movido desde scripts/parsear_ipp_dane.py, sin cambios de lógica)
# ---------------------------------------------------------------------------

def _norm(valor: object) -> str:
    """Normaliza texto de cabecera: sin acentos, casefold, sin notas al pie ni simbolos."""
    if valor is None:
        return ""
    s = unicodedata.normalize("NFKD", str(valor))
    s = "".join(c for c in s if not unicodedata.combining(c))      # quitar acentos
    s = s.casefold()
    s = "".join(c if (c.isalpha() or c.isspace()) else " " for c in s)  # quitar digitos/simbolos
    return " ".join(s.split())


def _detectar_columna(rows: list[tuple], serie: str) -> int:
    """Encuentra el indice de columna del agregado `serie` por nombre de cabecera.

    Escanea las primeras 8 filas (cabeceras). Devuelve el fallback conocido si no
    encuentra una coincidencia exacta.
    """
    objetivo = SERIES[serie]["objetivo"]
    for row in rows[:8]:
        for idx, cell in enumerate(row):
            if _norm(cell) == objetivo:
                logger.info("Cabecera '%s' detectada en columna indice %d",
                            SERIES[serie]["etiqueta"], idx)
                return idx
    fallback = SERIES[serie]["fallback_idx"]
    logger.warning("No se detecto la cabecera '%s'; usando indice de fallback %d",
                   SERIES[serie]["etiqueta"], fallback)
    return fallback


def parsear_excel_ipp(
    xlsx_path: Path | str,
    hoja: str = "IPP Histórico",
    serie: str = "oferta_interna",
) -> pd.DataFrame:
    """Extrae [fecha, ipp] del anexo del DANE. `hoja` se conserva por compatibilidad de CLI."""
    import openpyxl

    if serie not in SERIES:
        raise ValueError(f"serie invalida: {serie!r}. Opciones: {list(SERIES)}")

    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)

    # Encontrar la hoja correcta (puede tener nombre con encoding roto)
    sheet = None
    for name in wb.sheetnames:
        if "IPP" in name or "Hist" in name:
            sheet = wb[name]
            break
    if sheet is None:
        raise ValueError(f"No se encontro hoja IPP en {xlsx_path}. Hojas: {wb.sheetnames}")

    # Materializar (la hoja es chica) para evitar problemas de doble iteracion en read_only
    rows = list(sheet.iter_rows(values_only=True))

    col_idx = _detectar_columna(rows, serie)
    logger.info("Serie objetivo: %s", SERIES[serie]["etiqueta"])

    records = []
    anio_actual = None

    # Los datos arrancan en la fila 6 (1-based) = indice 5
    for row in rows[5:]:
        col_anio = row[0] if len(row) > 0 else None
        col_mes = row[1] if len(row) > 1 else None
        col_val = row[col_idx] if len(row) > col_idx else None

        if col_anio and str(col_anio).strip().isdigit():
            anio_actual = int(str(col_anio).strip())

        if anio_actual is None or col_mes is None or col_val is None:
            continue

        mes_str = str(col_mes).strip().split()[0]
        mes_num = MESES_ES.get(mes_str)
        if mes_num is None:
            continue

        try:
            ipp_val = float(col_val)
        except (ValueError, TypeError):
            continue

        records.append({"fecha": pd.Timestamp(anio_actual, mes_num, 1), "ipp": ipp_val})

    if not records:
        raise ValueError(
            f"No se extrajo ningun dato de la columna {col_idx}. "
            "Revisar la estructura del Excel o el flag --serie."
        )

    df = pd.DataFrame(records).drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    logger.info("Total meses: %d  Rango: %s -- %s",
                len(df), df["fecha"].min().date(), df["fecha"].max().date())
    return df


# ---------------------------------------------------------------------------
# Compuertas de validación y diff de revisiones
# ---------------------------------------------------------------------------

def validar_serie_ipp(df: pd.DataFrame, *, viejo: pd.DataFrame | None = None,
                      hoy: dt.date | None = None) -> None:
    """Compuertas antes de sobreescribir `ipp_manual.csv`. Lanza `ValidacionIPP` si algo falla.

    El principio: es mejor quedarse con el dato del mes pasado que sobreescribir la serie con
    algo roto. Un CSV corrupto se propaga a las features, al ajuste, a los `.pkl` y al
    dashboard, y para cuando se nota ya está commiteado.

    Antes esto era un `logger.info` que decía "Dic-2014 = X (esperado 100.0)" y seguía
    adelante pasara lo que pasara.
    """
    hoy = hoy or dt.date.today()
    fallas: list[str] = []

    if df.empty or "ipp" not in df.columns or "fecha" not in df.columns:
        raise ValidacionIPP("La serie viene vacía o sin las columnas fecha/ipp")

    s = df.dropna(subset=["ipp"]).sort_values("fecha").reset_index(drop=True)

    # 1. Base dic-2014 = 100: detecta un cambio de base del DANE (que reescalaría toda la serie)
    dic = s.loc[pd.to_datetime(s["fecha"]) == pd.Timestamp("2014-12-01"), "ipp"]
    if dic.empty:
        fallas.append("no aparece diciembre de 2014, que es la base del índice")
    elif abs(float(dic.iloc[0]) - BASE_DIC_2014) > TOL_BASE:
        fallas.append(
            f"dic-2014 = {float(dic.iloc[0]):.2f}, se esperaba {BASE_DIC_2014} "
            f"(±{TOL_BASE}). Posible cambio de base del DANE"
        )

    # 2. Longitud mínima: el anexo trae la historia completa, no un fragmento
    if len(s) < MIN_MESES:
        fallas.append(f"solo {len(s)} meses, se esperaban al menos {MIN_MESES}")

    # 3. Rango plausible: detecta que se leyó una columna que no es un índice
    if s["ipp"].min() < IPP_MIN_PLAUSIBLE or s["ipp"].max() > IPP_MAX_PLAUSIBLE:
        fallas.append(
            f"valores fuera de rango [{IPP_MIN_PLAUSIBLE}, {IPP_MAX_PLAUSIBLE}]: "
            f"min={s['ipp'].min():.2f} max={s['ipp'].max():.2f}"
        )

    # 4. Continuidad: un salto mensual grande delata columna equivocada o mezcla de agregados
    if len(s) > 1:
        dlog = np.log(s["ipp"]).diff().abs()
        if dlog.max() > MAX_DLOG_MENSUAL:
            i = int(dlog.idxmax())
            fallas.append(
                f"salto mensual de {dlog.max()*100:.2f}% en {pd.to_datetime(s.loc[i,'fecha']).date()} "
                f"(máximo tolerado {MAX_DLOG_MENSUAL*100:.0f}%; el récord histórico real es 3.89%)"
            )

    # 5. Sin fechas futuras
    ultima = pd.to_datetime(s["fecha"]).max().date()
    if ultima > hoy:
        fallas.append(f"el último mes ({ultima}) es futuro respecto a hoy ({hoy})")

    # 6. La serie nunca se acorta ni retrocede
    if viejo is not None and not viejo.empty:
        if len(s) < len(viejo):
            fallas.append(f"la serie se acortó: {len(viejo)} -> {len(s)} meses")
        ultima_vieja = pd.to_datetime(viejo["fecha"]).max().date()
        if ultima < ultima_vieja:
            fallas.append(f"el último mes retrocedió: {ultima_vieja} -> {ultima}")

    if fallas:
        raise ValidacionIPP(
            "La serie IPP descargada no pasó la validación; NO se sobreescribió nada:\n  - "
            + "\n  - ".join(fallas)
        )
    logger.info("Validación IPP OK: %d meses, %s -> %s, dic-2014=%.2f",
                len(s), pd.to_datetime(s['fecha']).min().date(), ultima,
                float(dic.iloc[0]) if not dic.empty else float("nan"))


def diff_series_ipp(viejo: pd.DataFrame | None,
                    nuevo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa lo nuevo de lo revisado.

    Devuelve `(nuevas, revisadas)`. `revisadas` lleva [fecha, ipp_viejo, ipp_nuevo, dif_pct]
    y es el rastro de la política de revisiones del DANE (provisional -> definitivo): sin esto,
    un valor que cambia entra como si siempre hubiera sido así.
    """
    cols_rev = ["fecha", "ipp_viejo", "ipp_nuevo", "dif_pct"]
    n = nuevo.copy()
    n["fecha"] = pd.to_datetime(n["fecha"])

    if viejo is None or viejo.empty:
        return n.reset_index(drop=True), pd.DataFrame(columns=cols_rev)

    v = viejo.copy()
    v["fecha"] = pd.to_datetime(v["fecha"])

    m = v[["fecha", "ipp"]].merge(n[["fecha", "ipp"]], on="fecha", how="outer",
                                  suffixes=("_viejo", "_nuevo"))
    nuevas = (m[m["ipp_viejo"].isna() & m["ipp_nuevo"].notna()][["fecha", "ipp_nuevo"]]
              .rename(columns={"ipp_nuevo": "ipp"}).sort_values("fecha").reset_index(drop=True))

    ambos = m[m["ipp_viejo"].notna() & m["ipp_nuevo"].notna()].copy()
    # 1e-4 en puntos de índice: por debajo de eso es redondeo del Excel, no una revisión.
    cambiadas = ambos[(ambos["ipp_viejo"] - ambos["ipp_nuevo"]).abs() > 1e-4].copy()
    if cambiadas.empty:
        return nuevas, pd.DataFrame(columns=cols_rev)

    cambiadas["dif_pct"] = 100 * (cambiadas["ipp_nuevo"] / cambiadas["ipp_viejo"] - 1)
    return nuevas, cambiadas[cols_rev].sort_values("fecha").reset_index(drop=True)
