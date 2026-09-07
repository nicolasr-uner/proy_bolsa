# proybolsa

Sistema interno de proyección para el mercado colombiano:

1. **Precio de bolsa de energía horario** (XM)
2. **IPP** — Índice de Precios del Productor (DANE)

Modelos interpretables + ML ligero, con **auto-recalibración mensual**: cada vez
que entran datos nuevos, el sistema se reajusta, mide su error real contra lo
predicho y rebalancea. Ver la **metodología completa** en
[docs/METODOLOGIA.md](docs/METODOLOGIA.md) y el estudio de fuentes en [docs/fuentes.md](docs/fuentes.md).

## Entorno

- Python **3.12** (gestionado con [uv](https://docs.astral.sh/uv/)).
- Dependencias fijadas en `pyproject.toml` (pandas 2.2.x por compatibilidad con `pydataxm`).

### Instalación

```powershell
# Crear el entorno (Python 3.12) e instalar el paquete + dev
py -m uv venv --python 3.12 .venv
py -m uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

### Verificación rápida

```powershell
.\.venv\Scripts\python.exe -m pytest          # tests del andamiaje
.\.venv\Scripts\python.exe scripts\check_pydataxm.py        # acceso real a XM
.\.venv\Scripts\python.exe scripts\check_modeling_stack.py  # stack de modelado
```

## Estructura

```
src/proybolsa/      Paquete: ingest, validate, features, models/{bolsa,ipp}, backtest, recalibrate, report
config/             settings.yaml + diccionarios de variables (bolsa, ipp)
data/               raw / interim / processed  (no versionado)
outputs/            forecasts / reports        (no versionado)
scripts/            scripts de validación y utilidades
tests/              pruebas (pytest)
docs/               estudio de fuentes y notas
```

## Estado

Las 9 fases están **implementadas y desplegadas**: ingesta incremental (XM, FRED, DANE manual,
NOAA), features, modelos de bolsa e IPP, backtesting walk-forward, auto-recalibración mensual
(`run_monthly_update.py`) y dashboard Streamlit de 5 vistas en Streamlit Community Cloud. La suite
pasa **185 tests** (2 skipped).

- [x] Fase 0/1: entorno reproducible, fuente XM validada, andamiaje y diccionarios.
- [x] Fase 2: ingesta incremental (XM, FRED, DANE manual, NOAA) + data layer histórico.
- [x] Fase 3-9: features, modelos, backtesting, auto-recalibración, dashboard, deploy.

> **Estado del modelo (importante):** el sistema corre y está desplegado, pero medido en backtest
> ambos modelos aún **pierden contra baselines triviales** (naive / deriva) en los horizontes que
> importan. El valor está en la incertidumbre calibrada y los escenarios, no en el punto. Ver §6 de
> [docs/METODOLOGIA.md](docs/METODOLOGIA.md) y el plan en [docs/PLAN_MEJORAS_2026-09.md](docs/PLAN_MEJORAS_2026-09.md).

> Nota OneDrive: el proyecto vive en una carpeta sincronizada. Conviene excluir
> `.venv/` y `data/` de la sincronización para evitar lentitud y bloqueos.
