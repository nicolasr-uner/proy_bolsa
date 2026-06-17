"""Carga de configuracion del proyecto desde archivos YAML en config/."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Raiz del repo = dos niveles arriba de este archivo (src/proybolsa/config.py).
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


def load_yaml(name: str) -> dict[str, Any]:
    """Carga un YAML de la carpeta config/ por nombre (con o sin extension)."""
    path = CONFIG_DIR / name
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    """Configuracion general del proyecto (config/settings.yaml)."""
    return load_yaml("settings.yaml")


def variables(modelo: str) -> dict[str, Any]:
    """Diccionario de variables/supuestos para 'bolsa' o 'ipp'."""
    if modelo not in {"bolsa", "ipp"}:
        raise ValueError("modelo debe ser 'bolsa' o 'ipp'")
    return load_yaml(f"variables_{modelo}.yaml")
