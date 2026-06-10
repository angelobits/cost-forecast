"""Carga y acceso tipado a la configuración externalizada (``config/config.yaml``)

Centraliza rutas, umbrales, semilla y horizonte para evitar números/rutas mágicos
dispersos por el código. Cualquier módulo obtiene la configuración con
:func:`load_config`
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Devuelve la raíz del proyecto (dos niveles por encima de este archivo)"""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """Vista inmutable de la configuración del proyecto.

    Expone el diccionario crudo (``raw``) y helpers para resolver rutas relativas
    a la raíz del proyecto
    """

    raw: dict[str, Any]
    root: Path = field(default_factory=project_root)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, dotted_key: str) -> Path:
        """Resuelve una ruta declarada en ``paths`` (p. ej. ``"consolidated"``)"""
        rel = self.raw["paths"][dotted_key]
        return (self.root / rel).resolve()

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Lee el YAML de configuración y lo envuelve en :class:`Config`

    Args:
        path: Ruta al YAML. Si es ``None`` se usa ``config/config.yaml`` en la raíz

    Returns:
        Instancia de :class:`Config`

    Raises:
        FileNotFoundError: Si el archivo de configuración no existe
    """
    root = project_root()
    cfg_path = Path(path) if path is not None else root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No se encontró la configuración en {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=root)
