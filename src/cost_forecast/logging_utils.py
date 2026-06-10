"""Configuración de logging estructurado para todo el paquete

Se usa logging (no ``print``) para trazabilidad y para poder silenciar/redirigir
salida en tests y en producción
"""

from __future__ import annotations

import logging
import random
import sys

import numpy as np

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configura el handler raíz una sola vez (idempotente)"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger configurado para el módulo dado"""
    configure_logging()
    return logging.getLogger(name)


def set_global_seed(seed: int) -> None:
    """Fija la semilla global (random + numpy) para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
