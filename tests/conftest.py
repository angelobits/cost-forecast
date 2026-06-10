"""Fixtures compartidas para las pruebas."""

from __future__ import annotations

import pytest

from cost_forecast.config import load_config


@pytest.fixture(scope="session")
def cfg():
    """Configuración del proyecto cargada una vez por sesión de test"""
    return load_config()
