"""Set de pruebas de la API REST usando el cliente de prueba"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cost_forecast.api.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_selection_returns_data_driven_relationships():
    r = client.get("/selection")
    if r.status_code == 404:
        pytest.skip("Artefactos no generados en este entorno")
    data = r.json()
    assert data["Equipo1"]["selected"] == ["Y"]
    assert data["Equipo2"]["selected"] == ["Z"]


def test_estimate_no_change_matches_last():
    # Pedir el rango para anclar en el último valor observado
    r = client.get("/inputs/Equipo1")
    if r.status_code == 404:
        pytest.skip("Artefactos no generados (correr make all).")
    last = r.json()["bounds"]["Y"]["last"]
    resp = client.post(
        "/estimate",
        json={"equipo": "Equipo1", "input_levels": {"Y": last}, "confidence": 0.90},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert abs(body["predicted_level"] - body["last_equipo_level"]) < 1.0


def test_estimate_unknown_equipo_404():
    resp = client.post("/estimate", json={"equipo": "EquipoX", "input_levels": {}})
    assert resp.status_code == 404
