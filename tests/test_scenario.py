"""Set de pruebas del estimador de costos a partir de niveles de insumos"""

from __future__ import annotations

from cost_forecast.forecast.scenario import estimate_from_levels, selected_inputs


def test_no_change_returns_last_level(cfg):
    """Ingresar el último nivel observado del insumo reproduce el último costo (delta ~0)"""
    for tgt in cfg["schema"]["targets"]:
        _, last = selected_inputs(cfg, tgt)
        r = estimate_from_levels(cfg, tgt, last)
        assert abs(r.predicted_level - r.last_equipo_level) < 1.0
        assert r.lower <= r.predicted_level <= r.upper


def test_higher_input_raises_cost(cfg):
    """Un insumo más caro eleva el costo estimado (relación positiva)"""
    _, last = selected_inputs(cfg, "Equipo1")
    higher = {c: v * 1.1 for c, v in last.items()}
    base = estimate_from_levels(cfg, "Equipo1", last)
    up = estimate_from_levels(cfg, "Equipo1", higher)
    assert up.predicted_level > base.predicted_level
