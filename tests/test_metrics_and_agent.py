"""Set de pruebas de métricas, reconstrucción de nivel y herramientas del agente"""

from __future__ import annotations

import pandas as pd

from cost_forecast.agent import tools
from cost_forecast.evaluation.metrics import directional_metrics, error_metrics
from cost_forecast.models.base import onestep_level_from_changes


def test_error_metrics_perfect_prediction():
    y = pd.Series([10.0, 12.0, 11.0, 13.0])
    m = error_metrics(y, y, y)
    assert m["MAE"] == 0.0
    assert m["RMSE"] == 0.0
    assert m["R2"] == 1.0


def test_directional_metrics_perfect():
    y = pd.Series([1.0, 2.0, 1.5, 3.0, 2.0])
    d = directional_metrics(y, y)
    assert d["accuracy"] == 1.0
    assert d["f1"] == 1.0


def test_onestep_level_reconstruction():
    level = pd.Series(
        [100.0, 101.0, 103.0], index=pd.date_range("2020-01-31", periods=3, freq="ME")
    )
    changes = pd.Series([2.0], index=[level.index[-1]])
    rec = onestep_level_from_changes(level, changes)
    # nivel_{t-1}=101 + cambio 2 = 103
    assert rec.iloc[-1] == 103.0


def test_web_search_degrades_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = tools.web_search("acero precios")
    assert "note" in res and res["results"] == []


def test_agent_tools_read_artifacts_if_present():
    """Si el pipeline ya corrió, las tools devuelven la relación correcta"""
    sv = tools.get_selected_variables()
    if "error" in sv:
        return
    assert sv["Equipo1"]["selected"] == ["Y"]
    assert sv["Equipo2"]["selected"] == ["Z"]


def test_methodology_answers_why_questions():
    """La herramienta de metodología expone las justificaciones"""
    m = tools.get_methodology()
    assert "horizonte_pronostico" in m
    assert "6 meses" in m["horizonte_pronostico"]["valor"]
    assert len(m["horizonte_pronostico"]["justificacion"]) > 50
    # consulta por tema puntual
    h = tools.get_methodology("frecuencia")
    assert h["valor"] == "mensual"


def test_scenario_tool_runs_whatif():
    """La herramienta de escenario estima un costo para un nivel de insumo dado"""
    r = tools.get_scenario("Equipo1", {"Y": 600.0})
    if "error" in r:
        return  # artefactos no generados en este entorno
    assert "predicted_level" in r and r["lower"] <= r["predicted_level"] <= r["upper"]
    assert tools.get_scenario("EquipoX", {})["error"]
