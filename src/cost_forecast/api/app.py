"""API REST (FastAPI) para exponer el modelo como microservicio

Desacopla el modelo de cualquier front-end
Levantar en local:  ``make api``  →  http://localhost:8000/docs (Swagger)
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..agent import tools
from ..config import load_config
from ..forecast.scenario import estimate_from_levels, input_bounds

app = FastAPI(
    title="cost-forecast API",
    version="0.1.0",
    description="Estimación y proyección de costos de equipos a partir de materias primas X, Y, Z.",
)


# Esquemas (contratos de entrada/salida)
class EstimateRequest(BaseModel):
    """Escenario what-if: niveles de insumo para un equipo."""

    equipo: str = Field(..., examples=["Equipo1"])
    input_levels: dict[str, float] = Field(..., examples=[{"Y": 560.0}])
    confidence: float = Field(0.90, ge=0.5, le=0.99)


class AskRequest(BaseModel):
    """Pregunta para el agente conversacional."""

    question: str
    web_search: bool = False


# Endpoints de solo lectura (artefactos del modelo)
@app.get("/health", tags=["infra"])
def health() -> dict[str, str]:
    """Sonda de salud para orquestadores (Container Apps, K8s)."""
    return {"status": "ok"}


@app.get("/selection", tags=["modelo"])
def selection(equipo: str | None = None) -> dict[str, Any]:
    """Variables seleccionadas por votación, por equipo."""
    return _guard(tools.get_selected_variables(equipo))


@app.get("/metrics", tags=["modelo"])
def metrics(equipo: str | None = None) -> dict[str, Any]:
    """Métricas de error y direccionales, y mejor modelo por equipo."""
    return _guard(tools.get_metrics(equipo))


@app.get("/forecast", tags=["modelo"])
def forecast(equipo: str | None = None) -> dict[str, Any]:
    """Proyección de costos con banda de predicción por equipo."""
    return _guard(tools.get_forecast(equipo))


@app.get("/eda", tags=["modelo"])
def eda() -> dict[str, Any]:
    """Resumen del EDA (estacionariedad, correlaciones, cointegración)."""
    return _guard(tools.get_eda_summary())


@app.get("/methodology", tags=["modelo"])
def methodology(topic: str | None = None) -> dict[str, Any]:
    """Decisiones metodológicas y su justificación (por qué 6 meses, mensual, etc.)."""
    return _guard(tools.get_methodology(topic))


@app.get("/inputs/{equipo}", tags=["modelo"])
def inputs(equipo: str) -> dict[str, Any]:
    """Insumos seleccionados y su rango observado (para poblar formularios)."""
    cfg = load_config()
    try:
        selected, bounds = input_bounds(cfg, equipo)
    except KeyError as exc:  # equipo inexistente
        raise HTTPException(status_code=404, detail=f"Equipo no encontrado: {equipo}") from exc
    return {"equipo": equipo, "selected": selected, "bounds": bounds}


# Endpoint de estimación
@app.post("/estimate", tags=["modelo"])
def estimate(req: EstimateRequest) -> dict[str, Any]:
    """Estima el costo de un equipo para niveles de insumo dados (escenario what-if)."""
    cfg = load_config()
    if req.equipo not in cfg["schema"]["targets"]:
        raise HTTPException(status_code=404, detail=f"Equipo no encontrado: {req.equipo}")
    result = estimate_from_levels(cfg, req.equipo, req.input_levels, confidence=req.confidence)
    return result.as_dict()


# Endpoint del agente (opcional, requiere ANTHROPIC_API_KEY)
@app.post("/agent/ask", tags=["agente"])
def agent_ask(req: AskRequest) -> dict[str, str]:
    """Consulta al agente conversacional. Requiere ANTHROPIC_API_KEY y el extra 'agent'."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Agente no disponible: define ANTHROPIC_API_KEY (e instala el extra 'agent').",
        )
    try:
        from ..agent.graph import ask, build_agent

        agent = build_agent(include_web_search=req.web_search)
        return {"answer": ask(agent, req.question)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error del agente: {exc}") from exc


def _guard(payload: dict[str, Any]) -> dict[str, Any]:
    """Convierte un artefacto faltante en 404 legible."""
    if isinstance(payload, dict) and "error" in payload:
        raise HTTPException(status_code=404, detail=payload["error"])
    return payload
