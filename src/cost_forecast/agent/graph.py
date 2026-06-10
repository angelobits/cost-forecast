"""Agente conversacional con LangGraph + API de Anthropic.

El agente **percibe** (lee artefactos del modelo), **decide** qué herramienta usar
y **actúa** (consulta resultados o busca contexto de mercado en la web) para
alcanzar el objetivo: responder preguntas del evaluador sobre los costos
proyectados, combinando el modelo propio con conocimiento externo.

Las dependencias de LLM se importan de forma diferida para que el resto del
paquete funcione sin instalar el extra ``agent``.
"""

from __future__ import annotations

from typing import Any

from ..config import load_config
from . import tools

SYSTEM_PROMPT = (
    "Eres el asistente analítico del proyecto de estimación de costos de equipos de "
    "construcción. Respondes con base en los ARTEFACTOS del modelo (selección de "
    "variables por votación, métricas, proyección con bandas) usando las herramientas "
    "disponibles; NUNCA inventes números. Hallazgos clave derivados de los datos: el "
    "Equipo1 se explica por la materia prima Y y el Equipo2 por la Z; la materia prima "
    "no seleccionada es ruido para ese equipo. Para preguntas de DECISIONES o de 'por "
    "qué' (por qué 6 meses, por qué mensual, por qué se modela en cambios, cómo se eligió "
    "el modelo, etc.) usa la herramienta get_methodology, que devuelve la justificación "
    "documentada; no respondas que no hay información sin antes consultarla. Para escenarios "
    "del tipo 'si el insumo vale X, ¿cuánto costará el equipo?' usa get_scenario. Si te "
    "preguntan por contexto de mercado, usa la búsqueda web y distingue lo que viene del "
    "modelo de lo externo. Responde en español, conciso y cuantitativo."
)


def _build_tool_specs(include_web_search: bool) -> list[Any]:
    """Crea las herramientas LangChain a partir de las funciones puras.

    La búsqueda web se incluye solo si ``include_web_search`` es True (control de
    costo: la herramienta de búsqueda server-side de Anthropic se factura aparte).
    """
    from langchain_core.tools import tool

    specs = [
        tool(tools.get_selected_variables),
        tool(tools.get_metrics),
        tool(tools.get_forecast),
        tool(tools.get_eda_summary),
        tool(tools.get_methodology),
        tool(tools.get_scenario),
    ]
    if include_web_search:
        specs.append(tool(tools.web_search))
    return specs


def build_agent(include_web_search: bool | None = None):
    """Construye un agente ReAct (LangGraph) con el modelo Anthropic y las tools.

    Args:
        include_web_search: Si es None, usa ``agent.web_search`` de la config.
    """
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent

    cfg = load_config()["agent"]
    if include_web_search is None:
        include_web_search = bool(cfg.get("web_search", False))
    llm = ChatAnthropic(
        model=cfg["model"], max_tokens=cfg["max_tokens"], temperature=cfg["temperature"]
    )
    return create_react_agent(llm, _build_tool_specs(include_web_search), prompt=SYSTEM_PROMPT)


def ask(agent, question: str) -> str:
    """Envía una pregunta al agente y devuelve la respuesta final en texto."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content
