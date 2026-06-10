"""REPL del agente conversacional.

Uso: ``make agent`` o ``python -m cost_forecast.agent.cli``. Requiere el extra
``agent`` instalado y la variable de entorno ``ANTHROPIC_API_KEY``. Sin clave,
muestra un modo de demostración que consulta los artefactos directamente.
"""

from __future__ import annotations

import os
import sys

from ..logging_utils import get_logger
from . import tools

logger = get_logger(__name__)

_DEMO_QUESTIONS = [
    "¿Qué materias primas explican cada equipo y con qué consenso?",
    "¿Cuál es el mejor modelo por equipo y su MAPE?",
    "¿Cuál es la proyección de costo a 6 meses con su banda?",
]


def _demo_mode() -> None:
    """Sin LLM: demuestra que las herramientas/artefactos responden."""
    logger.info("Modo demo (sin ANTHROPIC_API_KEY): consulta directa de artefactos.")
    print("\n[Variables seleccionadas]\n", tools.get_selected_variables())
    print("\n[Métricas]\n", tools.get_metrics())
    print("\n[Proyección]\n", tools.get_forecast("Equipo1"))


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del REPL del agente."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _demo_mode()
        print(
            "\nDefine ANTHROPIC_API_KEY e instala el extra 'agent' "
            "(uv sync --extra agent) para el modo conversacional completo."
        )
        return 0

    from .graph import ask, build_agent

    agent = build_agent()
    print("Agente de costos listo. Escribe tu pregunta ('salir' para terminar).\n")
    print("Sugerencias:", "; ".join(_DEMO_QUESTIONS), "\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if not question:
            continue
        print(ask(agent, question), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
