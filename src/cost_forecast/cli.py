"""Interfaz de línea de comandos: despacha cada etapa del pipeline

Uso: ``cost-forecast <stage>`` con stage en {data, eda, features, train,
evaluate, forecast, all}. El Makefile envuelve estos comandos
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .logging_utils import get_logger
from .pipeline import (
    run_all,
    run_data,
    run_eda,
    run_features,
    run_forecast,
    run_train_evaluate,
)

logger = get_logger(__name__)

_STAGES = {
    "data": run_data,
    "eda": run_eda,
    "features": run_features,
    "train": run_train_evaluate,
    "evaluate": run_train_evaluate,
    "forecast": run_forecast,
    "all": lambda cfg: run_all(cfg),
}


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI"""
    parser = argparse.ArgumentParser(description="Pipeline de estimación de costos de equipos.")
    parser.add_argument("stage", choices=sorted(_STAGES), help="Etapa a ejecutar.")
    parser.add_argument("--config", default=None, help="Ruta a config.yaml.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger.info("Ejecutando etapa '%s' (semilla=%d)", args.stage, cfg.seed)
    _STAGES[args.stage](cfg)
    logger.info("Etapa '%s' completada.", args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
