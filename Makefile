# =============================================================================
# Makefile
# =============================================================================

UV ?= uv
RUN = PYTHONPATH=src $(UV) run --no-sync python -m
CLI = $(RUN) cost_forecast.cli

.DEFAULT_GOAL := help
.PHONY: help install data eda features train evaluate forecast agent ui report lint test all clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Crea el entorno e instala dependencias (agent + ui + api + dev)
	$(UV) sync --extra agent --extra ui --extra api --extra dev

data:  ## Carga, reconcilia y persiste datos procesados
	$(CLI) data

eda:  ## EDA: estacionariedad, correlaciones, lags, cointegración, outliers
	$(CLI) eda

features:  ## Selección de variables por votación + VIF (por equipo)
	$(CLI) features

train:  ## Entrena los modelos por equipo
	$(CLI) train

evaluate:  ## Evalúa real vs predicho (error + direccional) y residuales
	$(CLI) evaluate

forecast:  ## Proyecta costos al horizonte con bandas de predicción
	$(CLI) forecast

all:  ## Ejecuta el pipeline completo (data -> forecast)
	$(CLI) all

agent:  ## Lanza el agente conversacional en terminal
	$(RUN) cost_forecast.agent.cli

ui:  ## Lanza la UI Streamlit (dashboard + chat con el agente)
	PYTHONPATH=src $(UV) run --no-sync streamlit run app/streamlit_app.py

api:  ## Lanza la API REST (FastAPI) en http://localhost:8000/docs
	PYTHONPATH=src $(UV) run --no-sync uvicorn cost_forecast.api.app:app --reload --port 8000

report:  ## Recuerda dónde está el informe y los artefactos
	@echo "Informe:   reports/INFORME.md"
	@echo "Arquitectura Azure: reports/arquitectura_azure.md"
	@echo "Artefactos: reports/artifacts/  | Figuras: reports/figures/"

lint:  ## Verifica formato y estilo (ruff + black --check)
	$(UV) run --no-sync ruff check src tests
	$(UV) run --no-sync black --check src tests

test:  ## Ejecuta la batería de pruebas
	PYTHONPATH=src $(UV) run --no-sync pytest

clean:  ## Limpia artefactos generados
	rm -rf reports/artifacts/* reports/figures/* reports/eda/* data/processed/* \
		.pytest_cache **/__pycache__
