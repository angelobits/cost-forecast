"""Estimador: costo del equipo dado un nivel de insumo

Responde la pregunta de negocio directa —*"si la materia prima vale V, ¿cuánto
costará el equipo?"*— usando el modelo **interpretable** (ElasticNet sobre cambios)
de cada equipo. Se modela en cambios y se reconstruye el nivel anclando en el
último valor observado:

    nivel_equipo ≈ último_nivel_equipo + coef · (nivel_insumo_ingresado − último_nivel_insumo)

A diferencia de la proyección autónoma (donde el insumo evoluciona por sí solo),
aquí el usuario fija el escenario del insumo. Es la base para anticipar costos y
comparar proveedores
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from ..config import Config
from ..data.loaders import load_consolidated
from ..features.transforms import to_monthly_level
from ..models.regression import ElasticNetChangeModel

# Cuantil normal para la banda (p. ej. 90% -> 1.645).
_Z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


@dataclass
class ScenarioResult:
    """Resultado de un escenario what-if para un equipo"""

    target: str
    selected: list[str]
    last_inputs: dict[str, float]
    entered_inputs: dict[str, float]
    last_equipo_level: float
    predicted_level: float
    predicted_change: float
    lower: float
    upper: float
    confidence: float
    sensitivities: dict[str, float]  # Δnivel_equipo por +1 unidad de cada insumo

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "selected": self.selected,
            "last_inputs": {k: round(v, 3) for k, v in self.last_inputs.items()},
            "entered_inputs": {k: round(v, 3) for k, v in self.entered_inputs.items()},
            "last_equipo_level": round(self.last_equipo_level, 2),
            "predicted_level": round(self.predicted_level, 2),
            "predicted_change": round(self.predicted_change, 2),
            "lower": round(self.lower, 2),
            "upper": round(self.upper, 2),
            "confidence": self.confidence,
            "sensitivities": {k: round(v, 3) for k, v in self.sensitivities.items()},
        }


def _monthly_and_selected(cfg: Config, target: str) -> tuple[pd.DataFrame, list[str], str]:
    """Carga niveles mensuales (sin prefijo) y los insumos seleccionados del equipo."""
    prefix = cfg["schema"]["price_prefix"]
    daily = load_consolidated(cfg.path("consolidated")).rename(
        columns=lambda c: c.replace(prefix, "")
    )
    monthly = to_monthly_level(daily)
    sel_path = cfg.path("artifacts_dir") / "selection.json"
    if sel_path.exists():
        selected = json.loads(sel_path.read_text(encoding="utf-8"))[target]["selected"]
    else:  # pragma: no cover - fallback si no se corrió features
        selected = cfg["schema"]["inputs"]
    return monthly, selected, target


def selected_inputs(cfg: Config, target: str) -> tuple[list[str], dict[str, float]]:
    """Devuelve (insumos seleccionados, último nivel observado de cada uno)."""
    monthly, selected, _ = _monthly_and_selected(cfg, target)
    last = monthly[selected].iloc[-1]
    return selected, {c: float(last[c]) for c in selected}


def input_bounds(cfg: Config, target: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Rango observado (mín/máx histórico **diario**) y último nivel de cada insumo.

    Se usa para acotar el simulador al dominio real de los datos —no a un ±X% del
    último valor—, de modo que el usuario pueda recorrer todo el histórico observado
    """
    prefix = cfg["schema"]["price_prefix"]
    daily = load_consolidated(cfg.path("consolidated")).rename(
        columns=lambda c: c.replace(prefix, "")
    )
    monthly = to_monthly_level(daily)
    selected, _ = selected_inputs(cfg, target)
    bounds = {
        c: {
            "last": float(monthly[c].iloc[-1]),
            "min": float(daily[c].min()),
            "max": float(daily[c].max()),
        }
        for c in selected
    }
    return selected, bounds


def estimate_from_levels(
    cfg: Config, target: str, input_levels: dict[str, float], confidence: float = 0.90
) -> ScenarioResult:
    """Estima el costo del equipo para niveles de insumo dados

    Args:
        cfg: Configuración del proyecto.
        target: 'Equipo1' o 'Equipo2'.
        input_levels: Nivel deseado por insumo seleccionado (p. ej. {'Y': 560}).
        confidence: Nivel de la banda (0.80, 0.90 o 0.95)

    Returns:
        :class:`ScenarioResult` con nivel predicho, cambio, banda y sensibilidades
    """
    monthly, selected, _ = _monthly_and_selected(cfg, target)
    y_level = monthly[target]
    x_change = monthly[selected].diff()
    y_change = y_level.diff()
    joined = pd.concat([y_change.rename("y"), x_change], axis=1).dropna()

    model = ElasticNetChangeModel(cfg["models"]["elasticnet"]["l1_ratio"], cfg.seed).fit(
        joined[selected], joined["y"]
    )

    last_inputs = {c: float(monthly[selected].iloc[-1][c]) for c in selected}
    last_equipo = float(y_level.iloc[-1])
    entered = {c: float(input_levels.get(c, last_inputs[c])) for c in selected}

    delta = pd.DataFrame([{c: entered[c] - last_inputs[c] for c in selected}])
    pred_change = float(model.predict(delta).iloc[0])
    pred_level = last_equipo + pred_change

    # Sensibilidad: Δnivel_equipo ante +1 unidad de cada insumo (perturbación unitaria).
    sensitivities: dict[str, float] = {}
    for c in selected:
        unit = pd.DataFrame([{k: (1.0 if k == c else 0.0) for k in selected}])
        sensitivities[c] = float(model.predict(unit).iloc[0])

    z = _Z.get(round(confidence, 2), 1.645)
    margin = z * model.resid_std_
    return ScenarioResult(
        target=target,
        selected=selected,
        last_inputs=last_inputs,
        entered_inputs=entered,
        last_equipo_level=last_equipo,
        predicted_level=pred_level,
        predicted_change=pred_change,
        lower=pred_level - margin,
        upper=pred_level + margin,
        confidence=confidence,
        sensitivities=sensitivities,
    )
