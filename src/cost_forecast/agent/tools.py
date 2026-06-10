"""Herramientas del agente: lectura de artefactos del modelo y búsqueda web

Las funciones de lectura son puras (sin dependencias de LLM) para poder probarse
de forma unitaria. El agente las expone como tools y decide cuándo invocarlas.
Constituyen el componente RAG sobre los resultados propios del modelo
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..config import load_config


def _artifacts_dir() -> Path:
    cfg = load_config()
    return cfg.path("artifacts_dir")


def _read(name: str) -> dict[str, Any]:
    path = _artifacts_dir() / name
    if not path.exists():
        return {"error": f"Artefacto '{name}' no encontrado. Ejecuta el pipeline (make all)."}
    return json.loads(path.read_text(encoding="utf-8"))


def get_selected_variables(equipo: str | None = None) -> dict[str, Any]:
    """Devuelve las materias primas seleccionadas (por votación) por equipo

    Args:
        equipo: 'Equipo1' o 'Equipo2'. Si es None, devuelve ambos.
    """
    data = _read("selection.json")
    if "error" in data:
        return data
    summary = {
        k: {"selected": v["selected"], "consensus": v["consensus"], "threshold": v["threshold"]}
        for k, v in data.items()
    }
    return summary.get(equipo, summary) if equipo else summary


def get_metrics(equipo: str | None = None) -> dict[str, Any]:
    """Devuelve métricas de error y direccionales, el mejor modelo y el VIF por equipo."""
    data = _read("evaluation.json")
    if "error" in data:
        return data
    vif = _read("vif.json")
    targets = data.get("targets", {})
    out = {
        k: {
            "best_model": v["best_model"],
            "metrics": v["metrics"],  # incluye 'by_split' (test/validación) y direccionales
            "residual_diagnostics": v["residual_diagnostics"],
            "vif": vif.get(k) if isinstance(vif, dict) and "error" not in vif else None,
        }
        for k, v in targets.items()
    }
    return out.get(equipo, out) if equipo else out


def get_forecast(equipo: str | None = None) -> dict[str, Any]:
    """Devuelve la proyección de costos con banda de predicción por equipo."""
    data = _read("forecast.json")
    if "error" in data:
        return data
    return data.get(equipo, data) if equipo else data


def get_eda_summary() -> dict[str, Any]:
    """Devuelve un resumen del EDA: estacionariedad, correlaciones, parcial y cointegración."""
    data = _read("eda.json")
    if "error" in data:
        return data
    return {
        "stationarity_levels": data["stationarity_levels"],
        "stationarity_diffs": data["stationarity_diffs"],
        "correlation_diffs": data["correlation_diffs"],
        "partial_correlation": data.get("partial_correlation"),  # prueba de espuriedad
        "crosscorr_best": data["crosscorr_best"],
        "cointegration_engle_granger": data.get("cointegration_engle_granger"),
        "cointegration_johansen": data["cointegration_johansen"],
    }


def get_scenario(
    equipo: str, input_levels: dict[str, float], confidence: float = 0.90
) -> dict[str, Any]:
    """Estima el costo de un equipo para un escenario de precios de insumo (what-if).

    Permite responder preguntas del tipo "si Y sube a 560, ¿cuánto costará Equipo1?".

    Args:
        equipo: 'Equipo1' o 'Equipo2'.
        input_levels: nivel deseado por insumo seleccionado, p. ej. {'Y': 560}.
        confidence: nivel de la banda (0.80, 0.90, 0.95 o 0.99).
    """
    from ..forecast.scenario import estimate_from_levels

    cfg = load_config()
    if equipo not in cfg["schema"]["targets"]:
        return {"error": f"Equipo no encontrado: {equipo}"}
    return estimate_from_levels(cfg, equipo, input_levels, confidence=confidence).as_dict()


def get_methodology(topic: str | None = None) -> dict[str, Any]:
    """Devuelve las decisiones metodológicas del proyecto y su justificación.

    Permite que el agente responda preguntas de "por qué" (por qué 6 meses, por qué
    mensual, por qué se modela en cambios, cómo se eligió el modelo, etc.) con base en
    las decisiones documentadas, no inventando. Los valores numéricos provienen de la
    configuración (fuente única de verdad); las justificaciones están curadas.

    Args:
        topic: clave concreta (p. ej. 'horizonte_pronostico'); si None, devuelve todas.
    """
    cfg = load_config()
    w = cfg["window"]
    f = cfg["forecast"]
    sel = cfg["selection"]
    decisions: dict[str, Any] = {
        "horizonte_pronostico": {
            "valor": f"{w['forecast_horizon_months']} meses",
            "justificacion": (
                "Tres pilares: (1) la planeación/aprovisionamiento del proyecto opera por "
                "fases de ~1-2 trimestres; (2) la validación fuera de muestra cubre ~8 meses "
                "recientes, así que no se proyecta mucho más allá de lo verificado; (3) más "
                "allá de ~6 meses la banda de incertidumbre se ensancha tanto (los insumos se "
                "comportan como random walk) que el punto pierde valor accionable. El ancho de "
                "la banda casi se triplica entre el mes 1 y el 6. Es parametrizable en config."
            ),
        },
        "frecuencia": {
            "valor": "mensual",
            "justificacion": (
                "La planeación financiera del proyecto es mensual; a escala diaria hay mucho "
                "ruido. El EDA y la selección usan datos diarios (más potencia estadística), "
                "pero el entrenamiento, la evaluación y el pronóstico se hacen en mensual."
            ),
        },
        "representacion_niveles_vs_cambios": {
            "valor": "se modela en cambios (diferencias) y se reconstruye el nivel",
            "justificacion": (
                "Las series son no estacionarias en nivel (correlaciones espurias ~0.99 por "
                "tendencia común) y estacionarias en diferencias. Se modela el cambio mensual y "
                "se reconstruye el nivel acumulando cambios sobre el último nivel observado."
            ),
        },
        "consolidacion_mensual": {
            "valor": "nivel de fin de mes (último valor), no promedio",
            "justificacion": (
                "Supuesto de negocio: el último nivel del mes es lo más cercano al precio de la "
                "próxima compra y el ancla natural para reconstruir niveles. Es un supuesto "
                "deliberado; existe la alternativa de usar el promedio mensual."
            ),
        },
        "ventana_temporal": {
            "valor": (
                f"train {w['train_start']}→{w['train_end']}, "
                f"test {w['test_start']}→{w['test_end']}, "
                f"validation {w['validation_start']}→{w['validation_end']}"
            ),
            "justificacion": (
                "Partición temporal sin fuga: se entrena con el pasado y se evalúa con el "
                "futuro (test y validación posteriores). Evita filtrar el futuro al pasado."
            ),
        },
        "seleccion_modelo": {
            "valor": "mejor modelo por equipo según menor RMSE fuera de muestra",
            "justificacion": (
                "Se comparan ElasticNet, SARIMAX y GBM contra una media móvil (baseline). Se "
                "elige por RMSE (castiga errores grandes = riesgo presupuestal). Resultado: "
                "Equipo1=SARIMAX, Equipo2=ElasticNet. MAPE se usa para comunicar."
            ),
        },
        "banda_incertidumbre": {
            "valor": f"confianza {int(f['confidence_level'] * 100)}% (percentiles 5 y 95)",
            "justificacion": (
                "Se simulan 1.000 trayectorias (Monte Carlo) propagando incertidumbre del "
                "insumo y del modelo; la banda son los percentiles 5-95. Para presupuestar de "
                "forma prudente se usa el límite superior."
            ),
        },
        "baseline": {
            "valor": f"media móvil de {cfg['models']['baseline_ma_months']} meses (solo línea base)",
            "justificacion": (
                "Mide cuánto aportan realmente los insumos: si un modelo no la supera, no sirve. "
                "Los modelos la superan ~3-5x en MAPE. Nunca es el método final."
            ),
        },
        "seleccion_variables": {
            "valor": f"votación de 6 métodos, consenso >= {sel['consensus_threshold']}",
            "justificacion": (
                "Seis métodos heterogéneos votan (0/1) por insumo; entra si lo respalda >=60% "
                "de ellos. Robusto porque no depende de un solo criterio. Resultado: Equipo1->Y, "
                "Equipo2->Z (X es ruido)."
            ),
        },
    }
    return decisions.get(topic, decisions) if topic else decisions


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Búsqueda web para enriquecer el pronóstico con contexto de mercado.

    Usa la herramienta de búsqueda web del servidor de Anthropic si hay
    ``ANTHROPIC_API_KEY``; en su defecto degrada de forma elegante.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "query": query,
            "results": [],
            "note": "Sin ANTHROPIC_API_KEY: búsqueda web deshabilitada en este entorno.",
        }
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        cfg = load_config()
        resp = client.messages.create(
            model=cfg["agent"]["model"],
            max_tokens=cfg["agent"]["max_tokens"],
            messages=[{"role": "user", "content": query}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": max_results}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return {"query": query, "summary": text}
    except Exception as exc:  # pragma: no cover - depende de red/clave
        return {"query": query, "results": [], "note": f"Búsqueda web no disponible: {exc}"}
