"""UI del proyecto: dashboard de resultados, estimador y chat con el agente de IA.

Dos pestañas:
1. **Resultados**: Aquí se leen los artefactos del pipeline (selección por votación, métricas,
   proyección) y las figuras.
2. **Estimador**: Se ingresa manualmente el precio esperado del insumo clave para obtener
   la estimación de costo con el modelo interpretable (ElasticNet sobre cambios).
3. **Agente**: chat conversacional con el agente LangGraph (Se necesita una ANTHROPIC_API_KEY
   y el extra ``agent``). Tiene una búsqueda web que es opt-in para controlar el costo.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Asegura que el paquete (src/) sea importable al lanzar con streamlit run
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from cost_forecast.agent import tools  # noqa: E402
from cost_forecast.config import load_config  # noqa: E402
from cost_forecast.forecast.scenario import (  # noqa: E402
    estimate_from_levels,
    input_bounds,
)

st.set_page_config(page_title="Costos de equipos — IA", layout="wide")
CFG = load_config()
ART = CFG.path("artifacts_dir")
FIG = CFG.path("figures_dir")


def _load(name: str) -> dict:
    path = ART / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# Barra lateral
st.sidebar.title("⚙️ Configuración")
model = CFG["agent"]["model"]
st.sidebar.caption(f"Modelo del agente: **{model}**")
has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
if has_key:
    st.sidebar.success("ANTHROPIC_API_KEY")
else:
    st.sidebar.warning("Modelo demo sin LLM")
web_search = st.sidebar.toggle(
    "Habilitar búsqueda web",
    value=bool(CFG["agent"].get("web_search", False)),
    help="Activa la búsqueda web para que el agente complemente su respuesta con contexto de mercado.",
)

st.title("📊 Estimación y Proyección de Costos")

tab_res, tab_whatif, tab_agent = st.tabs(["Resultados del Modelo", "Estimador", "Agente de IA"])

# Pestaña 1: Resultados
with tab_res:
    selection = _load("selection.json")
    evaluation = _load("evaluation.json")
    forecast = _load("forecast.json")

    if not selection:
        st.info("Aún no hay artefactos. Ejecuta `make all` para generarlos.")
    else:
        st.subheader("Selección de variables por votación (por cada equipo)")
        cols = st.columns(2)
        for col, (tgt, data) in zip(cols, selection.items(), strict=False):
            with col:
                st.metric(f"{tgt} → insumo seleccionado", ", ".join(data["selected"]) or "—")
                st.caption(f"Consenso: {data['consensus']} (umbral {data['threshold']})")
                fig = FIG / f"voting_{tgt}.png"
                if fig.exists():
                    st.image(str(fig))

        st.divider()
        st.subheader("Desempeño de modelos (test + validation)")
        for tgt, d in evaluation.get("targets", {}).items():
            st.markdown(f"**{tgt}** — mejor modelo: `{d['best_model']}`")
            rows = {m: {k: v for k, v in mm["error"].items()} for m, mm in d["metrics"].items()}
            st.dataframe(rows, use_container_width=True)
            fig = FIG / f"real_vs_pred_{tgt}.png"
            if fig.exists():
                st.image(str(fig))

        st.divider()
        st.subheader("Proyección de costos con banda 90%")
        cols = st.columns(2)
        for col, (tgt, d) in zip(cols, forecast.items(), strict=False):
            with col:
                st.markdown(f"**{tgt}** — modelo `{d.get('model', '?')}`")
                fig = FIG / f"forecast_{tgt}.png"
                if fig.exists():
                    st.image(str(fig))
                st.dataframe(d["forecast"], use_container_width=True)

# Pestaña 2: Estimador
with tab_whatif:
    st.subheader("Estimador de Costos")
    st.caption(
        "Ingresa el precio esperado de la materia prima determinante de cada equipo y "
        "obtén el costo estimado con el modelo regularizado interpretable, anclado en el "
        "último valor observado."
    )
    if not _load("selection.json"):
        st.info("Se debe ejecutar `make all` para habilitar el estimador.")
    else:
        conf = st.select_slider(
            "Nivel de confianza de la banda",
            options=[0.80, 0.90, 0.95, 0.99],
            value=0.90,
        )
        cols = st.columns(2)
        for col, tgt in zip(cols, CFG["schema"]["targets"], strict=False):
            with col:
                sel, bounds = input_bounds(CFG, tgt)
                st.markdown(f"### {tgt}  ·  insumo **{', '.join(sel)}**")
                entered = {}
                for c in sel:
                    b = bounds[c]
                    rng = b["max"] - b["min"]
                    entered[c] = st.number_input(
                        f"Precio de {c} (rango observado {b['min']:.0f}–{b['max']:.0f}, "
                        f"último {b['last']:.2f})",
                        min_value=float(round(b["min"], 2)),
                        max_value=float(round(b["max"], 2)),
                        value=float(round(b["last"], 2)),
                        step=float(round(max(rng / 200, 0.01), 2)),
                        key=f"in_{tgt}_{c}",
                    )
                r = estimate_from_levels(CFG, tgt, entered, confidence=conf)
                delta = r.predicted_level - r.last_equipo_level
                st.metric(
                    f"Costo estimado {tgt}",
                    f"{r.predicted_level:,.2f}",
                    delta=f"{delta:+.2f} vs último ({r.last_equipo_level:,.2f})",
                )
                st.caption(
                    f"Banda {int(conf * 100)}%: [{r.lower:,.2f} – {r.upper:,.2f}]  ·  "
                    f"sensibilidad: +1 en {sel[0]} ⇒ {r.sensitivities[sel[0]]:+.3f} en {tgt}"
                )
        st.info(
            "El estimador calcula los costos basándose en el precio que se elija para cada insumo (usando el modelo regularizado interpretable)."
            " Mientras que la proyección automática a 6 meses (de la pestaña # 1) "
            "selecciona el mejor modelo predictivo para cada equipo y permite que el precio del insumo varíe libremente en el tiempo."
        )

# Pestaña 3: Agente
with tab_agent:
    st.subheader("Consultar al agente sobre los resultados")
    st.caption(
        "El agente lee los artefactos del modelo como herramientas (RAG) y, si se activa la busqueda web, complementa con contexto de mercado."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input("Ej: ¿qué insumo explica Equipo1 y cuál es su proyección a 6 meses?")
    if question:
        st.session_state.history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if has_key:
                try:
                    from cost_forecast.agent.graph import ask, build_agent

                    if "agent" not in st.session_state or st.session_state.get("ws") != web_search:
                        st.session_state.agent = build_agent(include_web_search=web_search)
                        st.session_state.ws = web_search
                    with st.spinner("Consultando…"):
                        answer = ask(st.session_state.agent, question)
                except Exception as exc:  # noqa: BLE001
                    answer = (
                        f"No se pudo invocar el agente ({exc}). "
                        "Instala el extra: `uv sync --extra agent`."
                    )
            else:
                # Modo demo sin LLM: responde desde los artefactos.
                answer = (
                    "**Modo demo (sin API key).** Datos del modelo:\n\n"
                    f"- Variables seleccionadas: `{tools.get_selected_variables()}`\n"
                    f"- Proyección Equipo1: `{tools.get_forecast('Equipo1').get('forecast', '—')[:1]}`\n"
                    f"- Proyección Equipo2: `{tools.get_forecast('Equipo2').get('forecast', '—')[:1]}`\n\n"
                    "Define `ANTHROPIC_API_KEY` para el chat conversacional completo."
                )
            st.markdown(answer)
        st.session_state.history.append(("assistant", answer))
