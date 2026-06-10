# IA convencional vs. Agente de IA

## Definiciones

- **IA convencional (modelo predictivo/generativo):** un sistema que, dado un input, produce un output —predice, clasifica o genera— a partir de patrones aprendidos de datos. Es **reactivo y sin objetivo propio**: no decide qué hacer a continuación ni
  interactúa con el entorno. En este proyecto, los modelos **ElasticNet / SARIMAX / GBM** son IA convencional: reciben los cambios de los insumos y devuelven el costo proyectado.

- **Agente de IA:** un sistema **autónomo** que **percibe** su entorno, **decide** qué acción tomar y **ejecuta** acciones (usando herramientas) para alcanzar un **objetivo**, iterando hasta lograrlo. En este proyecto, el **agente LangGraph + API de Anthropic** recibe una pregunta del evaluador, decide qué herramienta invocar (leer métricas, leer el pronóstico, o buscar contexto de mercado en la web), ejecuta esas acciones y compone una respuesta fundamentada.

## Las cuatro dimensiones clave

| Dimensión | IA convencional (nuestros modelos) | Agente de IA (nuestro agente LangGraph) |
|---|---|---|
| **Autonomía** | Ninguna: una llamada → una salida. | Decide el plan: qué herramientas usar y en qué orden, hasta responder. |
| **Uso de herramientas** | No usa herramientas. | Usa `get_selected_variables`, `get_metrics`, `get_forecast`, `get_eda_summary` (RAG sobre artefactos) y `web_search` (contexto externo). |
| **Memoria** | Sin memoria entre llamadas. | Memoria de la conversación (estado del grafo) y acceso persistente a los artefactos del modelo. |
| **Capacidad de acción** | Calcula un número. | Actúa: consulta resultados, busca noticias/tendencias del sector y **combina** modelo propio + conocimiento externo. |

## Cómo se materializa en este entregable

```
Pregunta del evaluador
        │
        ▼
  ┌───────────────┐   percibe   ┌───────────────────────────┐
  │  Agente       │◀───────────│ Artefactos del modelo      │
  │  (LangGraph + │            │ (selection/metrics/forecast)│
  │   Anthropic)  │   decide    └───────────────────────────┘
  │               │─────────────▶  Herramienta: web_search (mercado)
  └───────────────┘   ejecuta
        │ responde combinando modelo + contexto
        ▼
  Respuesta fundamentada (distingue dato del modelo vs. fuente externa)
```

- El **modelo** responde *“¿cuánto costará Equipo2 en 6 meses?”* con un número y su banda.
- El **agente** responde *“¿el pronóstico de Equipo2 es consistente con el mercado del insumo Z?”*: lee el pronóstico (herramienta interna), busca tendencias del sector (herramienta web), razona sobre ambos y entrega una conclusión accionable — algo que un modelo aislado no puede hacer.

> Implementación: `src/cost_forecast/agent/` (tools.py, graph.py, cli.py). Sin
> `ANTHROPIC_API_KEY`, el agente degrada a un **modo demo** que consulta directamente los
> artefactos, demostrando el componente RAG sin coste de API.