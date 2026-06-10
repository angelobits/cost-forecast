# IA convencional vs. Agente de IA

## Definiciones

- **IA convencional (modelo predictivo/generativo):** un sistema que, dado un input, produce un output —predice, clasifica o genera— a partir de patrones aprendidos de datos. Es **reactivo y sin objetivo propio**: no decide qué hacer a continuación ni
  interactúa con el entorno. En este proyecto, los modelos **ElasticNet / SARIMAX / GBM** son IA convencional porque reciben los cambios de los insumos y devuelven el costo proyectado.

- **Agente de IA:** un sistema **autónomo** que **percibe** su entorno, **decide** qué acción tomar y **ejecuta** acciones usando herramientas para alcanzar un **objetivo**, iterando hasta lograrlo. En este proyecto, el **agente LangGraph + API de Anthropic** recibe una pregunta, decide qué herramienta utilizar (leer métricas, leer el pronóstico, o buscar contexto de mercado en la web), ejecuta esas acciones y compone una respuesta fundamentada.

## Dimensiones clave

| Dimensión | IA convencional (modelos del proyecto) | Agente de IA (agente LangGraph del proyecto) |
|---|---|---|
| **Autonomía** | Ninguna: una llamada → una salida. | Decide el plan, qué herramientas usar y en qué orden, hasta responder. |
| **Uso de herramientas** | No usa herramientas. | Para este caso de usa `get_selected_variables`, `get_metrics`, `get_forecast`, `get_eda_summary` (RAG sobre artefactos) y `web_search` (contexto externo). |
| **Memoria** | Sin memoria entre llamadas. | Memoria de la conversación (estado del grafo) y acceso persistente a los artefactos del modelo. |
| **Capacidad de acción** | Calcula un número. | Actúa, consulta resultados, busca noticias/tendencias del sector y **combina** modelo propio + conocimiento externo. |

## ¿Cómo se materializa en el proyecto?

```
Alguien Pregunta
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