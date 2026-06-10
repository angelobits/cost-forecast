# Estimación y proyección de costos de equipos de construcción a partir de precios de materias primas

**Caso de consultoría**
Autor: Angelo Parra Cortez

---

## 1. Resumen

Este proyecto estima y proyecta el costo de adquisición de dos equipos de construcción (Equipo1 y Equipo2) a partir del comportamiento histórico de tres materias primas (X, Y, Z). El objetivo es el de identificar con evidencia estadística, qué insumos determinan el costo de cada equipo, y construir un mecanismo reproducible para anticipar dichos costos con un horizonte justificado y una cuantificación explícita de la incertidumbre.

El análisis concluye que el costo de Equipo1 se explica por la materia prima Y y el de Equipo2 por la Z, mientras que X constituye ruido. Estas relaciones se derivan de un procedimiento de selección por consenso (seis métodos), se confirman mediante correlación parcial y pruebas de cointegración, y se sostienen al modelar en cambios para evitar la correlación espuria que aparece al trabajar en niveles. 

Los modelos seleccionados alcanzan un error porcentual medio (MAPE) de 1.7 % para Equipo1 (SARIMAX) y 2.8 % para Equipo2 (ElasticNet) sobre datos fuera de muestra, frente a aproximadamente 9 % de una media móvil utilizada como línea base.

La solución se entrega como un paquete de Python modular y reproducible, acompañado de tres interfaces de consumo, una aplicación web (Streamlit), una API REST (FastAPI) y un agente conversacional (LangGraph con la API de Anthropic).

---

## 2. Planteamiento del problema

### 2.1 Contexto

Una empresa constructora en fase de planificación debe abastecer de forma continua dos equipos críticos. Históricamente, el costo de estos equipos ha mostrado un comportamiento variable que la empresa no ha logrado anticipar, lo que ha producido desviaciones presupuestales recurrentes. La gerencia presume una relación con la dinámica de ciertos insumos del mercado de materias primas, pero carece de un modelo formal que la respalde y de claridad sobre qué insumo es determinante para cada equipo.

### 2.2 Datos disponibles

Se dispone de cuatro archivos en `data/input/`:

| Archivo | Contenido | Rango | Observaciones |
|---|---|---|---|
| `historico_equipos.csv` | Consolidado: fecha, X, Y, Z, Equipo1, Equipo2 | 2010-01-04 a 2023-08-31 | Diario, alineado, sin valores faltantes (3.530 filas). |
| `X.csv` | Serie individual de X | 1988 a 2024 | Formato ISO, orden descendente. |
| `Y.csv` | Serie individual de Y | 2006 a 2023 | Separador `;`, decimal con coma, fecha D/M/Y, BOM. |
| `Z.csv` | Serie individual de Z | 2010 a 2023 | Columnas invertidas (precio, fecha). |

El consolidado es la fuente utilizada para el modelado por estar alineado en una misma temporalidad, mientras que las series individuales, con formatos heterogéneos deliberados, se utilizan para demostrar la ingeniería de datos, reconciliar contra el consolidado y validar el histórico.

### 2.3 Objetivo

1. Determinar, a partir de los datos, qué materia prima explica el costo de cada equipo.
2. Construir una metodología reproducible para estimar el costo de forma sistemática.
3. Proyectar el costo esperado a un horizonte justificado, con intervalos de confianza.
4. Exponer los resultados a través de un agente de IA y documentar una arquitectura cloud.

---

## 3. Metodología

### 3.1 Ingeniería de datos

Se implementó un parser específico por archivo que normaliza cada formato a una serie ordenada. Luego, cada serie individual se reconcilia contra su columna en el consolidado, la coincidencia es del 100 % dentro de tolerancia, con diferencia máxima nula, y las tres series individuales extienden el histórico, lo que confirma la consistencia del consolidado.

### 3.2 Análisis exploratorio y estacionariedad

Se aplicaron las pruebas de Dickey-Fuller aumentada (ADF) y KPSS, complementarias entre sí. Todas las series resultan no estacionarias en nivel y estacionarias en su primera diferencia. Se analizaron además correlaciones (en niveles y en diferencias), correlación cruzada con rezagos, correlación parcial, cointegración (Engle-Granger y Johansen) y valores atípicos.

### 3.3 Representación: niveles frente a cambios

En niveles, las correlaciones entre insumos y equipos son cercanas a 0.99, pero son espurias (provienen de una tendencia creciente común). Al modelar en cambios (primeras diferencias), la señal genuina emerge y el resto se desvanece. La correlación parcial confirma este punto, la correlación de Z con Equipo1 cae de 0.84 a 0.00 al descontar el efecto de Y, mientras que la de Y se mantiene. Por ello se modela sobre cambios estacionarios y se reconstruye el nivel acumulando los cambios sobre el último valor observado, que es lo que el negocio necesita.

### 3.4 Selección de variables por votación

Para decidir qué insumo entra al modelo de cada equipo se emplearon seis métodos independientes (correlación con rezagos, ElasticNet, importancia de Random Forest, importancia de Gradient Boosting, significancia OLS por p-valor e información mutua). Cada método emite un voto binario y un insumo se selecciona si lo respalda al menos el 60 % de los métodos. El resultado fue Equipo1 con Y (consenso 0.83) y Equipo2 con Z (consenso 1.00).

### 3.5 Ventana temporal

El entrenamiento abarca 2010-2021 (88 % de los datos), la prueba 2022 (7 %) y la validación 2023 (5 %), respetando el orden temporal para evitar fuga de información. La frecuencia de trabajo del pronóstico es mensual, consolidando cada mes con el nivel de fin de mes. El horizonte de proyección es de seis meses, justificado por el ciclo de aprovisionamiento del proyecto, por la extensión de la validación disponible y por el ensanchamiento de la banda de incertidumbre más allá de ese punto.

### 3.6 Modelado y evaluación

Se compararon tres familias de modelos —regresión regularizada (ElasticNet), series de tiempo con variables exógenas (SARIMAX) y aprendizaje automático (Gradient Boosting)— contra una media móvil de tres meses usada exclusivamente como línea base. El mejor modelo por equipo se eligió por menor RMSE fuera de muestra. La evaluación reporta métricas de error (MAE, RMSE, MAPE, R², MASE) y métricas direccionales sobre el signo del cambio (precisión, recall, F1 y matriz de confusión), desglosadas por prueba y validación, junto con el diagnóstico de residuales del mejor modelo.

### 3.7 Proyección con incertidumbre

La proyección se realiza por simulación de Monte Carlo (1.000 trayectorias) que propaga la incertidumbre del insumo y la del modelo. La banda de predicción corresponde a los percentiles 5 y 95 (90 % de confianza). Como el modelado es en cambios, cada trayectoria reconstruye el nivel a partir del último valor observado.

---

## 4. Arquitectura del software

### 4.1 Estructura del proyecto

```
cost-forecast/
├── src/cost_forecast/
│   ├── config.py             # Carga de configuración tipada desde config.yaml
│   ├── logging_utils.py      # Logging estructurado y semilla global
│   ├── pipeline.py           # Orquestador de etapas (data → forecast)
│   ├── cli.py                # Interfaz de línea de comandos
│   ├── data/
│   │   ├── loaders.py        # Parsers por archivo (X, Y, Z y consolidado)
│   │   └── reconcile.py      # Reconciliación de series individuales vs. consolidado
│   ├── features/
│   │   ├── transforms.py     # Diferencias, retornos, lags y agregación mensual
│   │   ├── stationarity.py   # Pruebas ADF y KPSS
│   │   ├── relationships.py  # Correlación cruzada con lags, parcial y cointegración
│   │   ├── selection.py      # Selección de variables por votación (seis métodos)
│   │   └── vif.py            # Factor de inflación de varianza (multicolinealidad)
│   ├── models/
│   │   ├── base.py           # Contrato común y reconstrucción de nivel
│   │   ├── baselines.py      # Media móvil (línea base)
│   │   ├── regression.py     # ElasticNet sobre cambios
│   │   ├── sarimax_model.py  # SARIMAX con variables exógenas
│   │   └── gbm.py            # Gradient Boosting
│   ├── forecast/
│   │   ├── project.py        # Proyección con bandas (Monte Carlo)
│   │   └── scenario.py       # Estimador what-if (escenarios de insumo)
│   ├── evaluation/
│   │   ├── split.py          # Partición temporal sin fuga (train/test/validation)
│   │   ├── metrics.py        # Métricas de error y direccionales
│   │   └── diagnostics.py    # Diagnóstico de residuales
│   ├── viz/plots.py          # Gráficos reutilizables
│   ├── agent/
│   │   ├── tools.py          # Herramientas: lectura de artefactos y estimador
│   │   ├── graph.py          # Construcción del agente (LangGraph)
│   │   └── cli.py            # REPL del agente
│   └── api/app.py            # API REST (FastAPI)
├── config/config.yaml        # Parámetros: rutas, umbrales, horizonte y semilla
├── app/streamlit_app.py      # Aplicación web (resultados, estimador y agente)
├── reports/                  # Informe y anexos (arquitectura cloud, IA vs. agente)
├── tests/                    # Pruebas con pytest
├── Dockerfile.api, Dockerfile.ui
├── Makefile                  # Targets: install, all, ui, api, agent, test, lint
├── pyproject.toml            # Dependencias y configuración de herramientas
└── README.md
```

Las figuras y los artefactos (`reports/figures/`, `reports/artifacts/`) y los datos procesados
se generan al ejecutar el pipeline y no se versionan, ya que son reproducibles con `make all`.

### 4.2 Principios y lineamientos de desarrollo

El proyecto sigue principios de bajo acoplamiento y alta cohesión, donde cada módulo cumple una responsabilidad única y se comunica mediante interfaces simples, por ejemplo, la aplicación web, la API y el agente consumen la misma capa de modelado sin duplicar lógica, y el estimador interactivo reutiliza el modelo de regresión ya implementado, lo que facilita el mantenimiento y la escalabilidad-

Los lineamientos de desarrollo aplicados son:

- Disposición `src-layout` y paquete instalable, gestión de dependencias con UV y archivo de bloqueo (`uv.lock`) para entornos reproducibles.
- Configuración externalizada en un único archivo YAML, sin rutas ni umbrales codificados.
- Anotaciones de tipo (type hints) y docstrings en las funciones públicas.
- Registro mediante `logging` en lugar de `print`, y manejo explícito de errores.
- Semilla global fija para garantizar resultados deterministas.
- Estilo y formato verificados con ruff y black, además de pruebas automatizadas con pytest.

---

## 5. Instalación y ejecución

### 5.1 Requisitos

- [UV](https://docs.astral.sh/uv/) como gestor de paquetes y entornos.
- Python 3.11 o superior (gestionado por UV).

### 5.2 Instalación

```bash
make install        # equivale a: uv sync --extra agent --extra ui --extra api --extra dev
```

### 5.3 Ejecución del pipeline

```bash
make all            # ejecuta data, eda, features, train, evaluate y forecast
make test           # ejecuta la batería de pruebas
make lint           # verifica formato y estilo (ruff y black)
```

Las etapas también pueden ejecutarse individualmente (`make data`, `make eda`, `make features`, `make train`, `make evaluate`, `make forecast`). Los resultados se persisten en `reports/artifacts/` (JSON y CSV) y las figuras en `reports/figures/`.

### 5.4 Interfaces

```bash
make ui             # aplicación web (dashboard, estimador y agente)
make api            # API REST en http://localhost:8000/docs
make agent          # agente conversacional en terminal
```

La aplicación Streamlit organiza tres pestañas, 1. resultados del modelo, 2. estimador interactivo (escenarios de precio de insumo) y 3. chat con el agente. La API expone los resultados como microservicio (`/selection`, `/metrics`, `/forecast`, `/estimate`, `/methodology`, `/agent/ask`), lo que desacopla el modelo de cualquier interfaz cliente.

### 5.5 Pruebas rápidas de la API

Con la API corriendo (`make api`), se pueden ejecutar los siguientes comandos para validar los endpoints principales:

```bash
# Estado del servicio
curl -s http://localhost:8000/health | python3 -m json.tool

# Modelo seleccionado y justificación (Equipo1)
curl -s "http://localhost:8000/selection?equipo=Equipo1" | python3 -m json.tool

# Métricas de desempeño fuera de muestra (Equipo2)
curl -s "http://localhost:8000/metrics?equipo=Equipo2" | python3 -m json.tool

# Estimación puntual con escenario de precio de insumo
curl -s -X POST http://localhost:8000/estimate \
  -H "Content-Type: application/json" \
  -d '{"equipo": "Equipo1", "inputs": {"Y": 750}}' | python3 -m json.tool

# Pregunta al agente conversacional
curl -s -X POST http://localhost:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Por qué se eligió SARIMAX para Equipo1?"}' | python3 -m json.tool
```

La documentación interactiva completa (Swagger UI) está disponible en `http://localhost:8000/docs`. El agente utiliza el modelo Claude Haiku 4.5; requiere la variable de entorno `ANTHROPIC_API_KEY` para el modo conversacional, si no se ha definido, opera en un modo de demostración que consulta directamente los artefactos.

---

## 6. Resultados

Relaciones identificadas: Equipo1 con Y, Equipo2 con Z, X es ruido para ambos. La correlación parcial confirma que la asociación aparente de Z con Equipo1 (0.84 en niveles) es espuria, inducida por Y, y se anula al controlar por esta.

Desempeño fuera de muestra del mejor modelo por equipo:

| Equipo | Modelo | MAPE (test / val) | RMSE (test / val) | Precisión direccional (test / val) |
|---|---|---|---|---|
| Equipo1 | SARIMAX | 1.67 % / 1.67 % | 12.08 / 9.83 | 1.00 / 1.00 |
| Equipo2 | ElasticNet | 2.67 % / 2.98 % | 37.29 / 32.79 | 0.82 / 0.86 |

La consistencia entre prueba y validación indica ausencia de sobreajuste. La media móvil utilizada como línea base alcanza un MAPE cercano al 9 % y una precisión direccional próxima al azar, lo que confirma el aporte de los insumos seleccionados. El diagnóstico de residuales del mejor modelo no detecta autocorrelación, indica normalidad y homocedasticidad.

Proyección a seis meses con banda del 90 %: el costo esperado se mantiene estable, con una banda que se ensancha de forma monótona con el horizonte (de aproximadamente ±49 en el primer mes a ±151 en el sexto para Equipo1). Para una planeación conservadora se recomienda presupuestar sobre el límite superior de la banda.

---

## 7. Limitaciones y trabajo futuro

- La proyección autónoma depende de la evolución incierta de los insumos, incorporar fuentes de mercado (futuros, indicadores macroeconómicos) reduciría esa incertidumbre.
- Un modelo de corrección de error (VECM/ECM) que explote la cointegración permitiría modelar el nivel de forma directa.
- Los intervalos conformales ofrecerían una cobertura garantizada como alternativa a las bandas de Monte Carlo.
- En producción conviene monitorear la deriva (drift) y reentrenar cuando el error supere un umbral.
- La consolidación mensual por nivel de fin de mes es un supuesto de negocio revisable con el cliente, se plantea una alternativa por promedio mensual que está implementada.

---

## 8. Documentación complementaria

- `reports/INFORME.md` — informe del caso.
- `reports/IA_vs_agente.md` — diferencia entre IA convencional y un agente de IA.
- `reports/arquitectura_azure.md` — arquitectura cloud propuesta.

---

## 9. Referencias metodológicas

[1] D. A. Dickey y W. A. Fuller, "Distribution of the Estimators for Autoregressive Time Series
with a Unit Root," *Journal of the American Statistical Association*, vol. 74, no. 366, 1979.

[2] D. Kwiatkowski, P. C. B. Phillips, P. Schmidt y Y. Shin, "Testing the null hypothesis of
stationarity against the alternative of a unit root," *Journal of Econometrics*, vol. 54, 1992.

[3] R. F. Engle y C. W. J. Granger, "Co-integration and Error Correction: Representation,
Estimation, and Testing," *Econometrica*, vol. 55, no. 2, 1987.

[4] H. Zou y T. Hastie, "Regularization and Variable Selection via the Elastic Net," *Journal of
the Royal Statistical Society: Series B*, vol. 67, no. 2, 2005.

[5] G. E. P. Box y G. M. Jenkins, *Time Series Analysis: Forecasting and Control*, Holden-Day, 1970.

[6] J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," *Annals of
Statistics*, vol. 29, no. 5, 2001.

[7] Bibliotecas: statsmodels, scikit-learn, pandas, LangGraph y la API de Anthropic.
