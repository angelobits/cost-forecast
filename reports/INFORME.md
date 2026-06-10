# Informe

**Caso:** Gestión de costos operativos a partir de precios de materias primas (X, Y, Z).
**Equipos objetivo:** Equipo1, Equipo2 · **Datos:** diarios 2010-01-04 → 2023-08-31 (3.530 obs.).
---

## 1) Explicación del caso

Una constructora planifica un proyecto con una ventana de ejecución definida y debe abastecer dos equipos críticos (Equipo1, Equipo2) cuyo costo de adquisición ha sido volátil y difícil de anticipar, generando **desviaciones presupuestales**. La gerencia sospecha que esos costos dependen de la dinámica de ciertas materias primas (X, Y, Z) pero carece de un modelo formal y de claridad sobre qué insumo determina el costo de cada equipo.

Las tareas que se deben desarrollan son: (a) identificar, con evidencia, qué insumos explican cada equipo; (b) construir una metodología **reproducible** para estimar el costo, y (c) **proyectar** el costo hacia el futuro con un horizonte justificado e incertidumbre cuantificada, para alimentar la planeación financiera, además, se definen ciertos entregables adicionales como un agente de IA para exponer resultados, propuesta de arquitectura cloud y un informe.

---

## 2) Supuestos

1. **Fuente de datos:** el consolidado `historico_equipos.csv` se utiliza para el modelado, mientras que las series individuales `X/Y/Z.csv` se usan para ingeniería de datos, reconciliación y validación.
2. **Objetivo de negocio:** ¿cuánto costará el equipo? (lo que en el desarrollo del ejercicio llamó NIVEL), no solo la dirección del cambio.
3. **No estacionariedad en niveles**, se modelo sobre **cambios (d1)** estacionarios y se **reconstruye el nivel**, propagando la incertidumbre.
4. **Modelos independientes por equipo** selección, VIF y modelado separados para cada equipo.
5. **Frecuencia de trabajo:** para el pronóstico (la planeación financiera mensual), el nivel mensual que se tomó es el **nivel de fin de mes** (último precio observado en cada mes), no el promedio de cambios diarios pero se debe tener en cuenta para el EDA y la selección de variable si se utilizó una granularidad **diaria**.
6. **Relación contemporánea:** la correlación cruzada con rezagos sitúa el mejor lag en **0–1 días**, por lo que a escala mensual el efecto insumo→equipo es contemporáneo.
7. **Semilla fija (42)** y configuración externalizada (`config/config.yaml`): sin números ni rutas mágicas en el código.
8. **Las relaciones insumo→equipo se sustentan en el análisis de datos** y son reproducibles.

---

## 3) Alternativas para resolver el caso

**Decisión central — niveles vs. cambios.** Caben tres representaciones:

| Opción | Idea | Riesgo / ventaja |
|---|---|---|
| A. Regresión en **niveles** | correlacionar precios directamente | Correlaciones ~0.99 → **regresión espuria** por tendencias comunes. ADF no rechaza raíz unitaria (p: X 0.43, Y 0.11, Z 0.21, Eq1 0.14, Eq2 0.33). |
| B. Modelar **cambios (d1)** y reconstruir nivel | series estacionarias (ADF p≈0; KPSS no rechaza) | Relaciones genuinas, requiere reconstruir el nivel y propagar incertidumbre. **(Elegida)** |
| C. **Cointegración / ECM-VECM** en niveles | usar relación de largo plazo si existe | Engle-Granger da cointegración Eq1~Y (p=0.013), Eq2~Z (p=0.016); Johansen sobreestima (5 relaciones) por tendencias comunes. Se usa como **respaldo**, no como modelo único. |

**Alternativa utilizada:** **B con respaldo de C**. Se entrenó sobre cambios mensuales (estacionarios) y se reconstruyó el nivel acumulando cambios sobre el último nivel observado, la cointegración confirma que la asociación Eq1–Y y Eq2–Z es de largo plazo (no espuria). Para el contraste de familias se incluyó **SARIMAX con exógenas en nivel** (integra d=1 internamente), que para
Equipo1 resultó el mejor modelo.

**Selección de variables — votación por equipo (≥6 métodos).** Un insumo entra al modelo de un equipo solo si lo votan ≥ **60%** de los métodos, se utilizaron: correlación con lags, ElasticNetCV, importancia RandomForest, importancia GBM, OLS por p-valor y mutual information.

**Comparación de modelos:** ElasticNet (interpretable) vs. SARIMAX (serie de tiempo con exógenas) vs. GBM (ML), con **media móvil de 3 meses solo como baseline**. Split temporal sin fuga (TRAIN/TEST/VALIDATION) y `TimeSeriesSplit` para walk-forward.

**Proyección:** Monte Carlo (1.000 trayectorias) que propaga la incertidumbre del **insumo** (SARIMA simulado) y del **modelo** (residual), produciendo bandas de predicción al 90%.

---

## 4) Resultados

### 4.1 Ingeniería de datos y EDA

- **Parsers por archivo:** X (ISO, orden descendente), Y (BOM + `;` + decimal `,` + fecha D/M/Y), Z (columnas invertidas). Normalizados a series ordenadas y de-duplicadas.
- **Reconciliación:** 100% match dentro de tolerancia; diff máx = 0.0, las tres series extienden el histórico → el consolidado es consistente y los individuales sirven para validar/extender.
- **Estacionariedad (ADF+KPSS):** todas las series son **no estacionarias en nivel** y **estacionarias en d1** (ADF p≈0, KPSS no rechaza). ⇒ se modela en cambios.
- **Correlaciones:** en **niveles** Eq1↔Y = 0.997 y Eq2↔Z = 0.983 (espurias), en **diferencias** caen a **Eq1↔Y = 0.386** y **Eq2↔Z = 0.431**, y el resto se desploma (Eq1↔Z = 0.024, Eq2↔X = 0.085), la brecha niveles vs diferencias es uno de los puntos claves en el análisis del caso.
- **Correlación parcial:** la correlación en nivel de **Z con Equipo1 (0.844)** colapsa a **0.000** al controlar por Y → es **espuria, inducida por Y**, la de Y se mantiene (0.990). Confirma, con álgebra de residuales, que Equipo1 responde a Y y no a Z.
- **Correlación cruzada con rezagos:** mejor lag en **0–1 días** (efecto contemporáneo).
- **Cointegración:** Eq1~Y y Eq1~Z, Eq2~Y y Eq2~Z cointegran; **X no cointegra** con ningún equipo.
- **Outliers** (|z|>4 sobre retornos): X 19, Y 51, Z 13, Eq1 3, Eq2 2 → conservados (volatilidad real).

### 4.2 Selección de variables por votación

**Matriz de votación (1 = vota a favor):**

| Equipo1 | X | Y | Z | | Equipo2 | X | Y | Z |
|---|---|---|---|---|---|---|---|---|
| corr_lag | 0 | 1 | 0 | | corr_lag | 0 | 0 | 1 |
| elasticnet | 1 | 1 | 1 | | elasticnet | 1 | 1 | 1 |
| rf_importance | 1 | 0 | 0 | | rf_importance | 0 | 0 | 1 |
| gbm_importance | 0 | 1 | 0 | | gbm_importance | 0 | 0 | 1 |
| ols_pvalue | 0 | 1 | 0 | | ols_pvalue | 0 | 1 | 1 |
| mutual_info | 0 | 1 | 0 | | mutual_info | 0 | 0 | 1 |
| **Consenso** | 0.33 | **0.83** | 0.17 | | **Consenso** | 0.17 | 0.33 | **1.00** |

➡ **Equipo1 → {Y}** (consenso 0.83), **Equipo2 → {Z}** (consenso 1.00), X termina siendo ruido para ambos equipos, la selección es unánime entre métodos para Equipo2 y mayoritaria para Equipo1.

### 4.3 Ventana temporal

- **Rango histórico de entrenamiento:** 2010-01 → 2021-12 (**144 meses, ~88%**). Con ello se cubre múltiples ciclos de precio y es suficiente para estimar la dinámica, no se recorta por quiebres porque las series en d1 son estacionarias y los modelos en cambios son robustos a cambios de nivel. Para **TEST** se utilizó 2022 (12 meses, ~7%) y para **VALIDATION** 2023-01→08 (8 meses, ~5%), ambos están fuera de muestra y posteriores en el tiempo.
- **Frecuencia de trabajo:** para el pronóstico se utilizó el supuesto que la planeación financiera se ejecuta mes a mes, mientras que para el EDA se trabajó con los registros diarios para mayor potencia estadística.
- **Horizonte de pronóstico:** El horizonte utilizado es de **6 meses** porque coincide con una ventana típica de planeación/aprovisionamiento financiero por fase de una obra, además, que más allá de 6 meses la banda de predicción se ensancha mucho y el pronóstico puntual pierde valor. Dentro del proyecto se dejó está ventana temporal parametrizable en `config.yaml` (`forecast_horizon_months`).

### 4.4 VIF / Multicolinealidad

Como resultado del proceso de selección por consenso, a cada equipo se le asignó una única variable explicativa ($Y$ para Equipo 1 y $Z$ para Equipo 2). Al contar con un solo predictor por modelo, el Factor de Inflación de Varianza (VIF) es idéntico a 1.0, lo que garantiza la ausencia absoluta de multicolinealidad. En el pipeline incluye un control automático con un umbral de tolerancia de 10, en caso de que futuras actualizaciones incorporen dos o más insumos por equipo, el sistema activararía automáticamente la evaluación de VIF y mitigará la redundancia mediante la eliminación de la variable con mayor indicador o apoyándose en la regularización ElasticNet que ya forma parte de la arquitectura.

### 4.5 Modelado y evaluación (real vs. predicho, test+validation)

| Equipo | Modelo | MAE | RMSE | MAPE % | R² | MASE | Dir. acc. | F1 dir. |
|---|---|---|---|---|---|---|---|---|
| **Equipo1** | **SARIMAX** (Seleccionado) | 8.58 | 11.23 | **1.67** | 0.983 | 0.378 | **1.00** | 1.00 |
| Equipo1 | ElasticNet | 11.83 | 13.64 | 2.30 | 0.975 | 0.522 | 0.95 | 0.94 |
| Equipo1 | GBM | 14.42 | 17.89 | 2.79 | 0.957 | 0.636 | 0.95 | 0.94 |
| Equipo1 | Baseline MM(3) | 49.54 | 59.18 | 9.59 | 0.527 | 2.186 | 0.47 | 0.44 |
| **Equipo2** | **ElasticNet** (Seleccionado) | 29.75 | 35.56 | **2.80** | 0.959 | 0.772 | 0.84 | 0.82 |
| Equipo2 | GBM | 37.16 | 42.36 | 3.42 | 0.941 | 0.964 | 0.79 | 0.75 |
| Equipo2 | SARIMAX | 37.13 | 45.90 | 3.57 | 0.931 | 0.963 | 0.79 | 0.78 |
| Equipo2 | Baseline MM(3) | 97.42 | 125.41 | 8.55 | 0.484 | 2.527 | 0.58 | 0.56 |

- **Lectura:** los modelos con insumos seleccionados baten al baseline por **~3–5×** en MAPE y pasan de una *dirección al azar* (acc≈0.5) a **0.84–1.00**. MASE < 1 en todos los modelos serios (mejores que el naïve), y ≈2.2–2.5 en la media móvil (peor que el naïve). Esto **demuestra cuantitativamente** que una media móvil simple sería insuficiente como método único.
- **Coeficientes interpretables (ElasticNet, sobre cambios):** Eq1: ΔEquipo1 ≈ **+27.95·ΔY**; Eq2: ΔEquipo2 ≈ **+41.99·ΔZ**. Signo positivo y económicamente coherente (sube el insumo → sube el costo del equipo).
- **Desempeño por split (test vs. validación, fuera de muestra; campo `by_split` en `evaluation.json`):** Equipo1/SARIMAX — test MAPE 1.67% (dir. 1.00) y validación 1.67% (dir. 1.00); Equipo2/ElasticNet — test 2.67% (dir. 0.82) y validación 2.98% (dir. 0.86). La **consistencia test↔validación** indica ausencia de sobreajuste. Matrices de confusión en el artefacto.
- **Diagnóstico de residuales** (mejor modelo por equipo): **sin autocorrelación** (Ljung-Box p=0.49 / 0.62), **normales** (Jarque-Bera p=0.50 / 0.66) y **homocedásticos** (Breusch-Pagan p=0.97 / 0.54). Los residuales se comportan como ruido blanco → especificación adecuada.
- Figuras: `reports/figures/real_vs_pred_*.png`, `voting_*.png`, `series_levels.png`, `crosscorr_*.png`.

---

## 5) Proyección de costos y horizonte de predicción

La proyección se configuró para un horizonte de 6 meses (comprendido entre septiembre de 2023 y febrero de 2024), empleando una simulación de Monte Carlo con 1.000 trayectorias para generar bandas de predicción al 90% de confianza, este enfoque metodológico propaga de forma conjunta dos fuentes críticas de incertidumbre, primero la del insumo exógeno (mediante la simulación del comportamiento de la materia prima a través de un modelo SARIMA) y segundo la del propio modelo de estimación (a través de la variabilidad de sus residuales).

Como punto de partida para la reconstrucción de los niveles, se tomaron los últimos valores reales observados al 31 de agosto de 2023: Equipo 1 = 451.73 y Equipo 2 = 955.35.

**Equipo1** (impulsado por Y):

| Mes | Pred | Inferior 90% | Superior 90% |
|---|---|---|---|
| 2023-09 | 440.6 | 392.1 | 490.1 |
| 2023-10 | 441.4 | 360.0 | 523.4 |
| 2023-11 | 441.7 | 337.3 | 540.6 |
| 2023-12 | 442.7 | 326.0 | 556.6 |
| 2024-01 | 441.2 | 300.6 | 573.8 |
| 2024-02 | 440.8 | 290.1 | 589.1 |

**Equipo2** (impulsado por Z):

| Mes | Pred | Inferior 90% | Superior 90% |
|---|---|---|---|
| 2023-09 | 953.0 | 863.9 | 1043.0 |
| 2023-10 | 950.2 | 823.7 | 1082.8 |
| 2023-11 | 952.7 | 789.5 | 1111.0 |
| 2023-12 | 953.0 | 774.9 | 1128.5 |
| 2024-01 | 950.1 | 739.6 | 1153.9 |
| 2024-02 | 950.5 | 713.8 | 1187.0 |

Los resultados apuntan a que el costo esperado para ambos equipos se mantendrá estable en el corto plazo, sin mostrar una tendencia al alza o a la baja estadísticamente significativa. Sin embargo, debido a la naturaleza estocástica del modelo, la banda de incertidumbre se ensancha de forma monótona a medida que se extiende el horizonte de proyección.

Para una gestión presupuestaria conservadora y la mitigación de riesgos, se recomienda planificar los flujos de caja utilizando el límite superior de la banda como un escenario de estrés. Por ejemplo, con un 90% de confianza, los costos máximos estimados para el cierre del horizonte (febrero de 2024) se ubican en $\le 589.1$ para el Equipo 1 y $\le 1187.0$ para el Equipo 2. Los gráficos correspondientes a estas proyecciones se encuentran consolidados en la ruta reports/figures/forecast_*.png.

Un punto a mencionar es que debido a que los modelos matemáticos operan sobre la primera diferencia de las series para garantizar estacionariedad, el nivel absoluto se reconstruye de forma acumulativa para cada una de las 1.000 trayectorias mediante la siguiente ecuación:

> $$nivel_t = nivel_{t0} + \sum_{i=1}^{t} \Delta cambios\_simulados_i$$

---

## 6) Mejoras o ajustes

1. **Modelado de equilibrio a largo plazo (VECM/ECM):** Se puede aprovechar las relaciones históricas estables encontradas entre los equipos y sus insumos clave, lo que permitiría crear un modelo de corrección de errores que entienda no solo los cambios del día a día, sino cómo los precios tienden a equilibrarse en el largo plazo.

2. **Pronóstico avanzado de materias primas:** Claramente al ser un caso hipotético, la opción de sustituir las simulaciones estadísticas actuales de los insumos ($X, Y, Z$) por variables reales del mercado, tales como indicadores macroeconómicos o precios de contratos de futuros ayudaría a aterrizar mejor las proyecciones y a reducir la incertidumbre en horizontes lejanos.

3. **Análisis de efectos retardados y relaciones no lineales:** Se podría expandir el sistema de votación actual para que evalúe el impacto de los precios de meses anteriores y detecte patrones complejos o no lineales, utilizando metodologías avanzadas de selección de variables (como pruebas de causalidad de Granger).

4. **Backtesting**: Implementar un esquema de evaluación continua con reentrenamiento mensual, además, se plantea la opción de evolucionar las bandas de Monte Carlo hacia intervalos conformales que es una técnica estadística moderna que asegura matemáticamente que los costos reales caigan dentro del rango previsto.

5. **Optimización del impacto presupuestal:** Una mejora interesante para el resultado del negocio, sería la de traducir las bandas de incertidumbre técnica directamente en métricas financieras explicables, que puedan ayudar a la gerencia o equipo financiero a calcular con precisión matemática cuál debería ser la reserva de contingencia óptima para cada proyecto.

---

## 7) Apreciaciones y comentarios del caso

- En mi opinión el principal aporte de este proyecto radica en la rigurosidad para descubrir las verdaderas relaciones del mercado, el marco metodológico implementado (votación por consenso, análisis de estacionariedad, correlación parcial y pruebas de cointegración) demuestran con evidencia estadística concluyente que el costo del Equipo 1 está determinado por la materia prima $Y$, mientras que el del Equipo 2 responde a la dinámica de $Z$. Si hubiese asumido relaciones de forma intuitiva o visual, sin este respaldo, habría terminado en un modelo deficiente y en un escenario real en errores graves de planificación.

- El punto de evaluar niveles frente a cambios es un punto clave, porque las correlaciones aparentes de $0.99$ al analizar los datos en niveles terminaron siendo un "espejismo" provocado por tendencias crecientes comunes en el tiempo (correlación espuria). Pero al modelar en cambios (primeras diferencias), la señal verdaderá se evidencia en un rango real de $0.39$ a $0.43$. Además, la correlación parcial actuó como un filtro definitivo, confirmando que la aparente relación entre $Z$ y el Equipo 1 era falsa e inducida matemáticamente por el efecto de $Y$.

- Dado que en contextos de alta volatilidad como lo es el caso planteado, un pronóstico puntual podría perder valor rápidamente, el principal entregable no es una proyección única, sino la banda de predicción calculada por Monte Carlo, donde se reconoce y cuantifica la incertidumbre con un 90% de confianza, permitiendo a la empresa anticipar los riesgos de forma honesta, fundamentando la recomendación práctica de presupuestar sobre el límite superior para blindar financieramente los proyectos de construcción.
