# Informe — Estimación y proyección de costos de equipos de construcción

**Caso:** Gestión de costos operativos a partir de precios de materias primas (X, Y, Z).
**Equipos objetivo:** Equipo1, Equipo2 · **Datos:** diarios 2010-01-04 → 2023-08-31 (3.530 obs.).
**Reproducibilidad:** todo el informe se deriva de `make all`; los números provienen de `reports/artifacts/*.json`. Semilla global = 42. Las relaciones insumo→equipo se sustentan en el análisis de los datos (votación por equipo, correlación parcial y cointegración).

---

## 1) Explicación del caso

Una constructora planifica un proyecto con ventana de ejecución definida y debe abastecer dos equipos críticos (Equipo1, Equipo2) cuyo costo de adquisición ha sido volátil y difícil de anticipar, generando **desviaciones presupuestales**. La gerencia sospecha que esos costos dependen de la dinámica de ciertas materias primas (X, Y, Z) pero carece de un modelo
formal y de claridad sobre **qué insumo determina cada equipo**.

El encargo del consultor es: (a) identificar, con evidencia, qué insumos explican cada equipo y cuáles son ruido; (b) construir una metodología **reproducible** para estimar el costo, y (c) **proyectar** el costo hacia el futuro con un horizonte justificado e incertidumbre cuantificada, para alimentar la planeación financiera. Entregables adicionales: agente de IA para exponer resultados, arquitectura cloud e informe.

---

## 2) Supuestos

1. **Fuente de verdad:** el consolidado `historico_equipos.csv` (alineado, sin NaNs). Las series individuales `X/Y/Z.csv` se usan para **ingeniería de datos, reconciliación y validación**. Reconciliación: 100% de coincidencia (máx. diff = 0.0) y las tres **extienden** el histórico (X desde 1988, Y desde 2006, Z desde 2010-01-01).
2. **Objetivo de negocio = NIVEL** (cuánto costará el equipo), no solo la dirección del cambio.
3. **No estacionariedad en niveles**; se modela sobre **cambios (d1)** estacionarios y se **reconstruye el nivel**, propagando incertidumbre.
4. **Modelos independientes por equipo** (selección, VIF y modelado separados).
5. **Frecuencia de trabajo = mensual** para el pronóstico (planeación financiera mensual), el nivel mensual es el **nivel de fin de mes** (último precio observado), no el promedio de cambios diarios. El EDA y la selección usan la granularidad **diaria** (mayor potencia).
6. **Relación contemporánea:** la correlación cruzada con rezagos sitúa el mejor lag en **0–1 días**, por lo que a escala mensual el efecto insumo→equipo es contemporáneo.
7. **Semilla fija (42)** y configuración externalizada (`config/config.yaml`): sin números ni rutas mágicas en el código.
8. **Las relaciones insumo→equipo se sustentan en el análisis de datos** y son reproducibles.

---

## 3) Formas para resolver el caso y la opción tomada en esta prueba

**Decisión central — niveles vs. cambios.** Caben tres representaciones:

| Opción | Idea | Riesgo / ventaja |
|---|---|---|
| A. Regresión en **niveles** | correlacionar precios directamente | Correlaciones ~0.99 → **regresión espuria** por tendencias comunes. ADF no rechaza raíz unitaria (p: X 0.43, Y 0.11, Z 0.21, Eq1 0.14, Eq2 0.33). |
| B. Modelar **cambios (d1)** y reconstruir nivel | series estacionarias (ADF p≈0; KPSS no rechaza) | Relaciones **genuinas**; requiere reconstruir el nivel y propagar incertidumbre. **(Elegida)** |
| C. **Cointegración / ECM-VECM** en niveles | usar relación de largo plazo si existe | Engle-Granger da cointegración Eq1~Y (p=0.013), Eq2~Z (p=0.016); Johansen sobreestima (5 relaciones) por tendencias comunes. Se usa como **respaldo**, no como modelo único. |

**Opción tomada:** **B con respaldo de C**. Se entrena sobre cambios mensuales (estacionarios) y se reconstruye el nivel acumulando cambios sobre el último nivel observado; la cointegración confirma que la asociación Eq1–Y y Eq2–Z es de largo plazo (no espuria). Para el contraste de familias se incluye **SARIMAX con exógenas en nivel** (integra d=1 internamente), que para
Equipo1 resultó el mejor modelo.

**Selección de variables — votación por equipo (≥6 métodos).** Un insumo entra al modelo de un equipo solo si lo votan ≥ **60%** de los métodos (parametrizable): correlación con lags, ElasticNetCV, importancia RandomForest, importancia GBM, OLS por p-valor y mutual information.

**Comparación de modelos:** ElasticNet (interpretable) vs. SARIMAX (serie de tiempo con exógenas) vs. GBM (ML), con **media móvil de 3 meses solo como baseline**. Split temporal sin fuga (TRAIN/TEST/VALIDATION) y `TimeSeriesSplit` para walk-forward.

**Proyección:** Monte Carlo (1.000 trayectorias) que propaga la incertidumbre del **insumo**
(SARIMA simulado) y del **modelo** (residual), produciendo bandas de predicción al 90%.

---

## 4) Resultados del análisis de los datos y los modelos

### 4.1 Ingeniería de datos y EDA
- **Parsers robustos por archivo:** X (ISO, orden descendente), Y (BOM + `;` + decimal `,` + fecha D/M/Y), Z (columnas invertidas). Normalizados a series ordenadas y de-duplicadas.
- **Reconciliación:** 100% match dentro de tolerancia; diff máx = 0.0; las tres series extienden el histórico → el consolidado es consistente y los individuales sirven para validar/extender.
- **Estacionariedad (ADF+KPSS):** todas las series son **no estacionarias en nivel** y **estacionarias en d1** (ADF p≈0, KPSS no rechaza). ⇒ se modela en cambios.
- **Correlaciones:** en **niveles** Eq1↔Y = 0.997 y Eq2↔Z = 0.983 (espurias); en **diferencias** caen a **Eq1↔Y = 0.386** y **Eq2↔Z = 0.431**, y el resto se desploma (Eq1↔Z = 0.024, Eq2↔X = 0.085). La brecha niveles-vs-diferencias es la señal central del caso.
- **Correlación parcial (prueba de espuriedad):** la correlación en nivel de **Z con Equipo1 (0.844)** colapsa a **0.000** al controlar por Y → es **espuria, inducida por Y**; la de Y se mantiene (0.990). Confirma, con álgebra de residuales, que Equipo1 responde a Y y no a Z.
- **Correlación cruzada con rezagos:** mejor lag en **0–1 días** (efecto contemporáneo).
- **Cointegración:** Eq1~Y y Eq1~Z, Eq2~Y y Eq2~Z cointegran; **X no cointegra** con ningún equipo.
- **Outliers** (|z|>4 sobre retornos): X 19, Y 51, Z 13, Eq1 3, Eq2 2 → conservados (volatilidad real).

### 4.2 Selección de variables por votación (POR EQUIPO)

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

➡ **Equipo1 → {Y}** (consenso 0.83), **Equipo2 → {Z}** (consenso 1.00). X queda como ruido para ambos equipos. La selección es unánime entre métodos para Equipo2 y mayoritaria para Equipo1.

### Capítulo — Definición de la ventana temporal

*(capítulo dedicado, inmediatamente posterior a la selección de variables)*

- **Rango histórico de entrenamiento:** 2010-01 → 2021-12 (**144 meses, ~88%**). Justificación: cubre múltiples ciclos de precio y es suficiente para estimar la dinámica; no se recorta por quiebres porque las series en d1 son estacionarias y los modelos en cambios son robustos a cambios de nivel. **TEST** 2022 (12 meses, ~7%) y **VALIDATION** 2023-01→08 (8 meses, ~5%), ambos **fuera de muestra y posteriores** en el tiempo (sin fuga).
- **Frecuencia de trabajo:** **mensual** para el pronóstico (la planeación financiera se ejecuta mes a mes), con EDA/selección en diario para mayor potencia estadística.
- **Horizonte de pronóstico:** **6 meses**. Razonamiento: (i) coincide con una ventana típica de planeación/aprovisionamiento por fase de obra; (ii) más allá de ~6 meses la banda de predicción se ensancha tanto (insumos tipo *random walk*) que el pronóstico puntual pierde valor accionable; (iii) es parametrizable en `config.yaml` (`forecast_horizon_months`).

### 4.3 VIF por equipo (multicolinealidad)
Tras la selección, cada equipo queda con **una** variable (Y para Eq1, Z para Eq2), por lo que el **VIF = 1.0** (no hay colinealidad). Umbral definido = 10; estrategia si se superara: regularización (ElasticNet, ya en uso) y/o eliminar la variable redundante de mayor VIF. El procedimiento de VIF por equipo queda implementado y se activaría automáticamente si la selección retornara ≥2 insumos.

### 4.4 Modelado y evaluación (real vs. predicho, test+validation)

| Equipo | Modelo | MAE | RMSE | MAPE % | R² | MASE | Dir. acc. | F1 dir. |
|---|---|---|---|---|---|---|---|---|
| **Equipo1** | **SARIMAX** ⭐ | 8.58 | 11.23 | **1.67** | 0.983 | 0.378 | **1.00** | 1.00 |
| Equipo1 | ElasticNet | 11.83 | 13.64 | 2.30 | 0.975 | 0.522 | 0.95 | 0.94 |
| Equipo1 | GBM | 14.42 | 17.89 | 2.79 | 0.957 | 0.636 | 0.95 | 0.94 |
| Equipo1 | Baseline MM(3) | 49.54 | 59.18 | 9.59 | 0.527 | 2.186 | 0.47 | 0.44 |
| **Equipo2** | **ElasticNet** ⭐ | 29.75 | 35.56 | **2.80** | 0.959 | 0.772 | 0.84 | 0.82 |
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

Horizonte **6 meses** (2023-09 → 2024-02), banda **90%** por Monte Carlo (1.000 trayectorias) con propagación de incertidumbre de insumo (SARIMA simulado) y de modelo (residual). Último nivel observado (2023-08-31): Equipo1 = 451.73; Equipo2 = 955.35.

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

**Interpretación para planeación financiera:** el costo esperado es **estable** (sin tendencia fuerte de corto plazo), pero la **banda se ensancha con el horizonte** — para presupuestar de forma conservadora conviene usar el **límite superior** (escenario de estrés): p. ej. a febrero 2024, Equipo1 ≤ ~589 y Equipo2 ≤ ~1187 con 90% de confianza. Figuras `reports/figures/forecast_*.png`.

> **Reconstrucción de nivel:** como se modeló en cambios, cada trayectoria reconstruye el nivel
> como `nivel_t = último_nivel_observado + Σ cambios_simulados`. La banda surge de los percentiles
> 5–95 de las 1.000 trayectorias, capturando ambas fuentes de incertidumbre.

---

## 6) Futuros ajustes o mejoras

1. **Modelo conjunto de largo plazo (VECM/ECM):** explotar la cointegración Eq–insumo para un modelo de corrección de error con relación de equilibrio explícita.
2. **Pronóstico endógeno de insumos:** sustituir el SARIMA de simulación por modelos de mercado (futuros, drivers macro) para insumos X/Y/Z y reducir la incertidumbre del horizonte.
3. **Selección con lags mensuales y no linealidades:** ampliar la votación a rezagos mensuales y  métodos como Boruta y Granger (ya previstos como opcionales).
4. **Backtesting walk-forward expandido** con reentrenamiento mensual y *intervalos conformales* (cobertura garantizada) en lugar de bandas Monte Carlo.
5. **Monitoreo de drift** en producción (PSI/MAPE rodante) con reentrenamiento disparado por ADF.
6. **Cuantificar el impacto presupuestal** (traducir banda → reserva de contingencia óptima).
7. **Validación continua del modelo** con datos nuevos para confirmar que las relaciones se mantienen y recalibrar si cambian.

---

## 7) Apreciaciones y comentarios del caso (opcional)

- **El valor está en derivar las relaciones de los datos.** La metodología por votación + estacionariedad + correlación parcial + cointegración **prueba** que Equipo1→Y y Equipo2→Z, y que la media móvil es el peor método (MAPE ~9% vs ~2%). Asumir relaciones sin evidencia habría llevado a un modelo peor y a conclusiones equivocadas.
- **La lección econométrica** (niveles vs. cambios) es el corazón técnico: correlaciones de 0.99 en niveles son un espejismo de tendencias comunes; en cambios la señal real es 0.39–0.43, y la correlación parcial confirma que la asociación de Z con Equipo1 era espuria (inducida por Y).
- **Honestidad sobre incertidumbre:** el pronóstico puntual importa menos que la banda; por eso se entrega y se recomienda presupuestar sobre el escenario superior.

---

### Anexos
- **IA convencional vs. Agente de IA:** ver `reports/IA_vs_agente.md`.
- **Arquitectura cloud (Azure) + diagrama Mermaid:** ver `reports/arquitectura_azure.md`.
- **Artefactos reproducibles:** `reports/artifacts/*.json`, figuras en `reports/figures/` (se regeneran con `make all`).
