"""Proyección de costos con incertidumbre y reconstrucción de nivel

Como el costo de un equipo depende de insumos cuyo futuro también es incierto, la
proyección se hace por Monte Carlo:

1. Se ajusta un SARIMA por insumo seleccionado y se simulan ``n`` trayectorias
   futuras de su nivel (incertidumbre del insumo).
2. Para cada trayectoria se calculan los cambios del insumo, se predice el cambio
   del equipo con el modelo de cambios entrenado y se añade ruido de residual del
   modelo (incertidumbre del modelo).
3. Se reconstruye el nivel del equipo acumulando cambios sobre el último nivel
   observado. Los percentiles de las ``n`` trayectorias dan la banda de predicción.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """Proyección de nivel con banda de predicción"""

    target: str
    frame: pd.DataFrame  # index=fechas futuras; cols: pred, lower, upper


def _simulate_input_paths(level: pd.Series, steps: int, n_paths: int, seed: int) -> np.ndarray:
    """Simula ``n_paths`` trayectorias futuras del nivel de un insumo con ARIMA(1,1,1).

    Returns:
        Array (n_paths, steps) de niveles simulados.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(level.values, order=(1, 1, 1)).fit()
        sims = res.simulate(
            nsimulations=steps,
            repetitions=n_paths,
            anchor="end",
            random_state=np.random.RandomState(seed),
        )
    # statsmodels puede devolver (steps,), (steps, n_paths) o (steps, 1, n_paths).
    sims = np.asarray(sims)
    sims = np.squeeze(sims)
    if sims.ndim == 1:
        sims = sims.reshape(steps, 1)
    # asegurar orientación (steps, n_paths) antes de transponer
    if sims.shape[0] != steps and sims.shape[-1] == steps:
        sims = sims.T
    return sims.T  # -> (n_paths, steps)


def forecast_equipo(
    y_level: pd.Series,
    inputs_level: pd.DataFrame,
    change_model,
    horizon: int,
    confidence: float,
    n_paths: int,
    seed: int,
    future_index: pd.DatetimeIndex,
) -> ForecastResult:
    """Proyecta el nivel del equipo propagando incertidumbre de insumos y modelo.

    Args:
        y_level: Nivel histórico mensual del equipo.
        inputs_level: Niveles mensuales de los insumos seleccionados.
        change_model: Modelo entrenado sobre cambios (expone ``predict`` y ``resid_std_``).
        horizon: Número de meses a proyectar.
        confidence: Nivel de confianza de la banda (p. ej. 0.90).
        n_paths: Número de trayectorias Monte Carlo.
        seed: Semilla.
        future_index: Índice de fechas futuras (longitud == horizon).

    Returns:
        :class:`ForecastResult` con media y banda inferior/superior.
    """
    rng = np.random.RandomState(seed)
    cols = list(inputs_level.columns)
    last_levels = inputs_level.iloc[-1]
    anchor = float(y_level.iloc[-1])
    resid_std = float(getattr(change_model, "resid_std_", 0.0))

    # Simular trayectorias por insumo
    sims = {
        c: _simulate_input_paths(inputs_level[c], horizon, n_paths, seed + i)
        for i, c in enumerate(cols)
    }

    equipo_paths = np.empty((n_paths, horizon))
    for p in range(n_paths):
        # Cambios del insumo: primer paso vs último nivel observado, luego diff intra-path
        change_frame = {}
        for c in cols:
            path = sims[c][p]
            prev = np.concatenate([[last_levels[c]], path[:-1]])
            change_frame[c] = path - prev
        x_change = pd.DataFrame(change_frame, index=future_index)
        pred_change = change_model.predict(x_change).values
        pred_change = pred_change + rng.normal(0.0, resid_std, size=horizon)
        equipo_paths[p] = anchor + np.cumsum(pred_change)

    lower_q = (1 - confidence) / 2
    upper_q = 1 - lower_q
    frame = pd.DataFrame(
        {
            "pred": equipo_paths.mean(axis=0),
            "lower": np.quantile(equipo_paths, lower_q, axis=0),
            "upper": np.quantile(equipo_paths, upper_q, axis=0),
        },
        index=future_index,
    )
    logger.info("Proyección %s: %d meses, banda %.0f%%", y_level.name, horizon, 100 * confidence)
    return ForecastResult(target=str(y_level.name), frame=frame)


def forecast_equipo_sarimax(
    y_level: pd.Series,
    inputs_level: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    horizon: int,
    confidence: float,
    n_paths: int,
    seed: int,
    future_index: pd.DatetimeIndex,
) -> ForecastResult:
    """Proyecta con SARIMAX(+exógenas) propagando incertidumbre de insumo y modelo.

    Para cada trayectoria simulada de los insumos se obtiene el pronóstico SARIMAX
    condicionado a ese exógeno y se añade ruido de la desviación estándar del
    pronóstico (incertidumbre del modelo). Los percentiles de las trayectorias dan
    la banda. Es el camino usado cuando SARIMAX es el mejor modelo del equipo
    """
    import warnings

    rng = np.random.RandomState(seed)
    cols = list(inputs_level.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(
            y_level.values,
            exog=inputs_level.values,
            order=tuple(order),
            seasonal_order=tuple(seasonal_order),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    sims = {
        c: _simulate_input_paths(inputs_level[c], horizon, n_paths, seed + i)
        for i, c in enumerate(cols)
    }
    equipo_paths = np.empty((n_paths, horizon))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(n_paths):
            exog_future = np.column_stack([sims[c][p] for c in cols])
            fc = res.get_forecast(steps=horizon, exog=exog_future)
            mean = np.asarray(fc.predicted_mean)
            se = np.asarray(fc.se_mean)
            equipo_paths[p] = mean + rng.normal(0.0, se)

    lower_q = (1 - confidence) / 2
    frame = pd.DataFrame(
        {
            "pred": equipo_paths.mean(axis=0),
            "lower": np.quantile(equipo_paths, lower_q, axis=0),
            "upper": np.quantile(equipo_paths, 1 - lower_q, axis=0),
        },
        index=future_index,
    )
    logger.info(
        "Proyección SARIMAX %s: %d meses, banda %.0f%%", y_level.name, horizon, 100 * confidence
    )
    return ForecastResult(target=str(y_level.name), frame=frame)
