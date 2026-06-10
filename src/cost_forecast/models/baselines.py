"""Baselines de referencia. La media móvil simple es SOLO baseline, nunca método único

Una media móvil de 3 meses se incluye como piso de comparación para demostrar que
cualquier modelo serio debe superarla; no es un método de pronóstico final
"""

from __future__ import annotations

import pandas as pd


class MovingAverageBaseline:
    """Predice el nivel del próximo paso como la media de los últimos ``window`` niveles"""

    name = "baseline_ma"

    def __init__(self, window: int) -> None:
        self._window = window

    def predict_level(self, y_level_history: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Predicción one-step: media móvil de los ``window`` valores previos"""
        ma = y_level_history.rolling(self._window).mean().shift(1)
        return ma.reindex(index)

    def forecast_level(
        self, y_level_history: pd.Series, steps: int, index: pd.DatetimeIndex
    ) -> pd.Series:
        """Pronóstico plano: repite la última media móvil disponible ``steps`` veces"""
        last_ma = float(y_level_history.rolling(self._window).mean().iloc[-1])
        return pd.Series([last_ma] * steps, index=index, name="pred")
