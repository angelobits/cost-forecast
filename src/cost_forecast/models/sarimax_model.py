"""Modelo de series de tiempo con exógenas: SARIMAX(p,d,q) + insumos en nivel

A diferencia de los modelos de cambios, SARIMAX integra internamente (d=1) y
modela el **nivel** del equipo con los insumos seleccionados como variables
exógenas, entregando intervalos de predicción analíticos
"""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SarimaxModel:
    """Envoltorio de ``statsmodels`` SARIMAX con exógenas en nivel"""

    name = "sarimax"

    def __init__(
        self, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]
    ) -> None:
        self._order = tuple(order)
        self._seasonal_order = tuple(seasonal_order)
        self._res = None
        self.exog_cols_: list[str] = []

    def fit(self, y_level: pd.Series, exog_level: pd.DataFrame | None) -> SarimaxModel:
        self.exog_cols_ = list(exog_level.columns) if exog_level is not None else []
        exog = None if exog_level is None else exog_level.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_level.values,
                exog=exog,
                order=self._order,
                seasonal_order=self._seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._res = model.fit(disp=False)
        return self

    def forecast_level(
        self, steps: int, exog_future: pd.DataFrame | None, index: pd.DatetimeIndex, alpha: float
    ) -> pd.DataFrame:
        """Pronóstico de nivel a ``steps`` pasos con intervalo (1-alpha)"""
        assert self._res is not None, "El modelo no ha sido entrenado"
        exog = None if exog_future is None else exog_future[self.exog_cols_].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._res.get_forecast(steps=steps, exog=exog)
            mean = fc.predicted_mean
            ci = fc.conf_int(alpha=alpha)
        return pd.DataFrame({"pred": mean, "lower": ci[:, 0], "upper": ci[:, 1]}, index=index)
