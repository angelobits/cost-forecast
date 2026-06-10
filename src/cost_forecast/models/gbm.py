"""Modelo de Machine Learning (Gradient Boosting)

Captura no linealidades e interacciones entre insumos, es menos interpretable que
ElasticNet pero útil como contraste de capacidad predictiva
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class GbmChangeModel:
    """Gradient Boosting de los cambios del equipo sobre cambios de insumos"""

    name = "gbm"

    def __init__(self, n_estimators: int, max_depth: int, learning_rate: float, seed: int) -> None:
        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        self.columns_: list[str] = []
        self.resid_std_: float = 0.0

    def fit(self, x_change: pd.DataFrame, y_change: pd.Series) -> GbmChangeModel:
        self.columns_ = list(x_change.columns)
        self._model.fit(x_change.values, y_change.values)
        resid = y_change.values - self._model.predict(x_change.values)
        self.resid_std_ = float(np.std(resid, ddof=1))
        return self

    def predict(self, x_change: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(x_change[self.columns_].values)
        return pd.Series(preds, index=x_change.index, name="pred_change")

    @property
    def importances(self) -> pd.Series:
        return pd.Series(self._model.feature_importances_, index=self.columns_, name="importance")
