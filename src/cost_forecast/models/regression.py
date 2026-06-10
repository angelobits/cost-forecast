"""Modelos de regresión sobre cambios (d1) con insumos seleccionados

``ElasticNetChangeModel`` es el modelo **interpretable** de referencia: regresión
regularizada (ElasticNet) que combina selección (L1) y estabilidad ante
colinealidad (L2). Trabaja sobre cambios estandarizados y expone los coeficientes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ElasticNetChangeModel:
    """Regresión ElasticNet de los cambios del equipo sobre cambios de insumos"""

    name = "elasticnet"

    def __init__(self, l1_ratio: list[float], seed: int) -> None:
        self._l1_ratio = l1_ratio
        self._seed = seed
        self._pipe: Pipeline | None = None
        self.columns_: list[str] = []
        self.resid_std_: float = 0.0

    def fit(self, x_change: pd.DataFrame, y_change: pd.Series) -> ElasticNetChangeModel:
        self.columns_ = list(x_change.columns)
        self._pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=self._l1_ratio, cv=5, random_state=self._seed, max_iter=10000
                    ),
                ),
            ]
        )
        self._pipe.fit(x_change.values, y_change.values)
        resid = y_change.values - self._pipe.predict(x_change.values)
        self.resid_std_ = float(np.std(resid, ddof=1))
        return self

    def predict(self, x_change: pd.DataFrame) -> pd.Series:
        assert self._pipe is not None, "El modelo no ha sido entrenado."
        preds = self._pipe.predict(x_change[self.columns_].values)
        return pd.Series(preds, index=x_change.index, name="pred_change")

    @property
    def coefficients(self) -> pd.Series:
        assert self._pipe is not None
        model = self._pipe.named_steps["model"]
        return pd.Series(model.coef_, index=self.columns_, name="coef")
