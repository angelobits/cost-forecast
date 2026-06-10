"""Particiones temporales sin fuga de información.

Se usan dos esquemas complementarios:

* **Split fijo TRAIN/TEST/VALIDATION** por fechas (declarado en config) para el
  reporte final reproducible.
* **Walk-forward** (``TimeSeriesSplit``) para validación robusta sin mirar al futuro.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..config import Config


@dataclass
class TemporalSplit:
    """Contenedor de los tres bloques temporales (índices de fecha)."""

    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    validation: pd.DatetimeIndex

    def describe(self, total: int) -> dict[str, object]:
        return {
            "train": f"{self.train.min().date()}→{self.train.max().date()} (n={len(self.train)}, {100*len(self.train)/total:.0f}%)",
            "test": f"{self.test.min().date()}→{self.test.max().date()} (n={len(self.test)}, {100*len(self.test)/total:.0f}%)",
            "validation": f"{self.validation.min().date()}→{self.validation.max().date()} (n={len(self.validation)}, {100*len(self.validation)/total:.0f}%)",
        }


def fixed_split(index: pd.DatetimeIndex, cfg: Config) -> TemporalSplit:
    """Construye el split fijo a partir de las fechas declaradas en config."""
    w = cfg["window"]
    train = index[(index >= w["train_start"]) & (index <= w["train_end"])]
    test = index[(index >= w["test_start"]) & (index <= w["test_end"])]
    val = index[(index >= w["validation_start"]) & (index <= w["validation_end"])]
    return TemporalSplit(train=train, test=test, validation=val)


def walk_forward_splits(n_samples: int, n_splits: int) -> TimeSeriesSplit:
    """Devuelve un ``TimeSeriesSplit`` configurado (walk-forward)."""
    return TimeSeriesSplit(n_splits=n_splits)
