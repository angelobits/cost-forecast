"""Contratos comunes de modelos y utilidades de reconstrucción de nivel

Todos los modelos exógenos del proyecto se entrenan sobre cambios (d1) por
estacionariedad y luego reconstruyen el nivel acumulando los cambios sobre un
ancla (último nivel observado). Esto resuelve la tensión niveles-vs-cambios:
modelamos lo estacionario y entregamos lo que el negocio necesita (el nivel)
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class LevelModel(Protocol):
    """Interfaz mínima de un modelo que predice el **nivel** de un equipo"""

    name: str

    def fit(self, y_level: pd.Series, exog_level: pd.DataFrame | None) -> LevelModel: ...

    def predict_level(
        self, y_level_history: pd.Series, exog_level: pd.DataFrame | None, index: pd.DatetimeIndex
    ) -> pd.Series: ...


def reconstruct_level(anchor: float, changes: pd.Series) -> pd.Series:
    """Reconstruye el nivel a partir de un ancla y una serie de cambios

    level_t = anchor + sum_{i<=t} change_i
    """
    return anchor + changes.cumsum()


def onestep_level_from_changes(
    true_level_history: pd.Series, predicted_changes: pd.Series
) -> pd.Series:
    """Nivel predicho one-step: nivel_real_{t-1} + cambio_predicho_t

    Reconstrucción honesta para evaluación: cada paso se ancla en el nivel real
    previo (no se arrastra el error), reflejando un *nowcast* dado el insumo
    """
    prev = true_level_history.shift(1)
    aligned = prev.reindex(predicted_changes.index)
    return aligned + predicted_changes
