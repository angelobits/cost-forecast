"""Transformaciones de series: diferencias, retornos log, lags y agregación mensual.

La decisión *niveles vs. cambios* se materializa aquí. Para el negocio (cuánto
costará) trabajamos el **nivel**; para validar estacionariedad y relaciones
usamos **diferencias** (``d1``) y retornos log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def difference(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Primera diferencia (cambio absoluto) de cada columna."""
    return df.diff(periods=periods).dropna(how="all")


def log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Retorno logarítmico de cada columna (requiere valores positivos)."""
    return np.log(df).diff().dropna(how="all")


def add_lags(df: pd.DataFrame, columns: list[str], max_lag: int) -> pd.DataFrame:
    """Agrega columnas rezagadas ``col_lagK`` para K=1..max_lag.

    Args:
        df: DataFrame fuente.
        columns: Columnas a rezagar.
        max_lag: Lag máximo (inclusive).

    Returns:
        DataFrame con las columnas originales más las rezagadas.
    """
    out = df.copy()
    for col in columns:
        for k in range(1, max_lag + 1):
            out[f"{col}_lag{k}"] = df[col].shift(k)
    return out


def to_monthly_level(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega a frecuencia mensual usando el **nivel de fin de mes**.

    Para planeación financiera interesa el nivel (cuánto costará el equipo), por
    lo que tomamos el último valor del mes (último precio observado), no el
    promedio de cambios diarios.
    """
    return df.resample("ME").last()


def to_monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega a frecuencia mensual usando el **promedio mensual** del nivel."""
    return df.resample("ME").mean()
