"""Diagnóstico de multicolinealidad (VIF) calculado **por equipo**.

El VIF se calcula sobre las variables ya seleccionadas de cada equipo por
separado (modelos independientes). VIF > umbral indica multicolinealidad alta;
la estrategia es eliminar/combinar variables o pasar a regularización (ElasticNet),
que es robusta a colinealidad.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from ..logging_utils import get_logger

logger = get_logger(__name__)


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el VIF de cada columna del DataFrame.

    Args:
        df: Matriz de variables (sin constante). Filas con NaN se descartan.

    Returns:
        DataFrame con columnas ``variable`` y ``vif`` ordenado desc.
    """
    data = df.dropna()
    if data.shape[1] == 1:
        # Con una sola variable no hay colinealidad: VIF = 1 por definición.
        return pd.DataFrame({"variable": list(data.columns), "vif": [1.0]})
    X = add_constant(data)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append(
            {"variable": col, "vif": round(float(variance_inflation_factor(X.values, i)), 3)}
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def vif_strategy(vif_df: pd.DataFrame, threshold: float) -> dict[str, object]:
    """Evalúa el VIF contra el umbral y propone una estrategia."""
    high = vif_df[vif_df["vif"] > threshold]["variable"].tolist()
    if not high:
        strategy = "Sin multicolinealidad relevante (todos los VIF < umbral)."
    else:
        strategy = (
            f"VIF alto en {high}: usar regularización (ElasticNet) y/o eliminar la "
            f"variable redundante de mayor VIF; reevaluar."
        )
    return {"threshold": threshold, "high_vif_vars": high, "strategy": strategy}
