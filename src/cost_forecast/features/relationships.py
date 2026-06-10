"""Relaciones insumo→equipo: correlación cruzada con rezagos y cointegración.

El efecto de una materia prima sobre el precio de un equipo puede no ser
contemporáneo (rezago de días). Además, en **niveles** las correlaciones pueden
ser espurias por tendencias comunes; por eso medimos también en **diferencias** y
probamos **cointegración** (Engle-Granger y Johansen) para decidir si un modelo
en niveles es legítimo (relación de largo plazo estable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from ..logging_utils import get_logger

logger = get_logger(__name__)


def cross_correlation_lags(
    input_series: pd.Series,
    target_series: pd.Series,
    max_lag: int,
) -> pd.DataFrame:
    """Correlación de ``target_t`` con ``input_{t-k}`` para k=0..max_lag.

    Un k>0 con correlación alta indica que el insumo **adelanta** al equipo.

    Returns:
        DataFrame con columnas ``lag`` y ``corr``.
    """
    rows = []
    for k in range(0, max_lag + 1):
        shifted = input_series.shift(k)
        joined = pd.concat([target_series, shifted], axis=1).dropna()
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        rows.append({"lag": k, "corr": corr})
    return pd.DataFrame(rows)


def best_lag(input_series: pd.Series, target_series: pd.Series, max_lag: int) -> tuple[int, float]:
    """Devuelve (lag, corr) con mayor ``|corr|`` en el rango evaluado."""
    cc = cross_correlation_lags(input_series, target_series, max_lag)
    idx = cc["corr"].abs().idxmax()
    return int(cc.loc[idx, "lag"]), float(cc.loc[idx, "corr"])


def partial_correlation(df: pd.DataFrame, a: str, b: str, controls: list[str]) -> float:
    """Correlación parcial entre ``a`` y ``b`` controlando por ``controls``.

    Aísla el efecto neto de ``a`` sobre ``b`` removiendo (por regresión) la parte
    explicada por las variables de control. Si la correlación bivariada es alta pero
    la parcial cae a ~0, la relación original era **espuria** (inducida por un control).

    Returns:
        Coeficiente de correlación parcial (Pearson de los residuales).
    """
    resid_a = sm.OLS(df[a], sm.add_constant(df[controls])).fit().resid
    resid_b = sm.OLS(df[b], sm.add_constant(df[controls])).fit().resid
    return float(np.corrcoef(resid_a, resid_b)[0, 1])


def partial_correlation_table(
    df: pd.DataFrame, inputs: list[str], targets: list[str]
) -> pd.DataFrame:
    """Tabla de correlación bivariada vs. parcial de cada insumo con cada equipo.

    Demuestra cuantitativamente qué correlaciones en nivel son genuinas y cuáles
    desaparecen al controlar por los demás insumos (espurias).
    """
    rows = []
    for tgt in targets:
        for inp in inputs:
            controls = [c for c in inputs if c != inp]
            biv = float(np.corrcoef(df[inp], df[tgt])[0, 1])
            par = partial_correlation(df, inp, tgt, controls)
            rows.append(
                {
                    "equipo": tgt,
                    "insumo": inp,
                    "corr_bivariada": round(biv, 4),
                    "corr_parcial": round(par, 4),
                    "espuria": bool(abs(biv) > 0.5 and abs(par) < 0.1),
                }
            )
    return pd.DataFrame(rows)


def engle_granger(y: pd.Series, x: pd.Series, significance: float = 0.05) -> dict[str, object]:
    """Prueba de cointegración de Engle-Granger entre dos series en niveles."""
    joined = pd.concat([y, x], axis=1).dropna()
    stat, pvalue, _ = coint(joined.iloc[:, 0], joined.iloc[:, 1])
    return {
        "test": "engle_granger",
        "pair": f"{y.name}~{x.name}",
        "stat": round(float(stat), 3),
        "pvalue": round(float(pvalue), 4),
        "cointegrated": bool(pvalue < significance),
    }


def johansen_trace(df: pd.DataFrame, significance: float = 0.05) -> dict[str, object]:
    """Prueba de Johansen (traza) sobre un conjunto de series en niveles.

    Devuelve el número de relaciones de cointegración detectadas comparando la
    estadística de traza con el valor crítico al 5%.
    """
    data = df.dropna()
    # det_order=0 (constante), k_ar_diff=1
    result = coint_johansen(data, det_order=0, k_ar_diff=1)
    crit_idx = {0.10: 0, 0.05: 1, 0.01: 2}.get(significance, 1)
    trace = result.lr1
    crit = result.cvt[:, crit_idx]
    n_relations = int(np.sum(trace > crit))
    return {
        "test": "johansen_trace",
        "vars": list(df.columns),
        "trace_stats": [round(float(v), 2) for v in trace],
        "crit_values": [round(float(v), 2) for v in crit],
        "n_cointegration_relations": n_relations,
    }
