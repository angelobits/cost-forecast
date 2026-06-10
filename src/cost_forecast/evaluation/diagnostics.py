"""Diagnóstico de residuales: autocorrelación, normalidad y heterocedasticidad."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tools.tools import add_constant


def residual_diagnostics(
    residuals: pd.Series, exog: pd.DataFrame | None = None
) -> dict[str, object]:
    """Calcula pruebas de residuales.

    Args:
        residuals: Serie de residuales (real - predicho).
        exog: Regresores para Breusch-Pagan (heterocedasticidad). Si es ``None``
            se usa una tendencia temporal simple.

    Returns:
        Diccionario con estadísticos y p-valores de Ljung-Box, Jarque-Bera y
        Breusch-Pagan.
    """
    r = residuals.dropna()
    out: dict[str, object] = {}

    lb = acorr_ljungbox(r, lags=[min(10, len(r) // 2)], return_df=True)
    out["ljung_box_p"] = round(float(lb["lb_pvalue"].iloc[-1]), 4)
    out["autocorrelacion"] = "sí" if out["ljung_box_p"] < 0.05 else "no"

    jb_stat, jb_p = stats.jarque_bera(r)[:2]
    out["jarque_bera_p"] = round(float(jb_p), 4)
    out["normalidad"] = "sí" if jb_p >= 0.05 else "no"

    if exog is None:
        exog = pd.DataFrame({"t": np.arange(len(r))}, index=r.index)
    X = add_constant(exog.loc[r.index])
    try:
        _, bp_p, _, _ = het_breuschpagan(r.values, X.values)
        out["breusch_pagan_p"] = round(float(bp_p), 4)
        out["homocedasticidad"] = "sí" if bp_p >= 0.05 else "no"
    except Exception:  # pragma: no cover - robustez numérica
        out["breusch_pagan_p"] = float("nan")
        out["homocedasticidad"] = "n/d"
    return out
