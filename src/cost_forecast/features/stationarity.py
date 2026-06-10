"""Pruebas de estacionariedad (ADF + KPSS) y diagnóstico de series.

ADF y KPSS son complementarias y se usan en conjunto:

* **ADF** H0: hay raíz unitaria (no estacionaria). p < alpha => estacionaria.
* **KPSS** H0: la serie es estacionaria. p < alpha => NO estacionaria.

La combinación desambigua casos frontera (p. ej. estacionariedad en tendencia).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class StationarityResult:
    """Resultado combinado ADF+KPSS para una serie."""

    name: str
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float
    adf_stationary: bool
    kpss_stationary: bool

    @property
    def verdict(self) -> str:
        """Veredicto combinado legible."""
        if self.adf_stationary and self.kpss_stationary:
            return "estacionaria"
        if not self.adf_stationary and not self.kpss_stationary:
            return "no estacionaria"
        if self.adf_stationary and not self.kpss_stationary:
            return "tendencia-estacionaria / diferenciar"
        return "ambiguo"

    def as_row(self) -> dict[str, object]:
        return {
            "serie": self.name,
            "adf_stat": round(self.adf_stat, 3),
            "adf_p": round(self.adf_pvalue, 4),
            "kpss_stat": round(self.kpss_stat, 3),
            "kpss_p": round(self.kpss_pvalue, 4),
            "veredicto": self.verdict,
        }


def assess_stationarity(
    series: pd.Series,
    adf_alpha: float = 0.05,
    kpss_alpha: float = 0.05,
) -> StationarityResult:
    """Aplica ADF y KPSS a una serie y devuelve el veredicto combinado."""
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    # KPSS emite InterpolationWarning cuando el p-valor se trunca a [0.01, 0.1];
    # es esperado y no afecta el veredicto, por lo que se silencia localmente.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    return StationarityResult(
        name=str(series.name),
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_p),
        kpss_stat=float(kpss_stat),
        kpss_pvalue=float(kpss_p),
        adf_stationary=adf_p < adf_alpha,
        kpss_stationary=kpss_p >= kpss_alpha,
    )


def stationarity_table(
    df: pd.DataFrame, adf_alpha: float = 0.05, kpss_alpha: float = 0.05
) -> pd.DataFrame:
    """Tabla ADF+KPSS para todas las columnas de un DataFrame."""
    rows = [assess_stationarity(df[col], adf_alpha, kpss_alpha).as_row() for col in df.columns]
    return pd.DataFrame(rows)
