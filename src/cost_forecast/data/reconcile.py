"""Reconciliación de las series individuales contra el consolidado limpio.

Objetivos:

1. Verificar que las series individuales coinciden con las columnas del
   consolidado dentro de una tolerancia (¿son la misma fuente?).
2. Detectar si una serie individual **extiende** el histórico (rango mayor).
3. Documentar la decisión de fuente de verdad: usamos el **consolidado** para el
   modelado (alineado, sin NaNs) y las individuales para auditoría/extensión.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .loaders import load_all_individual, load_consolidated

logger = get_logger(__name__)


@dataclass
class ReconciliationResult:
    """Resumen de la reconciliación de una materia prima."""

    name: str
    n_overlap: int
    n_match_within_tol: int
    match_rate: float
    max_abs_diff: float
    individual_start: pd.Timestamp
    individual_end: pd.Timestamp
    consolidated_start: pd.Timestamp
    consolidated_end: pd.Timestamp
    extends_history: bool

    def as_row(self) -> dict[str, object]:
        return {
            "serie": self.name,
            "n_overlap": self.n_overlap,
            "match_rate_%": round(100 * self.match_rate, 2),
            "max_abs_diff": round(self.max_abs_diff, 4),
            "ind_rango": f"{self.individual_start.date()}→{self.individual_end.date()}",
            "cons_rango": f"{self.consolidated_start.date()}→{self.consolidated_end.date()}",
            "extiende_historico": self.extends_history,
        }


def reconcile_series(
    individual: pd.Series,
    consolidated_col: pd.Series,
    tolerance: float,
) -> ReconciliationResult:
    """Compara una serie individual con su columna en el consolidado.

    Args:
        individual: Serie individual normalizada (``Price_X`` etc.).
        consolidated_col: Columna homóloga del consolidado.
        tolerance: Tolerancia relativa para considerar dos valores "iguales".

    Returns:
        :class:`ReconciliationResult` con métricas de coincidencia y cobertura.
    """
    name = str(individual.name)
    joined = pd.concat({"ind": individual, "cons": consolidated_col}, axis=1, join="inner").dropna()
    if joined.empty:
        abs_diff = pd.Series(dtype="float64")
        rel = pd.Series(dtype="float64")
    else:
        abs_diff = (joined["ind"] - joined["cons"]).abs()
        rel = abs_diff / joined["cons"].abs().replace(0, pd.NA)
    n_overlap = len(joined)
    within = int((rel <= tolerance).sum()) if n_overlap else 0
    extends = individual.index.min() < consolidated_col.index.min() or (
        individual.index.max() > consolidated_col.index.max()
    )
    result = ReconciliationResult(
        name=name,
        n_overlap=n_overlap,
        n_match_within_tol=within,
        match_rate=(within / n_overlap) if n_overlap else 0.0,
        max_abs_diff=float(abs_diff.max()) if n_overlap else float("nan"),
        individual_start=individual.index.min(),
        individual_end=individual.index.max(),
        consolidated_start=consolidated_col.index.min(),
        consolidated_end=consolidated_col.index.max(),
        extends_history=bool(extends),
    )
    logger.info(
        "Reconciliación %s: overlap=%d, match=%.1f%%, extiende=%s",
        name,
        n_overlap,
        100 * result.match_rate,
        result.extends_history,
    )
    return result


def reconcile_all(cfg: Config) -> pd.DataFrame:
    """Reconcilia X, Y, Z contra el consolidado y devuelve una tabla resumen."""
    consolidated = load_consolidated(cfg.path("consolidated"))
    individuals = load_all_individual(cfg)
    tol = float(cfg["data"]["reconcile_tolerance"])
    rows = []
    for key, series in individuals.items():
        col = f"{cfg['schema']['price_prefix']}{key}"
        rows.append(reconcile_series(series, consolidated[col], tol).as_row())
    return pd.DataFrame(rows)
