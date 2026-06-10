"""Selección de variables por **votación** y **por equipo** (modelos independientes).

Para cada equipo se evalúa qué materias primas (X, Y, Z) son determinantes usando
>=5 métodos independientes. Cada método emite un voto por insumo; un insumo entra
al modelo del equipo si lo votan >= ``consensus_threshold`` de los métodos.

La selección se realiza sobre **diferencias (d1)** alineadas al **mejor rezago**
de cada insumo, para evitar la correlación espuria que aparece en niveles por
tendencias comunes. Esto es lo que revela las relaciones genuinas del caso.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..logging_utils import get_logger
from .relationships import best_lag

logger = get_logger(__name__)


@dataclass
class SelectionResult:
    """Resultado de la selección por votación para un equipo."""

    target: str
    voting_matrix: pd.DataFrame  # filas=métodos, columnas=insumos, valores 0/1
    consensus: pd.Series  # fracción de votos por insumo
    selected: list[str]  # insumos seleccionados (>= umbral)
    best_lags: dict[str, int] = field(default_factory=dict)


def _build_diff_matrix(
    inputs: pd.DataFrame, target: pd.Series, max_lag: int
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Construye matriz de insumos en diferencias alineada al mejor lag por insumo."""
    target_d = target.diff()
    aligned = {}
    lags: dict[str, int] = {}
    for col in inputs.columns:
        lag, _ = best_lag(inputs[col].diff(), target_d, max_lag)
        lags[col] = lag
        aligned[col] = inputs[col].diff().shift(lag)
    X = pd.DataFrame(aligned)
    joined = pd.concat([target_d.rename("y"), X], axis=1).dropna()
    return joined[inputs.columns.tolist()], joined["y"], lags


def _vote_corr_lag(X: pd.DataFrame, y: pd.Series, threshold: float) -> dict[str, int]:
    return {c: int(abs(X[c].corr(y)) >= threshold) for c in X.columns}


def _vote_elasticnet(X: pd.DataFrame, y: pd.Series, l1_ratio, seed: int) -> dict[str, int]:
    Xs = StandardScaler().fit_transform(X.values)
    model = ElasticNetCV(l1_ratio=l1_ratio, cv=5, random_state=seed, max_iter=10000)
    model.fit(Xs, y.values)
    return {c: int(abs(coef) > 1e-8) for c, coef in zip(X.columns, model.coef_, strict=True)}


def _vote_tree_importance(X: pd.DataFrame, y: pd.Series, model, quantile: float) -> dict[str, int]:
    model.fit(X.values, y.values)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    cutoff = imp.quantile(quantile)
    return {c: int(imp[c] > cutoff) for c in X.columns}


def _vote_ols_pvalue(X: pd.DataFrame, y: pd.Series, alpha: float = 0.05) -> dict[str, int]:
    Xc = sm.add_constant(X)
    model = sm.OLS(y.values, Xc.values).fit()
    pvals = pd.Series(model.pvalues[1:], index=X.columns)
    return {c: int(pvals[c] < alpha) for c in X.columns}


def _vote_mutual_info(X: pd.DataFrame, y: pd.Series, quantile: float, seed: int) -> dict[str, int]:
    mi = mutual_info_regression(X.values, y.values, random_state=seed)
    mi_s = pd.Series(mi, index=X.columns)
    cutoff = mi_s.quantile(quantile)
    return {c: int(mi_s[c] > cutoff) for c in X.columns}


def select_for_target(inputs: pd.DataFrame, target: pd.Series, cfg: Config) -> SelectionResult:
    """Ejecuta la votación de >=5 métodos para un equipo y aplica el umbral de consenso.

    Args:
        inputs: DataFrame de insumos en niveles (columnas X, Y, Z).
        target: Serie del equipo en niveles.
        cfg: Configuración del proyecto.

    Returns:
        :class:`SelectionResult` con la matriz de votación y los insumos seleccionados.
    """
    sel = cfg["selection"]
    seed = cfg.seed
    X, y, lags = _build_diff_matrix(inputs, target, int(sel["max_lag"]))

    votes: dict[str, dict[str, int]] = {}
    votes["corr_lag"] = _vote_corr_lag(X, y, float(sel["corr_abs_threshold"]))
    votes["elasticnet"] = _vote_elasticnet(X, y, [0.1, 0.5, 0.7, 0.9, 1.0], seed)
    votes["rf_importance"] = _vote_tree_importance(
        X,
        y,
        RandomForestRegressor(n_estimators=300, random_state=seed),
        float(sel["rf_importance_quantile"]),
    )
    votes["gbm_importance"] = _vote_tree_importance(
        X, y, GradientBoostingRegressor(random_state=seed), float(sel["rf_importance_quantile"])
    )
    votes["ols_pvalue"] = _vote_ols_pvalue(X, y)
    votes["mutual_info"] = _vote_mutual_info(X, y, float(sel["mi_quantile"]), seed)

    voting_matrix = pd.DataFrame(votes).T  # filas=métodos, columnas=insumos
    voting_matrix = voting_matrix[inputs.columns.tolist()]
    consensus = voting_matrix.mean(axis=0)
    threshold = float(sel["consensus_threshold"])
    selected = [c for c in inputs.columns if consensus[c] >= threshold]

    logger.info(
        "Selección %s: consenso=%s -> seleccionados=%s",
        target.name,
        {k: round(v, 2) for k, v in consensus.items()},
        selected,
    )
    return SelectionResult(
        target=str(target.name),
        voting_matrix=voting_matrix,
        consensus=consensus,
        selected=selected,
        best_lags=lags,
    )
