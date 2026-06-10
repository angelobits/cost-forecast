"""Orquestador de extremo a extremo

Cada etapa es una función pura-ish que lee config, calcula y persiste artefactos
(JSON/CSV/figuras) en ``reports/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config, load_config
from .data.loaders import load_consolidated
from .data.reconcile import reconcile_all
from .evaluation.diagnostics import residual_diagnostics
from .evaluation.metrics import directional_metrics, error_metrics
from .evaluation.split import fixed_split
from .features.relationships import (
    cross_correlation_lags,
    engle_granger,
    johansen_trace,
    partial_correlation_table,
)
from .features.selection import select_for_target
from .features.stationarity import stationarity_table
from .features.transforms import difference, to_monthly_level
from .features.vif import compute_vif, vif_strategy
from .forecast.project import forecast_equipo, forecast_equipo_sarimax
from .logging_utils import get_logger, set_global_seed
from .models.base import onestep_level_from_changes
from .models.baselines import MovingAverageBaseline
from .models.gbm import GbmChangeModel
from .models.regression import ElasticNetChangeModel
from .models.sarimax_model import SarimaxModel
from .viz import plots

logger = get_logger(__name__)


# Helpers de E/S y de preparación de datos
def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Artefacto escrito: %s", path)


def _write_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info("Artefacto escrito: %s", path)


def _strip_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.replace(prefix, ""))


def load_frames(cfg: Config) -> dict[str, pd.DataFrame]:
    """Carga el consolidado y deriva niveles diarios y mensuales (sin prefijo)."""
    prefix = cfg["schema"]["price_prefix"]
    daily = _strip_prefix(load_consolidated(cfg.path("consolidated")), prefix)
    monthly = to_monthly_level(daily)
    return {"daily": daily, "monthly": monthly}


def _selected_vars(cfg: Config) -> dict[str, list[str]]:
    """Lee las variables seleccionadas persistidas; si no existen, las calcula."""
    art = cfg.path("artifacts_dir") / "selection.json"
    if art.exists():
        data = json.loads(art.read_text(encoding="utf-8"))
        return {t: data[t]["selected"] for t in cfg["schema"]["targets"]}
    return run_features(cfg)["selected"]


def _best_models(cfg: Config) -> dict[str, str]:
    """Lee el mejor modelo por equipo de ``evaluation.json``; default ElasticNet."""
    art = cfg.path("artifacts_dir") / "evaluation.json"
    if not art.exists():
        run_train_evaluate(cfg)
    data = json.loads(art.read_text(encoding="utf-8"))
    return {t: v.get("best_model", "elasticnet") for t, v in data.get("targets", {}).items()}


# Etapa: DATA
def run_data(cfg: Config) -> dict[str, Any]:
    """Reconcilia fuentes y persiste datos procesados (diario y mensual)."""
    set_global_seed(cfg.seed)
    recon = reconcile_all(cfg)
    frames = load_frames(cfg)
    _write_csv(recon, cfg.path("artifacts_dir") / "reconciliation.csv", index=False)
    _write_csv(frames["daily"], cfg.path("processed_dir") / "daily_levels.csv")
    _write_csv(frames["monthly"], cfg.path("processed_dir") / "monthly_levels.csv")
    return {"reconciliation": recon}


# Etapa: EDA
def run_eda(cfg: Config) -> dict[str, Any]:
    """Estacionariedad (niveles y d1), correlaciones, lags, cointegración, outliers."""
    set_global_seed(cfg.seed)
    frames = load_frames(cfg)
    daily = frames["daily"]
    inputs = cfg["schema"]["inputs"]
    targets = cfg["schema"]["targets"]
    adf_a = cfg["eda"]["adf_significance"]
    kpss_a = cfg["eda"]["kpss_significance"]

    stat_levels = stationarity_table(daily, adf_a, kpss_a)
    stat_diffs = stationarity_table(difference(daily), adf_a, kpss_a)
    corr_levels = daily.corr().round(4)
    corr_diffs = difference(daily).corr().round(4)
    # Correlación parcial: prueba qué correlaciones en nivel son espurias (inducidas por otro insumo)
    partial_corr = partial_correlation_table(daily, inputs, targets)

    # Correlación cruzada con lags y mejor lag insumo->equipo
    max_lag = int(cfg["eda"]["max_lag_crosscorr"])
    crosscorr: dict[str, dict[str, Any]] = {}
    for tgt in targets:
        crosscorr[tgt] = {}
        for inp in inputs:
            cc = cross_correlation_lags(daily[inp].diff(), daily[tgt].diff(), max_lag)
            best = cc.iloc[cc["corr"].abs().idxmax()]
            crosscorr[tgt][inp] = {
                "best_lag": int(best["lag"]),
                "corr": round(float(best["corr"]), 4),
            }
            plots.plot_crosscorr(
                cc,
                f"Cross-corr d1 {inp}->{tgt}",
                cfg.path("figures_dir") / f"crosscorr_{tgt}_{inp}.png",
            )

    # Cointegración
    eg = [
        engle_granger(daily[tgt], daily[inp], cfg["representation"]["cointegration_significance"])
        for tgt in targets
        for inp in inputs
    ]
    joh = johansen_trace(
        daily[inputs + targets], cfg["representation"]["cointegration_significance"]
    )

    # Outliers sobre retornos (z-score)
    z_thr = float(cfg["eda"]["outlier_z_threshold"])
    rets = daily.pct_change()
    z = (rets - rets.mean()) / rets.std()
    outliers = {c: int((z[c].abs() > z_thr).sum()) for c in daily.columns}

    # Figuras
    plots.plot_series(daily, "Series en nivel", cfg.path("figures_dir") / "series_levels.png")

    eda = {
        "stationarity_levels": stat_levels.to_dict(orient="records"),
        "stationarity_diffs": stat_diffs.to_dict(orient="records"),
        "correlation_levels": corr_levels.to_dict(),
        "correlation_diffs": corr_diffs.to_dict(),
        "partial_correlation": partial_corr.to_dict(orient="records"),
        "crosscorr_best": crosscorr,
        "cointegration_engle_granger": eg,
        "cointegration_johansen": joh,
        "outliers_count": outliers,
    }
    _write_json(eda, cfg.path("artifacts_dir") / "eda.json")
    _write_csv(stat_levels, cfg.path("eda_dir") / "stationarity_levels.csv", index=False)
    _write_csv(stat_diffs, cfg.path("eda_dir") / "stationarity_diffs.csv", index=False)
    _write_csv(corr_diffs, cfg.path("eda_dir") / "correlation_diffs.csv")
    return eda


# Etapa: FEATURES (selección por votación + VIF, por equipo)
def run_features(cfg: Config) -> dict[str, Any]:
    """Selección de variables por votación y VIF, ambos POR EQUIPO."""
    set_global_seed(cfg.seed)
    frames = load_frames(cfg)
    daily = frames["daily"]
    inputs = cfg["schema"]["inputs"]
    targets = cfg["schema"]["targets"]
    input_df = daily[inputs]

    selection: dict[str, Any] = {}
    vif_out: dict[str, Any] = {}
    selected_map: dict[str, list[str]] = {}
    for tgt in targets:
        res = select_for_target(input_df, daily[tgt], cfg)
        selected_map[tgt] = res.selected
        selection[tgt] = {
            "voting_matrix": res.voting_matrix.to_dict(),
            "consensus": {k: round(float(v), 3) for k, v in res.consensus.items()},
            "best_lags": res.best_lags,
            "selected": res.selected,
            "threshold": float(cfg["selection"]["consensus_threshold"]),
        }
        plots.plot_voting_heatmap(
            res.voting_matrix, f"Votación {tgt}", cfg.path("figures_dir") / f"voting_{tgt}.png"
        )
        # VIF sobre las variables seleccionadas (en cambios), por equipo
        sel = res.selected if res.selected else inputs
        vif_df = compute_vif(difference(daily[sel]))
        vif_out[tgt] = {
            "vif": vif_df.to_dict(orient="records"),
            **vif_strategy(vif_df, float(cfg["vif"]["threshold"])),
        }

    _write_json(selection, cfg.path("artifacts_dir") / "selection.json")
    _write_json(vif_out, cfg.path("artifacts_dir") / "vif.json")
    return {"selection": selection, "vif": vif_out, "selected": selected_map}


# Etapa: TRAIN + EVALUATE
def _metrics_on_subset(
    actual_eval: pd.Series,
    y_train_level: pd.Series,
    pred: pd.Series,
    idx: pd.DatetimeIndex,
) -> dict[str, Any] | None:
    """Métricas de error + direccionales restringidas a un subconjunto temporal (test o val)."""
    common = actual_eval.index.intersection(pred.dropna().index).intersection(idx)
    if len(common) < 2:  # se necesitan ≥2 puntos para evaluar la dirección
        return None
    a, p = actual_eval.loc[common], pred.loc[common]
    return {
        "n": int(len(common)),
        "error": error_metrics(a, p, y_train_level),
        "directional": directional_metrics(a, p),
    }


def _monthly_change_frame(monthly: pd.DataFrame, tgt: str, selected: list[str]):
    """Construye (y_change, x_change, y_level) mensuales para un equipo."""
    y_level = monthly[tgt]
    y_change = y_level.diff()
    x_change = monthly[selected].diff()
    joined = pd.concat([y_change.rename("y"), x_change], axis=1).dropna()
    return joined["y"], joined[selected], y_level


def run_train_evaluate(cfg: Config) -> dict[str, Any]:
    """Entrena modelos por equipo y evalúa real vs predicho (error + direccional)."""
    set_global_seed(cfg.seed)
    frames = load_frames(cfg)
    monthly = frames["monthly"]
    targets = cfg["schema"]["targets"]
    selected_map = _selected_vars(cfg)
    split = fixed_split(monthly.index, cfg)
    eval_idx = split.test.union(split.validation)

    results: dict[str, Any] = {}
    for tgt in targets:
        selected = selected_map[tgt]
        y_change, x_change, y_level = _monthly_change_frame(monthly, tgt, selected)

        train_mask = x_change.index <= cfg["window"]["train_end"]
        xc_tr, yc_tr = x_change[train_mask], y_change[train_mask]
        eval_change_idx = x_change.index.intersection(eval_idx)
        xc_ev = x_change.loc[eval_change_idx]

        # --- Modelos de cambios ---
        en = ElasticNetChangeModel(cfg["models"]["elasticnet"]["l1_ratio"], cfg.seed).fit(
            xc_tr, yc_tr
        )
        gb = GbmChangeModel(
            cfg["models"]["gbm"]["n_estimators"],
            cfg["models"]["gbm"]["max_depth"],
            cfg["models"]["gbm"]["learning_rate"],
            cfg.seed,
        ).fit(xc_tr, yc_tr)

        preds_level: dict[str, pd.Series] = {}
        for name, model in [("elasticnet", en), ("gbm", gb)]:
            pred_change = model.predict(xc_ev)
            preds_level[name] = onestep_level_from_changes(y_level, pred_change).dropna()

        # --- SARIMAX con exógenas en nivel (one-step sobre eval) ---
        try:
            exog_cols = selected
            y_tr = y_level[y_level.index <= cfg["window"]["train_end"]]
            exog_tr = monthly.loc[y_tr.index, exog_cols]
            sx = SarimaxModel(
                tuple(cfg["models"]["sarimax"]["order"]),
                tuple(cfg["models"]["sarimax"]["seasonal_order"]),
            ).fit(y_tr, exog_tr)
            exog_ev = monthly.loc[eval_change_idx, exog_cols]
            sx_fc = sx.forecast_level(
                len(eval_change_idx),
                exog_ev,
                eval_change_idx,
                1 - cfg["forecast"]["confidence_level"],
            )
            preds_level["sarimax"] = sx_fc["pred"]
        except Exception as exc:  # pragma: no cover
            logger.warning("SARIMAX falló para %s: %s", tgt, exc)

        # --- Baseline media móvil ---
        base = MovingAverageBaseline(cfg["models"]["baseline_ma_months"])
        preds_level["baseline_ma"] = base.predict_level(y_level, eval_change_idx).dropna()

        # --- Métricas por modelo (combinado test+val, y desglosado por split) ---
        actual_eval = y_level.loc[eval_change_idx]
        y_train_level = y_level[y_level.index <= cfg["window"]["train_end"]]
        test_idx = eval_change_idx.intersection(split.test)
        val_idx = eval_change_idx.intersection(split.validation)

        model_metrics: dict[str, Any] = {}
        for name, pred in preds_level.items():
            common = actual_eval.index.intersection(pred.dropna().index)
            a, p = actual_eval.loc[common], pred.loc[common]
            model_metrics[name] = {
                "error": error_metrics(a, p, y_train_level),
                "directional": directional_metrics(a, p),
                "by_split": {
                    "test": _metrics_on_subset(actual_eval, y_train_level, pred, test_idx),
                    "validation": _metrics_on_subset(actual_eval, y_train_level, pred, val_idx),
                },
            }

        # Mejor modelo por RMSE (excluyendo baseline)
        ranked = sorted(
            [(n, m["error"]["RMSE"]) for n, m in model_metrics.items() if n != "baseline_ma"],
            key=lambda kv: kv[1],
        )
        best_model = ranked[0][0]

        # Diagnóstico de residuales del mejor modelo
        best_pred = preds_level[best_model]
        common = actual_eval.index.intersection(best_pred.dropna().index)
        resid = actual_eval.loc[common] - best_pred.loc[common]
        diag = residual_diagnostics(resid)

        # Coeficientes interpretables (ElasticNet)
        coefs = en.coefficients.round(4).to_dict()

        # Figura real vs predicho
        plots.plot_real_vs_pred(
            actual_eval,
            {k: v for k, v in preds_level.items()},
            f"Real vs predicho — {tgt} (test+val)",
            cfg.path("figures_dir") / f"real_vs_pred_{tgt}.png",
        )

        results[tgt] = {
            "selected": selected,
            "metrics": model_metrics,
            "best_model": best_model,
            "elasticnet_coefficients": coefs,
            "gbm_importances": gb.importances.round(4).to_dict(),
            "residual_diagnostics": diag,
            "eval_period": f"{eval_change_idx.min().date()}→{eval_change_idx.max().date()}",
        }

    payload = {"split": split.describe(len(monthly)), "targets": results}
    _write_json(payload, cfg.path("artifacts_dir") / "evaluation.json")
    return payload


# Etapa: FORECAST
def run_forecast(cfg: Config) -> dict[str, Any]:
    """Proyecta el costo de cada equipo al horizonte con banda de predicción."""
    set_global_seed(cfg.seed)
    frames = load_frames(cfg)
    monthly = frames["monthly"]
    targets = cfg["schema"]["targets"]
    selected_map = _selected_vars(cfg)
    horizon = int(cfg["window"]["forecast_horizon_months"])
    conf = float(cfg["forecast"]["confidence_level"])
    n_paths = int(cfg["forecast"]["n_bootstrap"])

    last_date = monthly.index.max()
    future_index = pd.date_range(last_date, periods=horizon + 1, freq="ME")[1:]

    best_models = _best_models(cfg)  # mejor modelo por equipo (de la evaluación)

    out: dict[str, Any] = {}
    for tgt in targets:
        selected = selected_map[tgt]
        y_change, x_change, y_level = _monthly_change_frame(monthly, tgt, selected)
        chosen = best_models.get(tgt, "elasticnet")

        # Se proyecta con el MEJOR modelo del equipo. SARIMAX usa su propio MC con
        # exógenas en nivel, los modelos de cambios (elasticnet/gbm) reconstruyen el nivel
        if chosen == "sarimax":
            fc = forecast_equipo_sarimax(
                y_level,
                monthly[selected],
                cfg["models"]["sarimax"]["order"],
                cfg["models"]["sarimax"]["seasonal_order"],
                horizon,
                conf,
                n_paths,
                cfg.seed,
                future_index,
            )
        else:
            if chosen == "gbm":
                model = GbmChangeModel(
                    cfg["models"]["gbm"]["n_estimators"],
                    cfg["models"]["gbm"]["max_depth"],
                    cfg["models"]["gbm"]["learning_rate"],
                    cfg.seed,
                ).fit(x_change, y_change)
            else:
                chosen = "elasticnet"
                model = ElasticNetChangeModel(
                    cfg["models"]["elasticnet"]["l1_ratio"], cfg.seed
                ).fit(x_change, y_change)
            fc = forecast_equipo(
                y_level, monthly[selected], model, horizon, conf, n_paths, cfg.seed, future_index
            )

        plots.plot_forecast(
            y_level.iloc[-36:],
            fc.frame,
            f"Proyección {tgt} — {chosen} ({horizon} meses, banda {int(conf*100)}%)",
            cfg.path("figures_dir") / f"forecast_{tgt}.png",
        )
        out[tgt] = {
            "selected": selected,
            "model": chosen,
            "horizon_months": horizon,
            "confidence": conf,
            "last_observed": {
                "date": str(last_date.date()),
                "level": round(float(y_level.iloc[-1]), 2),
            },
            "forecast": [
                {
                    "date": str(d.date()),
                    "pred": round(float(r["pred"]), 2),
                    "lower": round(float(r["lower"]), 2),
                    "upper": round(float(r["upper"]), 2),
                }
                for d, r in fc.frame.iterrows()
            ],
        }
    _write_json(out, cfg.path("artifacts_dir") / "forecast.json")
    return out


# Pipeline completo
def run_all(cfg: Config | None = None) -> None:
    """Ejecuta todas las etapas en orden."""
    cfg = cfg or load_config()
    run_data(cfg)
    run_eda(cfg)
    run_features(cfg)
    run_train_evaluate(cfg)
    run_forecast(cfg)
    logger.info("Pipeline completo. Artefactos en %s", cfg.path("artifacts_dir"))
