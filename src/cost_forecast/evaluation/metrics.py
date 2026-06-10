"""Métricas de error y métricas direccionales (signo del cambio).

* **Error de nivel**: MAE, RMSE, MAPE, R², MASE (escala-independiente).
* **Direccional**: sobre el signo del cambio mes a mes (sube/baja) se reportan
  precisión, recall, F1 y matriz de confusión, porque para el negocio importa
  tanto *cuánto* como *en qué dirección* se mueve el costo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def error_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_train: pd.Series | None = None
) -> dict[str, float]:
    """Calcula MAE, RMSE, MAPE, R² y (si hay train) MASE.

    Args:
        y_true: Valores reales (nivel).
        y_pred: Valores predichos (nivel), alineados con ``y_true``.
        y_train: Serie de entrenamiento para el denominador naïve del MASE.

    Returns:
        Diccionario de métricas redondeadas.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yt - yp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / yt))) * 100
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_%": round(mape, 3),
        "R2": round(r2, 4),
    }
    if y_train is not None and len(y_train) > 1:
        naive = float(np.mean(np.abs(np.diff(np.asarray(y_train, dtype=float)))))
        out["MASE"] = round(mae / naive, 4) if naive > 0 else float("nan")
    return out


def directional_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, object]:
    """Métricas sobre el signo del cambio (sube=1 / baja=0) entre pasos consecutivos."""
    true_dir = (y_true.diff().dropna() > 0).astype(int)
    pred_dir = (pd.Series(np.asarray(y_pred), index=y_true.index).diff().dropna() > 0).astype(int)
    aligned = pd.concat([true_dir, pred_dir], axis=1).dropna()
    yt, yp = aligned.iloc[:, 0], aligned.iloc[:, 1]
    if len(yt) == 0:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    accuracy = float((yt.values == yp.values).mean())
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
        "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }
