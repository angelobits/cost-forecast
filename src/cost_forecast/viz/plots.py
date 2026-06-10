"""Gráficos reutilizables"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_series(df: pd.DataFrame, title: str, path: Path) -> Path:
    """Grafica varias series en el tiempo (normalizadas a índice 100 para comparar)"""
    fig, ax = plt.subplots(figsize=(11, 5))
    norm = df / df.iloc[0] * 100
    for col in norm.columns:
        ax.plot(norm.index, norm[col], label=col, linewidth=1.1)
    ax.set_title(f"{title} (base 100 = inicio)")
    ax.set_ylabel("Índice (base 100)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_crosscorr(cc: pd.DataFrame, title: str, path: Path) -> Path:
    """Grafica correlación cruzada por lag."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cc["lag"], cc["corr"], color="#3b6ea5")
    ax.set_title(title)
    ax.set_xlabel("lag (días)")
    ax.set_ylabel("correlación (d1)")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_voting_heatmap(voting: pd.DataFrame, title: str, path: Path) -> Path:
    """Heatmap de la matriz de votación (métodos x insumos)"""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(voting.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(voting.columns)))
    ax.set_xticklabels(voting.columns)
    ax.set_yticks(range(len(voting.index)))
    ax.set_yticklabels(voting.index, fontsize=8)
    for i in range(voting.shape[0]):
        for j in range(voting.shape[1]):
            ax.text(j, i, int(voting.values[i, j]), ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def plot_real_vs_pred(
    actual: pd.Series, predictions: dict[str, pd.Series], title: str, path: Path
) -> Path:
    """Compara la serie real contra una o más predicciones"""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(actual.index, actual.values, label="real", color="black", linewidth=1.6)
    for name, pred in predictions.items():
        ax.plot(pred.index, pred.values, label=name, linewidth=1.1, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("nivel")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_forecast(history: pd.Series, forecast: pd.DataFrame, title: str, path: Path) -> Path:
    """Grafica histórico + proyección con banda de predicción"""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(history.index, history.values, label="histórico", color="black", linewidth=1.3)
    ax.plot(forecast.index, forecast["pred"], label="proyección", color="#c44e52", linewidth=1.6)
    ax.fill_between(
        forecast.index,
        forecast["lower"],
        forecast["upper"],
        color="#c44e52",
        alpha=0.2,
        label="banda",
    )
    ax.set_title(title)
    ax.set_ylabel("nivel")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)
