"""Reusable plotting functions for model evaluation.

Centralizing these here — instead of writing matplotlib code inline in
`modeling/predict.py` — means any future script (or the EDA notebook)
can reuse the exact same functions instead of duplicating plotting
code. Same DRY principle used throughout the rest of the project.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from telco_churn.config import FIGURES_DIR


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = FIGURES_DIR / "confusion_matrix.png",
):
    """Plot and save a confusion matrix for binary churn predictions.

    Args:
        y_true: Ground-truth labels (0/1), shape (n_samples,).
        y_pred: Predicted labels (0/1), shape (n_samples,).
        save_path: Where to save the figure as a PNG.

    Returns:
        The matplotlib Figure.

    Example:
        >>> plot_confusion_matrix(y_true, preds)
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion Matrix (test set)")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path | None = FIGURES_DIR / "confusion_matrix.png",
    n_bins: int = 10,
):
    """Plot and save a calibration curve (reliability diagram).

    Args:
        y_true: Ground-truth labels (0/1), shape (n_samples,).
        y_prob: Predicted probabilities for the positive class.
        save_path: Where to save the figure as a PNG.
        n_bins: Number of probability bins to group predictions into.

    Returns:
        The matplotlib Figure.

    Example:
        >>> plot_calibration_curve(y_true, probs)
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=n_bins, ax=ax)
    ax.set_title("Calibration Curve (test set)")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_training_curves(
    metrics_csv: Path,
    save_path: Path = FIGURES_DIR / "training_curves.png",
):
    """Plot loss and accuracy per epoch from a Lightning CSVLogger output.

    Args:
        metrics_csv: Path to the `metrics.csv` file written by
            `lightning.pytorch.loggers.CSVLogger` during training.
        save_path: Where to save the figure as a PNG.

    Returns:
        The matplotlib Figure, or `None` if `metrics_csv` doesn't exist.

    Example:
        >>> plot_training_curves(MODELS_DIR / "training_logs/version_0/metrics.csv")
    """
    if not metrics_csv.exists():
        return None

    hist = pd.read_csv(metrics_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "acc"]):
        for stage in ["train", "val"]:
            col = f"{stage}_{metric}"
            if col in hist.columns:
                sub = hist.dropna(subset=[col])
                ax.plot(sub["epoch"], sub[col], marker="o", label=stage)
        ax.set_title(f"{metric.capitalize()} per epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig

def plot_numeric_distribution(df: pd.DataFrame, column: str):
    """Histogram of a single numeric column, for the interactive demo.

    Args:
        df: Dataframe containing `column`.
        column: Name of the numeric column to plot.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[column], bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def plot_categorical_vs_churn(df: pd.DataFrame, column: str):
    """Bar chart of % churn rate per category, for the interactive demo.

    Args:
        df: Dataframe containing `column` and a "Churn" column
            (values "Yes"/"No").
        column: Name of the categorical column to break down by.

    Returns:
        The matplotlib Figure.
    """
    churn_rate = df.groupby(column)["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
    churn_rate = churn_rate.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(churn_rate.index, churn_rate.values, color="#DD8452")
    ax.set_title(f"% churn by {column}")
    ax.set_ylabel("% of customers who churned")
    ax.set_xlabel(column)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    return fig
