"""Model evaluation: loads a trained checkpoint, runs it on the test
set, and generates the performance visualizations required by the
assignment (confusion matrix, calibration curve, training curves,
and the highest-loss samples).
"""

from pathlib import Path

import pandas as pd
import torch
import typer
from loguru import logger

from telco_churn.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TARGET_COLUMN
from telco_churn.modeling.train import ChurnClassifier
from telco_churn.plots import plot_calibration_curve, plot_confusion_matrix, plot_training_curves

app = typer.Typer()


def save_highest_loss_samples(
    test_df: pd.DataFrame,
    y_true,
    probs,
    save_path: Path = FIGURES_DIR / "highest_loss_samples.csv",
    top_n: int = 10,
) -> pd.DataFrame:
    """Save the `top_n` test samples with the highest binary cross-entropy loss.

    Useful for spotting patterns in the model's biggest mistakes (see
    `reports/report.md` for the analysis of these specific customers).

    Args:
        test_df: The test-set dataframe (features + target).
        y_true: Ground-truth labels (0/1), shape (n_samples,).
        probs: Predicted probabilities for the positive class.
        save_path: Where to save the resulting CSV.
        top_n: How many highest-loss samples to keep.

    Returns:
        A dataframe with the top_n rows, sorted by loss (descending).
    """
    eps = 1e-7
    p = probs.clip(eps, 1 - eps)
    loss = -(y_true * pd.Series(p).apply(lambda v: __import__("math").log(v)) +
             (1 - y_true) * pd.Series(p).apply(lambda v: __import__("math").log(1 - v)))
    out = test_df.copy()
    out["predicted_prob"] = probs
    out["loss"] = loss.to_numpy()
    out = out.sort_values("loss", ascending=False).head(top_n)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    return out


@app.command()
def main(
    checkpoint_path: Path = MODELS_DIR / "churn_model.ckpt",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
):
    """Evaluate the trained model on the test set and save all performance figures.

    Loads the checkpoint saved by `telco_churn.modeling.train`, runs
    inference on `test.csv`, and writes the confusion matrix,
    calibration curve, training curves, and highest-loss samples to
    `reports/figures/`.

    Args:
        checkpoint_path: Path to the trained Lightning checkpoint.
        test_path: Path to the processed test-set CSV.
    """
    logger.info(f"Loading model from {checkpoint_path}")
    model = ChurnClassifier.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    y_true = test_df[TARGET_COLUMN].to_numpy()
    X_test = torch.tensor(test_df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype="float32"))

    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).numpy()
    preds = (probs > 0.5).astype(int)

    plot_confusion_matrix(y_true, preds)
    logger.success("Saved confusion_matrix.png")

    plot_calibration_curve(y_true, probs)
    logger.success("Saved calibration_curve.png")

    candidates = sorted((MODELS_DIR / "training_logs").glob("version_*/metrics.csv"))
    if candidates and plot_training_curves(candidates[-1]) is not None:
        logger.success("Saved training_curves.png")
    else:
        logger.warning("No metrics.csv found, skipping training curves plot.")

    save_highest_loss_samples(test_df, y_true, probs)
    logger.success("Saved highest_loss_samples.csv")

    logger.info(f"Test accuracy: {(preds == y_true).mean():.3f}")


if __name__ == "__main__":
    app()
