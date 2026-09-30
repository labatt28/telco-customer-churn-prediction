"""Gradio 'Model Evaluation' tab: evaluate the currently active
model checkpoint on the held-out test set."""

import gradio as gr
import pandas as pd
import torch

from telco_churn.config import PROCESSED_DATA_DIR, TARGET_COLUMN
from telco_churn.modeling.train import ChurnClassifier
from telco_churn.plots import plot_calibration_curve, plot_confusion_matrix


def _evaluate(checkpoint_path: str):
    """Load a checkpoint, run it on the test set, and build the
    evaluation visualizations. Reuses the exact same inference logic
    as `telco_churn.modeling.predict`.
    """
    model = ChurnClassifier.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    y_true = test_df[TARGET_COLUMN].to_numpy()
    X_test = torch.tensor(test_df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype="float32"))

    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).numpy()
    preds = (probs > 0.5).astype(int)

    cm_fig = plot_confusion_matrix(y_true, preds, save_path=None)
    cal_fig = plot_calibration_curve(y_true, probs, save_path=None)

    acc = (preds == y_true).mean()
    metrics_md = f"**Test accuracy:** {acc:.1%} &nbsp;&nbsp;&nbsp; **Test samples:** {len(y_true)}"

    sample_table = pd.DataFrame({
        "true_label": ["Churn" if v else "No churn" for v in y_true[:10]],
        "predicted_label": ["Churn" if v else "No churn" for v in preds[:10]],
        "predicted_probability": probs[:10].round(3),
    })

    return cm_fig, cal_fig, metrics_md, sample_table


def build_eval_tab(checkpoint_state: gr.State) -> None:
    """Build the Model Evaluation tab.

    Args:
        checkpoint_state: Shared gr.State holding the path of the
            currently active model checkpoint (set by the Training
            Interface tab, or the app's default pre-trained model).
            Re-evaluates automatically whenever this changes.
    """
    gr.Markdown(
        "Evaluates the **currently active model** on the held-out "
        "test set (1,057 customers it never saw during training). "
        "By default this is the app's pre-trained model; if you "
        "train a new one in the **Training Interface** tab, this "
        "updates automatically."
    )
    with gr.Row():
        cm_plot = gr.Plot(label="Confusion Matrix")
        cal_plot = gr.Plot(label="Calibration Curve")
    metrics_md = gr.Markdown()
    samples_table = gr.Dataframe(label="Sample predictions (first 10 test customers)")
    refresh_btn = gr.Button("Re-evaluate current model")

    outputs = [cm_plot, cal_plot, metrics_md, samples_table]
    refresh_btn.click(fn=_evaluate, inputs=checkpoint_state, outputs=outputs)
    checkpoint_state.change(fn=_evaluate, inputs=checkpoint_state, outputs=outputs)


if __name__ == "__main__":
    # Quick standalone test: uv run --active python -m telco_churn.demo.eval_tab
    # (needs an existing checkpoint -- run
    # `uv run --active python -m telco_churn.modeling.train` first if you
    # don't have models/churn_model.ckpt yet)
    from telco_churn.config import MODELS_DIR

    with gr.Blocks() as _demo:
        state = gr.State(value=str(MODELS_DIR / "churn_model.ckpt"))
        build_eval_tab(state)
    _demo.launch()
