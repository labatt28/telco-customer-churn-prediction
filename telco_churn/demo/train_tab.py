"""Gradio 'Training Interface' tab: train the model with
user-adjustable hyperparameters, with live progress feedback."""

import gradio as gr
import lightning as L
import pandas as pd

from telco_churn.config import MODELS_DIR, PROCESSED_DATA_DIR
from telco_churn.modeling.train import ChurnClassifier, make_dataloader


class _GradioProgressCallback(L.Callback):
    """Lightning callback that reports epoch progress to a Gradio
    gr.Progress() instance, so the UI shows a live progress bar."""

    def __init__(self, progress: gr.Progress, max_epochs: int):
        self.progress = progress
        self.max_epochs = max_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        self.progress(epoch / self.max_epochs, desc=f"Epoch {epoch}/{self.max_epochs}")


def _train(lr: float, max_epochs: int, batch_size: int, progress=gr.Progress()):
    """Train a fresh ChurnClassifier with the given hyperparameters.

    Reuses the exact same model class and data-loading helper as
    `telco_churn.modeling.train` -- no duplicated logic.
    """
    progress(0, desc="Loading data...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")

    train_loader = make_dataloader(train_df, batch_size=int(batch_size), shuffle=True)
    val_loader = make_dataloader(val_df, batch_size=int(batch_size), shuffle=False)

    model = ChurnClassifier(input_dim=train_df.shape[1] - 1, lr=lr)
    trainer = L.Trainer(
        max_epochs=int(max_epochs),
        enable_progress_bar=False,
        logger=False,
        callbacks=[_GradioProgressCallback(progress, int(max_epochs))],
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)

    checkpoint_path = MODELS_DIR / "demo_user_trained.ckpt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint_path)

    metrics = trainer.callback_metrics
    summary = (
        f"**Training complete.**\n\n"
        f"- Train loss: {metrics.get('train_loss', 0):.3f}, "
        f"accuracy: {metrics.get('train_acc', 0):.1%}\n"
        f"- Val loss: {metrics.get('val_loss', 0):.3f}, "
        f"accuracy: {metrics.get('val_acc', 0):.1%}\n\n"
        f"Switch to the **Model Evaluation** tab to see it evaluated "
        f"on the held-out test set."
    )
    return str(checkpoint_path), summary


def build_train_tab(checkpoint_state: gr.State) -> None:
    """Build the Training Interface tab.

    Args:
        checkpoint_state: Shared gr.State holding the path of the
            currently active model checkpoint. Updated with the new
            checkpoint's path once training finishes, so the Model
            Evaluation tab picks it up automatically.
    """
    gr.Markdown(
        "Train a small neural network (PyTorch Lightning) on the "
        "Telco Churn training set. This is fast (a few seconds) "
        "thanks to the tiny model and dataset -- feel free to "
        "experiment with the hyperparameters below."
    )
    with gr.Row():
        lr_slider = gr.Slider(
            1e-4, 1e-1, value=1e-3, label="Learning rate",
            info="Step size for the Adam optimizer",
        )
        epochs_slider = gr.Slider(1, 50, value=15, step=1, label="Epochs")
        batch_slider = gr.Slider(8, 256, value=64, step=8, label="Batch size")

    train_btn = gr.Button("Train model", variant="primary")
    summary_md = gr.Markdown()

    train_btn.click(
        fn=_train,
        inputs=[lr_slider, epochs_slider, batch_slider],
        outputs=[checkpoint_state, summary_md],
    )


if __name__ == "__main__":
    # Quick standalone test: uv run --active python -m telco_churn.demo.train_tab
    with gr.Blocks() as _demo:
        state = gr.State(value="")
        build_train_tab(state)
    _demo.launch()
