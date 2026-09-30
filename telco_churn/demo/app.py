"""Gradio demo entry point (the PyPI console-script target, run via uvx)."""

import pandas as pd
import lightning as L
import gradio as gr

from telco_churn.config import MODELS_DIR, PROCESSED_DATA_DIR
from telco_churn.dataset import clean_data, download_raw_data, load_raw_data
from telco_churn.demo.data_tab import build_data_tab
from telco_churn.demo.eval_tab import build_eval_tab
from telco_churn.demo.train_tab import build_train_tab
from telco_churn.features import encode_features, scale_numeric, split_data
from telco_churn.modeling.train import ChurnClassifier, make_dataloader

DEFAULT_CHECKPOINT = MODELS_DIR / "demo_default.ckpt"


def ensure_data_ready() -> None:
    """Download + clean + featurize the dataset if not already done."""
    csv_path = download_raw_data()
    if not (PROCESSED_DATA_DIR / "train.csv").exists():
        df = clean_data(load_raw_data(csv_path))
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_DIR / "telco_churn_clean.csv", index=False)

        df = encode_features(df)
        train, val, test = scale_numeric(*split_data(df))
        train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        val.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
        test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)


def ensure_model_ready() -> None:
    """Train a default model once if no checkpoint exists yet, so the
    Model Evaluation tab has something to show immediately."""
    if DEFAULT_CHECKPOINT.exists():
        return
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    model = ChurnClassifier(input_dim=train_df.shape[1] - 1)
    trainer = L.Trainer(max_epochs=15, enable_progress_bar=False, logger=False, enable_checkpointing=False)
    trainer.fit(
        model,
        make_dataloader(train_df, shuffle=True),
        make_dataloader(val_df, shuffle=False),
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(DEFAULT_CHECKPOINT)


ensure_data_ready()
ensure_model_ready()

def main():
    with gr.Blocks(title="Telco Customer Churn Prediction") as demo:
        gr.Markdown(
            "# Telco Customer Churn Prediction\n"
            "Explore the data, train the model with your own hyperparameters, "
            "and evaluate its performance -- all without touching any code."
        )
        checkpoint_state = gr.State(value=str(DEFAULT_CHECKPOINT))
        with gr.Tabs():
            with gr.Tab("Data Exploration"):
                build_data_tab()
            with gr.Tab("Training Interface"):
                build_train_tab(checkpoint_state)
            with gr.Tab("Model Evaluation"):
                build_eval_tab(checkpoint_state)
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
