"""Model training: a small MLP trained with PyTorch Lightning to
predict customer churn from the processed, model-ready features.

Run as a script to train and checkpoint the model:
    uv run --active python -m telco_churn.modeling.train
"""

from pathlib import Path

import lightning as L
import pandas as pd
import torch
import typer
from loguru import logger
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from telco_churn.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, TARGET_COLUMN

app = typer.Typer()

L.seed_everything(RANDOM_SEED)


class ChurnClassifier(L.LightningModule):
    """Small MLP for binary churn classification.

    Attributes:
        net: 3-layer feedforward network (input_dim -> 32 -> 16 -> 1).
        loss_fn: Binary cross-entropy with logits.
    """

    def __init__(self, input_dim: int, lr: float = 1e-3):
        """Initialize the model.

        Args:
            input_dim: Number of input features (columns, excluding target).
            lr: Learning rate for the Adam optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        """Return raw logits (before sigmoid) for a batch of inputs."""
        return self.net(x).squeeze(-1)

    def _shared_step(self, batch, stage: str):
        """Compute loss/accuracy for one batch and log them under `stage`."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Lightning hook: one training batch. See `_shared_step`."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Lightning hook: one validation batch. See `_shared_step`."""
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        """Return the Adam optimizer used for training."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def make_dataloader(df: pd.DataFrame, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    """Wrap a processed dataframe into a PyTorch DataLoader.

    Args:
        df: Processed dataframe (features + target column).
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle (True for training, False otherwise).

    Returns:
        A DataLoader yielding (X, y) float32 tensor batches.

    Example:
        >>> train_loader = make_dataloader(train_df, batch_size=64, shuffle=True)
    """
    y = torch.tensor(df[TARGET_COLUMN].to_numpy(dtype="float32"))
    X = torch.tensor(df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype="float32"))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    val_path: Path = PROCESSED_DATA_DIR / "val.csv",
    max_epochs: int = 15,
    batch_size: int = 64,
):
    """Train the model and save a checkpoint plus per-epoch metrics."""
    logger.info("Loading processed train/val splits...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_loader = make_dataloader(train_df, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_df, batch_size=batch_size, shuffle=False)

    input_dim = train_df.shape[1] - 1  # all columns except the target
    model = ChurnClassifier(input_dim=input_dim)

    csv_logger = CSVLogger(save_dir=MODELS_DIR, name="training_logs")
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=csv_logger,
        accelerator="auto",
        log_every_n_steps=10,
    )

    logger.info(f"Training on {len(train_df)} samples, validating on {len(val_df)}...")
    trainer.fit(model, train_loader, val_loader)

    checkpoint_path = MODELS_DIR / "churn_model.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    logger.success(f"Model checkpoint saved to {checkpoint_path}")
    logger.info(f"Training metrics logged to {csv_logger.log_dir}/metrics.csv")


if __name__ == "__main__":
    app()
