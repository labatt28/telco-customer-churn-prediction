"""Project-wide configuration: paths and dataset-specific constants.

Centralizing these means notebooks and scripts never hardcode
filesystem locations -- they just do
``from telco_churn.config import ...``.
"""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
"""Root directory of the project (two levels up from this file)."""
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
"""Root directory for all project data."""

RAW_DATA_DIR = DATA_DIR / "raw"
"""Original, immutable input data. Not tracked in git -- see the
'Data' section in the README for how to download it."""

INTERIM_DATA_DIR = DATA_DIR / "interim"
"""Intermediate data that has been transformed but is not yet
model-ready."""

PROCESSED_DATA_DIR = DATA_DIR / "processed"
"""Final, model-ready datasets: the train/val/test splits produced
by `telco_churn.features`."""

EXTERNAL_DATA_DIR = DATA_DIR / "external"
"""Data from third-party sources (unused in this project so far)."""

MODELS_DIR = PROJ_ROOT / "models"
"""Trained model checkpoints, produced by `telco_churn.modeling.train`."""

REPORTS_DIR = PROJ_ROOT / "reports"
"""Generated analysis, including the final report."""

FIGURES_DIR = REPORTS_DIR / "figures"
"""Generated performance visualizations, produced by
`telco_churn.modeling.predict`."""

ID_COLUMN = "customerID"
"""Name of the customer identifier column in the raw dataset."""

TARGET_COLUMN = "Churn"
"""Name of the binary target column (`"Yes"`/`"No"`) in the raw dataset."""

RANDOM_SEED = 42
"""Seed used everywhere a random operation needs to be reproducible
(train/val/test split, model initialization)."""

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
