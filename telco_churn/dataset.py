"""Data loading and cleaning for the Telco Customer Churn dataset.

Run as a script to regenerate the cleaned CSV:
    uv run --active python -m telco_churn.dataset
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from telco_churn.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def download_raw_data(
    dest_path: Path = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    repo_id: str = "labatt28/telco-customer-churn-raw",
) -> Path:
    """Download the raw CSV from its Hugging Face Dataset mirror if
    it's not already present locally.

    Used by the interactive demo (telco_churn/demo/) so a fresh user
    doesn't need to manually download anything from Kaggle first.

    Args:
        dest_path: Where the CSV should end up.
        repo_id: The Hugging Face Dataset repo hosting the CSV.

    Returns:
        The path to the (now guaranteed to exist) CSV file.

    Example:
        >>> download_raw_data()
    """
    if dest_path.exists():
        return dest_path

    import shutil
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="WA_Fn-UseC_-Telco-Customer-Churn.csv",
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(downloaded, dest_path)
    return dest_path

def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV, unmodified.

    Args:
        input_path: Path to the raw CSV file.

    Returns:
        The raw dataframe, exactly as read from disk (no cleaning).

    Example:
        >>> load_raw_data(RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    """
    return pd.read_csv(input_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data quality issues.

    `TotalCharges` is read as text because 11 brand-new customers
    (tenure == 0) have a blank value instead of a number, since they
    haven't been billed yet. We convert the column to numeric and fill
    those 11 rows with 0.

    Args:
        df: Raw dataframe as returned by `load_raw_data`.

    Returns:
        Cleaned dataframe, with `TotalCharges` as a proper numeric column.

    Example:
        >>> clean_data(raw_df)
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    output_path: Path = PROCESSED_DATA_DIR / "telco_churn_clean.csv",
):
    """Clean the raw dataset and save the result to data/processed/."""
    logger.info(f"Loading raw data from {input_path}")
    df = load_raw_data(input_path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    logger.info("Cleaning data...")
    df = clean_data(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Cleaned dataset saved to {output_path}")


if __name__ == "__main__":
    app()
