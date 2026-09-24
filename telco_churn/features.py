"""Feature engineering: turns the cleaned dataset into a model-ready,
fully numeric table, split into stratified train/val/test sets.

Run as a script to regenerate the splits:
    uv run --active python -m telco_churn.features
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from telco_churn.config import ID_COLUMN, PROCESSED_DATA_DIR, RANDOM_SEED, TARGET_COLUMN

app = typer.Typer()

BINARY_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
CATEGORICAL_COLUMNS = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod",
]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the cleaned dataframe into a fully numeric one.

    Drops the customer identifier (not predictive), maps Yes/No
    columns (including the target) to 1/0, and one-hot encodes
    multi-category columns (`drop_first=True` to avoid the dummy
    variable trap). Numeric columns are left untouched here; scaling
    happens only after the train/val/test split, to avoid leaking
    val/test statistics into how train is scaled.

    Args:
        df: Cleaned dataframe, as returned by `telco_churn.dataset.clean_data`.

    Returns:
        A fully numeric dataframe, ready to be split.

    Example:
        >>> encode_features(clean_df)
    """
    df = df.drop(columns=[ID_COLUMN]).copy()

    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)
    for col in BINARY_COLUMNS:
        df[col] = (df[col] == "Yes").astype(int)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    return df


def split_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """Stratified 70/15/15 train/val/test split.

    Stratified on the target so every split keeps the same churn
    ratio as the full dataset.

    Args:
        df: Fully numeric dataframe, as returned by `encode_features`.
        val_size: Fraction of the full dataset used for validation.
        test_size: Fraction of the full dataset used for testing.

    Returns:
        A `(train, val, test)` tuple of dataframes.

    Example:
        >>> train, val, test = split_data(encoded_df)
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COLUMN], random_state=RANDOM_SEED
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val[TARGET_COLUMN],
        random_state=RANDOM_SEED,
    )
    return train, val, test


def scale_numeric(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   columns=NUMERIC_COLUMNS):
    """Standardize numeric columns, fitting the scaler on train only.

    Fitting on train only (never on val/test) prevents leaking
    information about val/test into training.

    Args:
        train: Training split.
        val: Validation split.
        test: Test split.
        columns: Numeric column names to scale.

    Returns:
        A `(train, val, test)` tuple with `columns` standardized.

    Example:
        >>> train, val, test = scale_numeric(train, val, test)
    """
    scaler = StandardScaler()
    train, val, test = train.copy(), val.copy(), test.copy()
    train[columns] = scaler.fit_transform(train[columns])
    val[columns] = scaler.transform(val[columns])
    test[columns] = scaler.transform(test[columns])
    return train, val, test


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "telco_churn_clean.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
):
    """Encode, split, and scale the cleaned dataset; save train/val/test CSVs."""
    logger.info(f"Loading cleaned data from {input_path}")
    df = pd.read_csv(input_path)

    logger.info("Encoding features...")
    df = encode_features(df)
    logger.info(f"Encoded shape: {df.shape}")

    logger.info("Splitting into train/val/test (70/15/15, stratified)...")
    train, val, test = split_data(df)
    logger.info(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    logger.info("Scaling numeric columns (fit on train only)...")
    train, val, test = scale_numeric(train, val, test)

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    logger.success(f"Saved train.csv, val.csv, test.csv to {output_dir}")


if __name__ == "__main__":
    app()
