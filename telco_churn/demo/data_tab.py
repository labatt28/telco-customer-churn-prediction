"""Gradio 'Data Exploration' tab: lets users explore the dataset
interactively without reading any code."""

import gradio as gr
import pandas as pd

from telco_churn.config import PROCESSED_DATA_DIR
from telco_churn.plots import plot_categorical_vs_churn, plot_numeric_distribution

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport",
]


def build_data_tab() -> None:
    """Build the Data Exploration tab: stats, sample rows, interactive plots."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "telco_churn_clean.csv")

    gr.Markdown(
        "Explore the **Telco Customer Churn** dataset (7,043 customers, "
        "26.5% churn rate). Pick a column below to see how it relates to churn."
    )

    with gr.Row():
        gr.Dataframe(value=df.describe().reset_index(), label="Summary statistics")
    with gr.Row():
        gr.Dataframe(value=df.head(20), label="Sample rows")

    with gr.Row():
        numeric_dropdown = gr.Dropdown(choices=NUMERIC_COLUMNS, value="tenure", label="Numeric column")
        numeric_plot = gr.Plot(value=plot_numeric_distribution(df, "tenure"), label="Distribution")
    numeric_dropdown.change(
        fn=lambda col: plot_numeric_distribution(df, col),
        inputs=numeric_dropdown,
        outputs=numeric_plot,
    )

    with gr.Row():
        cat_dropdown = gr.Dropdown(choices=CATEGORICAL_COLUMNS, value="Contract", label="Categorical column")
        cat_plot = gr.Plot(value=plot_categorical_vs_churn(df, "Contract"), label="% Churn by category")
    cat_dropdown.change(
        fn=lambda col: plot_categorical_vs_churn(df, col),
        inputs=cat_dropdown,
        outputs=cat_plot,
    )


if __name__ == "__main__":
    # Quick standalone test: uv run --active python -m telco_churn.demo.data_tab
    # (needs data/processed/telco_churn_clean.csv to already exist --
    # run `uv run --active python -m telco_churn.dataset` first if not)
    with gr.Blocks() as _demo:
        build_data_tab()
    _demo.launch()
