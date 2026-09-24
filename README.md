# Telco Customer Churn Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

ML project that predicts whether a customer will churn based on account and service usage data

--------

## Authors
 Alberto Bellera
 Iñaki Garatea
 Pablo Labat

--------

## Data

This project uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset (Kaggle, originally an IBM Sample Data Set), licensed CC BY 4.0.
The raw CSV is not tracked in this repository (excluded via
`.gitignore`) to keep the repo lightweight.

To get it:
1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the Kaggle
   link above (requires a free Kaggle account).
2. Place it at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Then continue with "How to run" below.

--------

## Status

- [x] Project structure (Cookiecutter Data Science)
- [x] Dataset integration (`telco_churn/dataset.py`)
- [x] Exploratory Data Analysis (`notebooks/1.0-eda.ipynb`)
- [x] Feature engineering / preprocessing (`telco_churn/features.py`)
- [x] Model training (`telco_churn/modeling/train.py`)
- [x] Performance evaluation (`telco_churn/modeling/predict.py`)
- [x] Final report (`reports/report.pdf`)
- [x] Code documentation (Google-style docstrings, all modules)
- [x] HTML documentation (`pdoc`, see `docs/`)
- [x] Pinned dependencies (`uv`, verified on a clean environment)

--------

## Environment setup

This project uses `uv` for dependency and environment management.
Everything needed to reproduce the environment is declared in
`pyproject.toml` (and locked in `uv.lock`) -- no external environment
or lab-specific setup is required.

    uv sync

This creates a `.venv/` and installs every dependency (including a
CUDA-enabled PyTorch build from PyPI), pinned to the versions in
`uv.lock`. Verified end-to-end on a completely clean machine (see
"Dependency management" below).

### Known dependency issue: `torchvision`

`lightning` transitively pulls in `torchmetrics`, which -- purely for
its optional image-metric functions, which this project never uses --
imports `torchvision`. On this environment, the `torchvision` wheel
that resolves is binary-incompatible with the installed `torch`
build (`RuntimeError: operator torchvision::nms does not exist`).
Since we never import `torchvision` directly, we exclude it outright
in `pyproject.toml`:

    [tool.uv]
    exclude-dependencies = ["torchvision"]

This requires `uv >= 0.9.8` (`uv self update` if you hit an "unknown
field" error on this setting).

--------

## How to run

Run in order — each step reads the output of the previous one:

    # 1. Clean the raw data
    uv run python -m telco_churn.dataset

    # 2. Build model-ready features (train/val/test splits)
    uv run python -m telco_churn.features

    # 3. Explore the data (optional, for inspection)
    uv run jupyter lab
    # then open notebooks/1.0-eda.ipynb

    # 4. Train the model
    uv run python -m telco_churn.modeling.train

    # 5. Evaluate and generate performance figures
    uv run python -m telco_churn.modeling.predict

Outputs: cleaned/processed data in `data/processed/`, the trained
checkpoint in `models/churn_model.ckpt`, and performance figures in
`reports/figures/`.

--------

## Documentation

API documentation is generated from the project's docstrings
(Google style) using [`pdoc`](https://pdoc.dev/), and stored as a
static HTML site in `docs/`.

To regenerate it after changing any docstrings:

    uv run pdoc telco_churn -o docs

To browse it locally with live-reload instead:

    uv run pdoc telco_churn --http localhost:8080

--------

## Dependency management

Dependencies are declared in `pyproject.toml` and managed with `uv`.
Version constraints follow a consistent strategy:

- Packages with a stable major version (>=1.0) are pinned as
  `package~=X.Y` — fixes the major version, allows minor/patch
  updates (PEP 440 compatible release: `~=2.14` means `>=2.14,<3.0`).
- Packages still in `0.x` (`seaborn`, `loguru`, `typer`) are pinned
  more conservatively as `package~=0.Y.Z` (patch-only updates), since
  pre-1.0 releases don't guarantee API stability between minor
  versions.
- Only direct imports are listed as dependencies (e.g. `torchmetrics`
  was removed — it's never imported directly, only pulled in
  automatically by `lightning`).

Reproducibility was verified end-to-end on a completely clean
environment (no pre-existing packages, no lab-specific setup): a bare
`uv sync` correctly resolves and installs every dependency, including
a working CUDA-enabled PyTorch build, and the full pipeline
(`dataset` → `features` → `train` → `predict`) runs to completion
with matching results.

--------

## Project Organization

├── LICENSE <- Open-source license if one is chosen
├── Makefile <- Makefile with convenience commands like make data or make train
├── README.md <- The top-level README for developers using this project.
├── data
│ ├── external <- Data from third party sources.
│ ├── interim <- Intermediate data that has been transformed.
│ ├── processed <- The final, canonical data sets for modeling.
│ └── raw <- The original, immutable data dump.
│
├── docs <- pdoc-generated HTML API documentation
│
├── models <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks <- Jupyter notebooks. Naming convention is a number (for ordering),
│ the creator's initials, and a short - delimited description, e.g.
│ 1.0-jqp-initial-data-exploration.
│
├── pyproject.toml <- Project configuration file with package metadata for
│ telco_churn and configuration for tools like black
│
├── references <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports <- Generated analysis as HTML, PDF, LaTeX, etc.
│ └── figures <- Generated graphics and figures to be used in reporting
│
├── requirements.txt <- The requirements file for reproducing the analysis environment, e.g.
│ generated with pip freeze > requirements.txt
│
├── setup.cfg <- Configuration file for flake8
│
└── telco_churn <- Source code for use in this project.
│
├── init.py <- Makes telco_churn a Python module
│
├── config.py <- Store useful variables and configuration
│
├── dataset.py <- Scripts to download or generate data
│
├── features.py <- Code to create features for modeling
│
├── modeling
│ ├── init.py
│ ├── predict.py <- Code to run model inference with trained models
│ └── train.py <- Code to train models
│
└── plots.py <- Code to create visualizations

--------

## Project progress

This project was built following a complete software-engineering
workflow, from raw data to a trained, evaluated, and documented model:

- **Environment & dependencies**: managed with `uv`. Dependencies are
  version-pinned (`~=`) and reproducibility was verified on a clean
  environment (see "Dependency management" above).
- **Project structure**: generated with the official
  `cookiecutter-data-science` (`ccds`) tool, as recommended in the
  assignment.
- **Data ingestion & cleaning** (`telco_churn/dataset.py`): loads the
  raw Kaggle CSV and fixes a data quality issue found during
  inspection (11 blank `TotalCharges` values, all `tenure == 0`
  customers).
- **Exploratory Data Analysis** (`notebooks/1.0-eda.ipynb`): target
  distribution, feature distributions, feature-vs-churn relationships,
  and a correlation heatmap, each interpreted in writing.
- **Feature engineering** (`telco_churn/features.py`): categorical
  encoding, stratified 70/15/15 split, leakage-safe scaling.
- **Model training** (`telco_churn/modeling/train.py`): a small MLP
  (~1.5K parameters) trained with PyTorch Lightning, reaching 79.2%
  test accuracy (vs. a 73.5% naive baseline).
- **Performance evaluation** (`telco_churn/modeling/predict.py` +
  `telco_churn/plots.py`): confusion matrix, calibration curve,
  training curves, and highest-loss-sample analysis, all saved to
  `reports/figures/`.
- **Report** (`reports/report.pdf` / `reports/report.md`): a 2-page
  report interpreting all performance visualizations.
- **Code documentation**: Google-style docstrings across every
  module, with a working `pdoc` HTML export in `docs/`.

--------

## Contributing

1. Create a feature branch from `main`:
       git checkout -b feature/your-change-name
2. Make your changes. Keep commits focused and use descriptive
   messages.
3. Before opening a pull request, run the formatting and linting
   tools:
       uv run black telco_churn/
       uv run isort telco_churn/
       uv run flake8 telco_churn/
4. If you changed any docstrings, regenerate the HTML documentation
   so it stays in sync:
       uv run pdoc telco_churn -o docs
5. Push your branch and open a Pull Request against `main` on
   GitHub. Briefly describe what changed and why.
6. Once reviewed and merged, delete the feature branch.

Please do not commit directly to `main`, and never commit the raw
dataset, API keys, or other sensitive data (see `.gitignore`).
  
