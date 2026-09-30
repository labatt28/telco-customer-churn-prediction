"""Convenience script for local testing: `uv run --active python app.py`.
The real entry point (used by PyPI/uvx) is telco_churn.demo.app:main.
"""
from telco_churn.demo.app import main

if __name__ == "__main__":
    main()
