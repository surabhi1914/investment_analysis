# src/config.py

# ---------------------------------------------
# Importing Libraries
# ---------------------------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"

HOUSING_DATA = RAW_DATA_DIR / "housing.csv"

DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

MODEL_DIR = PROJECT_ROOT / "models"
