# src/fetch_data.py

"""
This file will be used to download the data from the web and extract csv and save in the data folder for futher analysis
"""

# ---------------------------------------------
# Importing Libraries
# ---------------------------------------------
from src.config import RAW_DATA_DIR
import os
import urllib.request
import tarfile
from pathlib import Path

# ---------------------------------------------
# Setup
# ---------------------------------------------
WEBSOURCE = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_URL = f"{WEBSOURCE}datasets/housing/housing.tgz"
RAW_DATA_DIR = Path(RAW_DATA_DIR)


# ---------------------------------------------
# Function definition
# ---------------------------------------------
def fetch_housing_data(housing_url=HOUSING_URL):
    if not RAW_DATA_DIR.exists():
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
    tgz_path = RAW_DATA_DIR / "housing.tgz"
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=RAW_DATA_DIR, filter="data")
    housing_tgz.close()


# ---------------------------------------------
# call the function
# ---------------------------------------------
fetch_housing_data(HOUSING_URL)
