# Investment Analysis: California Housing Price Modeling

A notebook-driven machine learning project that explores California housing data and compares regression models for predicting district-level median house values.

## Problem Statement

Real estate investment decisions depend on understanding which location and district-level factors are associated with higher housing values. This project uses the California Housing Prices dataset to explore those patterns and build regression models that estimate `median_house_value` from demographic, geographic, and housing attributes.

The goal is not to provide financial advice or a production investment system. It is a practical ML workflow for data exploration, preprocessing, model training, evaluation, and experiment tracking.

## What This Project Does

- Downloads the California housing dataset from the Hands-On ML public dataset source.
- Performs exploratory data analysis on housing attributes, missing values, geographic patterns, and correlations.
- Creates a stratified train/test split using income categories to preserve income distribution.
- Builds a preprocessing pipeline for numeric and categorical features.
- Engineers ratio-based features such as rooms per household and population per household.
- Trains and evaluates several regression models.
- Saves model experiment artifacts as `.pkl` files for later comparison.

## Key Features

- **Reproducible data fetch:** `src/fetch_data.py` downloads and extracts `housing.tgz` into `data/raw/`.
- **EDA notebook:** `notebooks/01_housing_EDA.ipynb` documents dataset structure, distributions, geographic patterns, and correlations.
- **Preprocessing pipeline:** median imputation, custom feature engineering, standard scaling, and one-hot encoding are composed with scikit-learn pipelines.
- **Model comparison:** Linear Regression, Decision Tree, Random Forest, and Support Vector Regression are trained and evaluated.
- **Cross-validation:** 10-fold RMSE validation is used to compare model generalization.
- **Experiment artifacts:** saved model dictionaries include the model object, predictions, CV scores, and mean RMSE.

## Technical Architecture / Workflow

```text
Download data
    -> Explore raw dataset
    -> Create stratified train/test split
    -> Separate features and target
    -> Impute missing numeric values
    -> Engineer ratio features
    -> Scale numeric features
    -> One-hot encode ocean proximity
    -> Train regression models
    -> Evaluate with RMSE and cross-validation
    -> Save experiment artifacts
```

The workflow is currently implemented primarily in notebooks, with shared project paths centralized in `src/config.py`.

## Repository Structure

```text
investment_analysis/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- docs/
|   |-- about_data.md
|   |-- data_preprocessing.md
|   |-- training.md
|   `-- images/
|-- notebooks/
|   |-- 01_housing_EDA.ipynb
|   `-- 02_preprocess_train_eval_fine_tune.ipynb
|-- src/
|   |-- config.py
|   `-- fetch_data.py
|-- data/        # ignored; raw and split data generated locally
`-- models/      # ignored; saved model artifacts generated locally
```

## Tech Stack

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- SciPy
- joblib
- Jupyter / ipykernel

## Setup Instructions

1. Clone the repository and move into the project folder.

   ```bash
   git clone <repository-url>
   cd investment_analysis
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project

1. Download the dataset.

   ```bash
   python -m src.fetch_data
   ```

   This creates `data/raw/housing.tgz` and extracts `data/raw/housing.csv`.

2. Run the EDA notebook in VS Code, Jupyter, or another notebook environment.

   This explores the raw dataset and creates stratified train/test files:

   - `data/raw/strat_train.csv`
   - `data/raw/strat_test.csv`

3. Run the preprocessing, training, and evaluation notebook.

   This builds the preprocessing pipeline, trains models, evaluates RMSE, and saves model artifacts under `models/`.

## Example Workflow

After setup, a typical project run is:

```bash
python -m src.fetch_data
```

Then execute these notebooks in order:

1. `notebooks/01_housing_EDA.ipynb`
2. `notebooks/02_preprocess_train_eval_fine_tune.ipynb`

The second notebook compares model behavior and stores experiment outputs such as:

- `models/lin_reg_v1.pkl`
- `models/decision_tree_v1.pkl`
- `models/random_forest_v1.pkl`
- `models/svr_v1.pkl`

## Data Sources / Inputs

The project uses the California Housing Prices dataset from the public Hands-On ML dataset repository:

```text
https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz
```

Dataset details documented in `docs/about_data.md`:

- 20,640 rows and 10 columns
- Target: `median_house_value`
- Features include longitude, latitude, housing age, room counts, population, households, median income, and ocean proximity
- `total_bedrooms` contains missing values
- `ocean_proximity` is categorical

## Outputs / Results

The repository includes EDA visuals in `docs/images/`:

![Housing feature histograms](docs/images/image2.png)

![Geographic housing value and population view](docs/images/image-1.png)

![Median income vs. median house value](docs/images/image.png)

Training notes in `docs/training.md` and notebook outputs show the following model comparison:

| Model | Training RMSE | 10-Fold CV Mean RMSE | Notes |
| --- | ---: | ---: | --- |
| Linear Regression | 68,627.87 | 69,104.08 | Underfits relative to more flexible models |
| Decision Tree | 0.00 | 71,604.87 | Training score indicates overfitting |
| Random Forest | 18,653.85 | 50,319.44 | Best CV RMSE among documented models |
| Support Vector Regression | 118,578.69 | 118,584.56 | Performs poorly with current default-style configuration |

## Evaluation Metrics

The main evaluation metric is **Root Mean Squared Error (RMSE)**, computed on training predictions and through 10-fold cross-validation. RMSE is appropriate here because the target is a continuous housing value and larger prediction errors should be penalized more heavily.

## Screenshots / Demo

Current demo assets are static EDA plots stored in `docs/images/`. A future version could add:

- TODO: add a concise notebook walkthrough screenshot
- TODO: add an inference example using a saved model artifact

## Known Limitations

- The project is notebook-first and does not yet provide a production inference API or command-line training pipeline.
- The dataset target appears capped around `$500,000`, which can limit model quality for high-value districts.
- The Decision Tree overfits the training data when unconstrained.
- The Random Forest still shows signs of overfitting because its training RMSE is much lower than its cross-validation RMSE.
- Fine-tuning is started as a section in the training notebook, but no completed tuning workflow is documented yet.
- Data and model artifacts are ignored by Git, so they must be regenerated locally.

## Future Improvements

- Add hyperparameter tuning for Random Forest and SVR.
- Evaluate the final selected model on the held-out stratified test set.
- Package preprocessing and model training into reusable scripts.
- Add a model inference example for a single district record.
- Track experiments in a structured results file.
- Add tests for data loading, feature engineering, and pipeline construction.

## Learning / Project Highlights

- Practiced an end-to-end supervised learning workflow on a real tabular dataset.
- Used stratified sampling to preserve median-income distribution across train and test splits.
- Built scikit-learn preprocessing pipelines with custom feature engineering.
- Compared underfitting, overfitting, and ensemble model behavior using RMSE and cross-validation.
- Documented EDA observations, preprocessing choices, and model evaluation results.

## Author

Surabhi Nair

This project is licensed under the MIT License. See `LICENSE` for details.
