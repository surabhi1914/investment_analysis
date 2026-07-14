# About the Data and Preprocessing

## Purpose of This Document

This document summarizes what the California Housing dataset contains, what I learned during exploratory data analysis (EDA), which data-quality issues I found, and how those findings guided preprocessing before model training.

EDA and preprocessing are connected. EDA helps me notice missing values, skewed distributions, capped values, useful relationships, and features that need a different representation. Preprocessing then turns those observations into repeatable data-cleaning and feature-transformation steps.

The main sources are [`01_housing_EDA.ipynb`](../notebooks/01_housing_EDA.ipynb) and [`02_preprocess_baseline.ipynb`](../notebooks/02_preprocess_baseline.ipynb). The second notebook is the repository's available counterpart to the requested `02_preprocess_and_baseline.ipynb` name.

## Dataset Overview

The dataset describes California census block groups. Each row contains location, housing, population, household, income, and ocean-proximity information for one block group. The goal is to use those input features to predict that block group's median house value.

| Item | Value | Notes |
| ---- | ----- | ----- |
| Dataset name | California Housing Prices | Housing and demographic data for California block groups |
| Number of rows | 20,640 | Confirmed by the EDA notebook |
| Number of columns | 10 | Nine numeric columns and one categorical column |
| Target variable | `median_house_value` | The continuous value the model must predict |
| Input variables | 9 columns | Every original column except `median_house_value` |
| Numerical columns | 9 total | Eight numerical inputs plus the numerical target |
| Categorical columns | `ocean_proximity` | Text categories must be encoded for modeling |
| Missing-value columns | `total_bedrooms` | 207 values are missing in the full dataset |

**Dependent variable / target:** `median_house_value`

**Independent variables / features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, and `ocean_proximity`.

This is a supervised regression problem. It is supervised because the training data includes the correct target values, and it is regression because `median_house_value` is a continuous number rather than a category.

## Column Summary

| Column | Type | Missing Values | Meaning / Notes |
| ------ | ---- | -------------: | --------------- |
| `longitude` | Numeric (`float64`) | 0 | East-west geographic coordinate. Location can help the model identify regional price patterns. |
| `latitude` | Numeric (`float64`) | 0 | North-south geographic coordinate. It has a negative linear relationship with the target in this dataset. |
| `housing_median_age` | Numeric (`float64`) | 0 | Median age of homes in the block group. Its maximum and visible spike at 52 suggest capping. |
| `total_rooms` | Numeric (`float64`) | 0 | Total rooms in the block group. It is right-skewed and strongly affected by block-group size. |
| `total_bedrooms` | Numeric (`float64`) | 207 | Total bedrooms in the block group. Only 20,433 of 20,640 values are non-null, so imputation is required. |
| `population` | Numeric (`float64`) | 0 | Number of people in the block group. It is right-skewed and partly measures area size. |
| `households` | Numeric (`float64`) | 0 | Number of households in the block group. It is related to population, rooms, and bedrooms. |
| `median_income` | Numeric (`float64`) | 0 | Median income measure for the block group. It is the strongest obvious linear predictor of house value. |
| `median_house_value` | Numeric (`float64`) | 0 | Target value to predict. Its maximum is 500,001, with a large spike around $500,000. |
| `ocean_proximity` | Categorical text | 0 | Location category relative to the ocean. It requires numerical encoding. |

## Target Variable: `median_house_value`

`median_house_value` is the value the model is trying to predict. The EDA notebook reports values from **$14,999** to **$500,001**, with a median of **$179,700**.

The distribution has a large spike near $500,000, and the maximum is $500,001. This strongly suggests that high values were capped when the dataset was created.

If the dataset records many expensive houses as exactly $500,000, the model may not know the true values above that cap. Districts with meaningfully different high-end prices can appear to have the same target. This hides information from the model, limits how much it can learn about expensive areas, and may lead to underprediction when the model is used on uncapped high-value properties.

![Feature distributions, including median house value](images/image2.png)

The image shows the notebook's numeric histograms. The final panel contains the clear target spike at the upper boundary. It also shows the spike at 52 for `housing_median_age` and the long right tails of several count features.

## Exploratory Data Analysis Observations

### Housing Median Age

`housing_median_age` has a visible spike at 52, and 52 is also the maximum reported value. This suggests that 52 may mean "52 or older" rather than exactly 52 for every row.

This matters because a model sees every value of 52 as identical. It cannot distinguish a block group with a true median housing age of 52 from one whose true median might be 60 or 80. The capped feature therefore loses detail about older housing areas.

### Room, Bedroom, Population, and Household Counts

`total_rooms`, `total_bedrooms`, `population`, and `households` are right-skewed. Most block groups have moderate counts, while a small number have extremely large counts. For example, the full dataset has maximums of 39,320 rooms, 6,445 bedrooms, 35,682 people, and 6,082 households.

These raw totals are also connected. A larger block group naturally tends to have more rooms, bedrooms, people, and households. As a result, the totals may describe block-group size more than living conditions.

Ratio features can be more meaningful:

- Rooms per household estimates average home size more directly than total rooms.
- Bedrooms per room describes the composition of the available rooms.
- Population per household provides a rough measure of household density.

This is why the preprocessing workflow added ratio-based features rather than relying only on raw totals.

### Median Income

`median_income` is right-skewed, with most observations concentrated in the lower and middle part of its range and fewer observations at high values.

It is strongly positively related to `median_house_value`. This relationship makes practical sense: areas with higher household purchasing power can often support higher housing prices. The EDA results made median income especially important both as a model feature and as the basis for a representative train/test split.

### Ocean Proximity and Location

Longitude and latitude place each block group on the California map. The geographic plot uses point size for population and color for median house value.

Higher values appear around coastal and high-demand regions, including the Los Angeles, San Diego, and San Francisco areas. Inland and Central Valley areas generally appear lower priced. These observations support keeping both coordinates and `ocean_proximity` as model inputs.

![Geographic housing-value pattern](images/image-1.png)

Location matters because two districts with similar housing and population counts may have very different prices when one is near the coast or a major employment center and the other is inland.

### Target Capping

The large `median_house_value` spike around $500,000 is not a normal smooth tail. It suggests that many values at or above a limit were stored at nearly the same maximum.

This can limit model performance because no algorithm can learn differences that are missing from the labels. Prediction errors in expensive areas may remain high, and a model trained on capped labels may underestimate true values above the cap. This limitation should be remembered when interpreting RMSE and high-value residuals.

## Correlation Analysis

Correlation measures the direction and strength of a **linear** relationship between two numeric variables:

- A positive value means the variables tend to increase together.
- A negative value means one tends to decrease as the other increases.
- A value near zero means there is little linear relationship, although a nonlinear relationship may still exist.

The training-set correlation output reported:

| Feature | Correlation with `median_house_value` | Interpretation |
| ------- | ------------------------------------: | -------------- |
| `median_income` | 0.687151 | Strongest positive relationship among the original inputs |
| `total_rooms` | 0.135140 | Weak positive linear relationship |
| `housing_median_age` | 0.114146 | Weak positive linear relationship |
| `households` | 0.064590 | Very weak positive linear relationship |
| `total_bedrooms` | 0.047781 | Very weak positive linear relationship |
| `population` | -0.026882 | Almost no linear relationship |
| `longitude` | -0.047466 | Very weak negative linear relationship by itself |
| `latitude` | -0.142673 | Negative relationship; values tend to be lower farther north in this dataset |

`median_income` is clearly the strongest original linear signal. Raw bedroom, household, and population counts have weak correlations because their meaning is mixed with block-group size.

Weak linear correlation does not mean a feature is useless. Location works through combinations of longitude and latitude, and nonlinear models can learn thresholds and interactions that a single correlation coefficient cannot show.

After feature engineering, `rooms_per_household` had a correlation of **0.146255**, while `bedrooms_per_rooms` had a correlation of **-0.259952** in the EDA notebook. These results show that ratios can reveal relationships hidden by raw totals.

![Median income compared with median house value](images/image.png)

This image is the notebook's income-versus-house-value scatter plot. The upward pattern supports the positive correlation, while the horizontal line at the top also makes the target cap visible. The repository does not contain a separate saved correlation heatmap image; the notebook reports the correlations as a sorted numeric series.

## Key EDA Takeaways

- The dataset has 207 missing values in `total_bedrooms`.
- Several numerical features are right-skewed and contain very large observations.
- `housing_median_age` and `median_house_value` appear capped.
- Raw count features mostly reflect block-group size, so ratio-based feature engineering is useful.
- Location and ocean proximity contain important housing-price information.
- `median_income` is the strongest obvious predictor in the correlation results.
- Target capping may limit prediction quality, especially for expensive areas.

## Train/Test Split Strategy

The EDA notebook explored simple and random splitting, then created the saved datasets with **stratified sampling**. It used `StratifiedShuffleSplit` with a 20% test size and `random_state=42`.

A purely random split can accidentally place too many high- or low-income districts in one subset. That would make the test set less representative and could make evaluation results change more than expected from one split to another.

Median income was used for stratification because EDA showed that it has the strongest relationship with house value. The notebook created a temporary `income_cat` by dividing `median_income` by 1.5, rounding upward, and combining all categories 5 and above into category 5. These categories represented income bands, not a new model feature.

The notebook then sampled train and test rows so that the income-category proportions remained close to those of the full dataset. Finally, it removed `income_cat` from both sets and saved them.

The saved split sizes are:

- Training set: **16,512 rows** and 10 columns
- Test set: **4,128 rows** and 10 columns

Keeping the test set separate provides a fair final evaluation on data that did not influence model fitting or preprocessing decisions.

## Separating Features and Labels

The preprocessing notebook loaded the stratified training set and separated it into:

```python
housing_train = strat_train.drop("median_house_value", axis=1)
housing_train_label = strat_train["median_house_value"].copy()
```

`housing_train` contains the nine input features, often called **X**. `housing_train_label` contains only `median_house_value`, often called **y**. The target artifact is saved as `housing_labels.pkl`, so the documentation may also refer to these values generally as `housing_labels`.

This separation prevents the correct answer from being included among the input features. If the target were accidentally passed through preprocessing and into the model inputs, it would cause target leakage and produce misleadingly strong results.

## Missing Value Handling

`total_bedrooms` has 20,433 non-null values out of 20,640, so **207 values are missing**. Most scikit-learn estimators cannot directly train on missing numerical values.

The notebook used:

```python
SimpleImputer(strategy="median")
```

Median imputation replaces a missing value with the training column's median. The median is a sensible choice for these right-skewed features because extremely large values affect it less than they affect the mean.

The stratified split was created before preprocessing. The imputer was then fitted on the training features, so it learned training-set medians rather than using the protected test data. After imputation, all eight numerical training columns had 16,512 non-null values.

The rule for future data is: **fit preprocessing on training data only, then transform validation and test data with that fitted preprocessing.**

## Categorical Encoding

`ocean_proximity` contains text categories, but machine-learning estimators require numerical inputs. The full dataset's confirmed categories and counts are:

| Category | Rows |
| -------- | ---: |
| `<1H OCEAN` | 9,136 |
| `INLAND` | 6,551 |
| `NEAR OCEAN` | 2,658 |
| `NEAR BAY` | 2,290 |
| `ISLAND` | 5 |

The preprocessing notebook first explored `LabelEncoder`, which assigns one integer to each category. That representation is risky for unordered categories because a model may treat category 4 as meaningfully greater than category 1, even though ocean-proximity labels have no natural numeric order.

The final preprocessing instead used `OneHotEncoder`. It creates one binary column per category. A row receives 1 in its matching category column and 0 in the others. This gives the model numeric inputs without creating a fake ranking between categories.

Ocean proximity is likely useful because the geographic EDA showed different price patterns between coastal and inland areas.

## Feature Engineering

The EDA notebook tested ratio features, and the preprocessing notebook implemented them in a reusable custom transformer named `CombinedAttributesAdder`.

Raw totals often describe block size, while ratios describe living conditions and density more directly.

### `rooms_per_household`

```text
rooms_per_household = total_rooms / households
```

`total_rooms` alone tends to be larger for larger block groups. Dividing by households estimates the average number of rooms available per household and is therefore closer to average home size.

### `population_per_household`

```text
population_per_household = population / households
```

Population alone may simply mean that a block group is large. Population per household gives a rough measure of household occupancy or density.

### `bedrooms_per_room`

```text
bedrooms_per_room = total_bedrooms / total_rooms
```

This ratio describes room composition. It may help distinguish areas with different housing layouts even when their total room counts are similar. The EDA notebook stored this column as `bedrooms_per_rooms`; the custom transformer uses the clearer internal name `bedrooms_per_room`.

The numerical pipeline calls `CombinedAttributesAdder()` with its default setting, so all three ratio features are included in the final prepared data.

## Feature Scaling

The numerical pipeline used `StandardScaler`. Standardization centers each numerical feature and scales it according to its variation, which puts differently sized measurements on more comparable scales.

Scaling is especially important for distance- and margin-based methods such as SVR, where a feature with large raw values could otherwise dominate the calculation. Linear Regression can also benefit from consistent numerical conditioning. Random Forest is much less sensitive to scale, but keeping scaling in one shared pipeline makes preprocessing consistent across the baseline models.

## Preprocessing Pipeline

A pipeline records the preprocessing steps in the order they must happen. This reduces manual mistakes, makes experiments reproducible, and ensures that future rows receive the same transformations as the training data.

The numerical pipeline contained:

1. `SimpleImputer(strategy="median")` to fill missing numeric values.
2. `CombinedAttributesAdder()` to add the three ratio features.
3. `StandardScaler()` to standardize the numerical columns.

The full pipeline used `ColumnTransformer` to apply:

- the numerical pipeline to the eight numerical input columns; and
- `OneHotEncoder` to `ocean_proximity`.

It then combined both outputs into one prepared feature matrix. Fitting this workflow on training data and reusing the fitted object for validation, test, and future predictions helps prevent data leakage. The fitted pipeline was saved as `models/full_pipeline.pkl`.

## Final Prepared Data

`housing_prepared` is the transformed feature matrix created by applying the full pipeline to `housing_train`. It contains imputed and standardized numerical features, the three engineered ratios, and one-hot category columns.

`housing_train_label` contains the corresponding `median_house_value` targets. The notebook saved the prepared inputs as `data/processed/housing_prepared_x.pkl` and the labels as `data/processed/housing_labels.pkl`. These prepared training data were then used for baseline model training.

The notebook confirms that the input training set has **16,512 rows**, but it does not display `housing_prepared.shape`. The exact final matrix shape should be confirmed from the notebook output or saved artifact before it is recorded as a result.

The same fitted `full_pipeline` should be loaded and used to transform the untouched test data and future raw prediction inputs. Refitting preprocessing on those datasets would make transformations inconsistent and could leak information.

## What I Learned

- I learned that EDA helps reveal problems before modeling, including missing values, skewness, caps, and unusual observations.
- I learned that high training performance is not useful if preprocessing is incorrect or leaks information.
- I learned that missing values must be handled carefully and that the replacement values should be learned from training data.
- I learned that categorical variables need numerical encoding before most models can use them.
- I learned that label encoding can create fake ordered relationships between categories.
- I learned that ratio features can describe living conditions better than raw totals that mainly measure area size.
- I learned that capped targets can hide real differences and limit model performance.
- I learned that train/test splitting should preserve important distributions such as median income.
- I learned that preprocessing should be fitted only on training data and reused without refitting on validation or test data.

## Decision Rules for Future Projects

- Always inspect dataset shape, columns, data types, and missing values first.
- Identify the target and remove it from the input features before preprocessing.
- Check the target distribution before modeling.
- Look for capped values, skewness, and outliers.
- Use median imputation when numerical data is skewed or contains influential outliers.
- Avoid label encoding for unordered categories.
- Use `OneHotEncoder` for nominal categorical variables.
- Create ratio features when raw counts mostly represent size.
- Use stratification when an important variable must remain representative across splits.
- Use pipelines to make preprocessing repeatable and reduce data-leakage risk.
- Fit preprocessing only on training data.
- Transform validation and test data with the already-fitted pipeline.

## Final Summary

- The California Housing dataset has **20,640 rows** and **10 columns**.
- The target is `median_house_value`, making this a supervised regression problem.
- `total_bedrooms` contains **207 missing values**.
- Several features are right-skewed, while housing age and the target appear capped.
- EDA showed that income, geographic location, and ocean proximity are important signals.
- Preprocessing handled missing values, categorical encoding, ratio feature engineering, and numerical scaling.
- The final prepared training data were used for baseline model training, and the fitted pipeline was saved for consistent reuse.
