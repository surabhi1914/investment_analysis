# Model Training

This document explains the baseline model experiments in [`02_preprocess_baseline.ipynb`](../notebooks/02_preprocess_baseline.ipynb). It records why each model was tried, what its results meant, and why Random Forest was selected for fine-tuning.

> **Result synchronization note:** The previous version of this document listed a Random Forest training RMSE of `18866.85405671534`, a Random Forest cross-validation mean of `50345.24055547834`, and a Decision Tree cross-validation mean of `71431.40907533008`. The notebook's latest executed outputs report `18650.698705770003`, `50435.58092066179`, and `71436.7939540488`, respectively. The results below use the latest notebook outputs as the source of truth.

## Purpose of Baseline Model Training

Baseline model training gives me a starting point. Before spending time fine-tuning one algorithm, I train several models with simple or default settings and compare their results. This helps me learn whether the problem needs a simple model, a more flexible model, or improvements to the data and features.

I do not start with fine-tuning immediately because tuning can be expensive and may optimize the wrong model family. A quick comparison first shows which model has the most potential. It also creates reference results that later experiments must improve.

The comparison included both simple and complex models:

- Linear Regression provided a simple, easy-to-understand baseline.
- Decision Tree tested whether a flexible nonlinear model could learn relationships that Linear Regression missed.
- Random Forest tested whether averaging many trees improved the unstable behavior of one tree.
- Support Vector Regression provided another nonlinear approach with a different learning method.

I used root mean squared error (RMSE) because this is a regression problem. RMSE summarizes the size of the differences between predicted and actual housing values. Lower RMSE means smaller prediction errors.

## Data Used for Training

The notebook loaded 16,512 training rows. It separated them into:

- `housing_train`: the input features.
- `housing_train_label`: the target values copied from `median_house_value`.

The target is sometimes described generally as `housing_labels`. In this notebook, its actual variable name is `housing_train_label`, and it was saved as `data/processed/housing_labels.pkl`.

The original input features were:

- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`
- `ocean_proximity`

The preprocessing pipeline:

- filled missing numeric values with each column's median;
- added `rooms_per_household`, `population_per_household`, and `bedrooms_per_room` features;
- standardized the numeric features; and
- one-hot encoded the categorical `ocean_proximity` feature.

`housing_prepared` is the transformed feature matrix produced by that pipeline. It contains numeric values in a form the models can use. Preprocessing must happen because models cannot directly learn from missing values or raw category names, and models such as SVR are sensitive to feature scales.

The fitted preprocessing pipeline was saved as `models/full_pipeline.pkl`. The prepared features and labels were also saved for later experiments.

## Metric: RMSE

RMSE means **Root Mean Squared Error**. It is calculated from the differences between predictions and true values. Squaring those differences gives larger mistakes more influence, and taking the square root returns the result to the target's original unit.

Here, the target is `median_house_value`, so RMSE is measured in dollars. An RMSE of about $68,000 means the model's typical prediction error is very large compared with the housing prices it is trying to predict. Many district values are around $120,000-$265,000, so an error of this size is not satisfying.

Lower RMSE is better, but the dataset used to calculate it matters. Training RMSE shows how well a model fits examples it already saw. Cross-validation RMSE is more useful for estimating how it may perform on unseen examples.

## Linear Regression

Linear Regression represents the prediction as a weighted combination of the input features. In simple terms, it tries to fit one overall linear relationship between the features and the target.

I tried it first because it is fast, understandable, and a good baseline. If a simple linear model performed well, a more complicated model might not be necessary.

The training result was:

```text
RMSE = 68627.87390018745
```

This error was high relative to the district housing values. Linear Regression likely could not capture nonlinear relationships and interactions in the data. Its high training error was evidence of **underfitting**, which means the model was too simple to learn the important patterns, or the available features did not provide enough information in a form the model could use.

Because the simple model underfit, I decided to try a more flexible model.

## Decision Tree

A Decision Tree makes predictions by repeatedly splitting the data into smaller groups using feature-based rules. It can learn nonlinear relationships and interactions without requiring me to define them manually.

I tried it after Linear Regression because it is much more flexible. Its training result was:

```text
RMSE = 0.0
```

A perfect training score looked impressive, but it was suspicious. An unconstrained Decision Tree can keep splitting until it memorizes individual training examples. This is **overfitting**: the model learns the training data's details and noise instead of learning patterns that transfer to new data.

A perfect training score therefore does not mean the model is good. It only means it can reproduce examples it already saw. I needed cross-validation to check its behavior on unseen validation folds.

## Random Forest

A Random Forest combines predictions from many Decision Trees. Each tree learns from a different sample of the data and considers a random subset of features at each split. Averaging the trees usually makes the final prediction more stable and less sensitive to the mistakes of one tree.

I tried Random Forest after the single Decision Tree because it keeps the tree's ability to learn nonlinear patterns while reducing some of its instability and overfitting.

The training result was:

```text
RMSE = 18650.698705770003
```

This was much lower than Linear Regression's training RMSE, and it avoided the Decision Tree's suspicious `0.0`. Random Forest therefore looked promising.

However, a low training RMSE alone was not enough to trust the model. A forest can still overfit. I needed cross-validation to estimate whether its improvement continued on examples that were not used to fit each fold's model.

## Support Vector Regression

Support Vector Regression (SVR) tries to learn a function that keeps predictions within an acceptable error margin while balancing model complexity. The notebook used an RBF kernel, which allows SVR to model nonlinear patterns.

I tested SVR as a different nonlinear alternative to the tree-based models. The notebook used its common default-scale settings:

```python
SVR(kernel="rbf", C=1.0, epsilon=0.1)
```

Its training result was:

```text
RMSE = 118578.69234925653
```

This was the worst training RMSE of the four models. SVR is sensitive to feature scaling and hyperparameter choices. The notebook standardized the input features, but the target values remained in dollars. Against target values in the hundreds of thousands, `C=1.0` and `epsilon=0.1` are very small settings. They did not give this model enough flexibility to fit the housing-value scale well.

SVR was therefore not a promising baseline with these settings. It might become competitive with target scaling and careful tuning of `C`, `epsilon`, and kernel parameters, but that would require a separate tuning effort.

## Why Cross-Validation Was Needed

Training scores can be misleading because a model is evaluated on the same examples it learned from. This is especially clear for the Decision Tree: its training RMSE was perfect, but that did not show whether it learned general patterns.

The notebook used 10-fold cross-validation. The training data was divided into 10 folds. For each run, the model trained on 9 folds, or about 90% of the data, and was evaluated on the remaining fold. This was repeated until every fold had served as validation data once.

Training and evaluating the model multiple times reduces dependence on one lucky or unlucky validation split. The mean of the 10 RMSE values gives a better estimate of generalization than training RMSE alone.

Cross-validation checks whether the model learned general patterns or only memorized the training data.

One methodology limitation should be noted: the notebook fitted `full_pipeline` on all training rows before calling `cross_val_score` on `housing_prepared`. A stricter future experiment should combine preprocessing and the estimator in one scikit-learn `Pipeline`, then cross-validate that full pipeline. This would fit imputation, scaling, and category encoding separately inside each training fold and prevent preprocessing information from carrying across folds.

## K-Fold Cross-Validation Results

| Model | Training RMSE | CV Mean RMSE | Interpretation |
| ----- | ------------: | -----------: | -------------- |
| Linear Regression | 68,627.87390018745 | 69,104.07998247063 | Both errors were high and similar, which indicates underfitting. |
| Decision Tree | 0.0 | 71,436.7939540488 | Perfect training fit but poor validation performance confirmed severe overfitting. |
| Random Forest | 18,650.698705770003 | 50,435.58092066179 | Best CV mean RMSE, although the large training-CV difference showed remaining overfitting. |
| SVR | 118,578.69234925653 | 118,584.55594251942 | Both errors were extremely high, so the default configuration performed poorly overall. |

The notebook also reported the variation across the 10 validation folds:

- Linear Regression CV standard deviation: **2,880.328210**
- Decision Tree CV standard deviation: **2,668.847349**
- Random Forest CV standard deviation: **2,203.338141**
- SVR CV standard deviation: **2,609.612082**

Random Forest had both the lowest mean CV RMSE and the lowest fold-to-fold standard deviation among these baseline experiments.

## What Cross-Validation Revealed

### Linear Regression

Linear Regression's training RMSE and CV RMSE were both high and close to each other. It did not merely fail on unseen folds; it also fit the training data poorly. This confirmed underfitting and showed that the linear model was not powerful enough for the current features and relationships.

### Decision Tree

The Decision Tree's training RMSE was `0.0`, but its CV mean RMSE rose to **71,436.793954**. This confirmed severe overfitting. The tree memorized the training examples but failed to reproduce that performance on unseen validation folds.

Cross-validation changed the interpretation completely: the apparently perfect training model generalized worse than Linear Regression.

### Random Forest

Random Forest's training RMSE of **18,650.698706** was much lower than its CV mean RMSE of **50,435.580921**. This gap showed that the forest still overfit the training data.

Even with that limitation, its CV mean RMSE was about **18,668.50** lower than Linear Regression's and **21,001.21** lower than the Decision Tree's. Random Forest generalized better than every other tested baseline, making it the strongest candidate for fine-tuning.

### SVR

SVR's training and CV RMSE were both around **118,000**. The similar values showed that its poor result was not mainly caused by a train-validation gap; it performed poorly on both seen and unseen data with the chosen defaults. It was not selected for further tuning at this stage.

## Model Selection Decision

Random Forest was selected as the model to fine-tune because it achieved the best cross-validation RMSE among the models tested. Linear Regression underfit, Decision Tree overfit severely, and SVR performed poorly with its initial settings.

Random Forest still showed some overfitting because its training RMSE was much lower than its cross-validation RMSE. Even so, it generalized better than the alternatives. It was therefore the strongest baseline and the best candidate for regularization, Grid Search, and Randomized Search.

The next step was not to trust the baseline as final. It was to tune Random Forest hyperparameters such as the number of trees, the number of features considered at each split, tree depth, and minimum leaf size.

## Thought Process Behind Each Decision

I started with Linear Regression because it is simple, fast, and easy to understand. Its high training and validation errors showed me that it underfit, so I tried a more powerful nonlinear model.

The Decision Tree looked perfect on the training data, but that result was suspicious. Cross-validation confirmed that it was overfitting badly. This taught me that a perfect training score can hide poor performance on new examples.

I tried Random Forest because it combines many trees and averages their predictions. It improved validation performance substantially compared with both Linear Regression and the single Decision Tree. It still overfit, but it had the best cross-validation result.

I also tested SVR because it offers a different nonlinear approach. It performed poorly, likely because the default-scale `C` and `epsilon` settings were not suitable for housing targets measured in hundreds of thousands of dollars.

Cross-validation helped me avoid trusting misleading training scores. I chose Random Forest because it had the best validation performance, not because it had the best training performance.

## What I Learned

- Training score alone is not enough to choose a model.
- A perfect training score can be a warning sign instead of a success.
- Cross-validation is important for checking how well a model generalizes.
- Underfitting means the model is too simple, the features are not informative enough, or both.
- Overfitting means the model learned training-specific details that do not transfer well to new data.
- Random Forest was the best baseline because it had the lowest mean CV RMSE.
- Even the best baseline may still need regularization and hyperparameter tuning.
- Model selection should be based mainly on validation performance, not training performance.
- A fair cross-validation workflow should fit preprocessing separately within each training fold.

## Decision Rules for Future Projects

- Start with a simple baseline model that is easy to understand.
- Compare more than one model family before investing in fine-tuning.
- If training and validation errors are both high, suspect underfitting.
- If training error is very low but validation error is high, suspect overfitting.
- Do not trust a model only because it performs well on training data.
- Use cross-validation before choosing a model.
- Cross-validate the complete preprocessing-and-model pipeline when possible.
- Choose the model with the best validation performance, not the best training performance.
- If the best model still overfits, tune or regularize it.
- Save model artifacts, predictions, metrics, and experiment settings for later comparison.

## Saved Model Artifacts

The notebook saved an experiment dictionary for each baseline. Each dictionary was intended to contain the fitted model, all cross-validation scores, mean RMSE, and training predictions.

The following files currently exist in `models/`:

| Artifact | Intended contents |
| -------- | ----------------- |
| `lin_reg_v1.pkl` | Linear Regression model, CV results, and predictions |
| `decision_tree_v1.pkl` | Decision Tree model, CV results, and predictions |
| `random_forest_v1.pkl` | Random Forest model, CV results, and predictions |
| `svr_v1.pkl` | SVR model, CV results, and predictions |

Saving these experiment files is useful because I can reload a fitted model without retraining it, reproduce comparisons, and inspect exactly which results supported a decision. Saving predictions also makes it possible to build an error dashboard, compare errors across model types, and inspect the examples each model handled well or poorly.

There is one important issue to correct before relying on every artifact: the notebook's Decision Tree save cell stores `"model_object": forest_tree` while pairing it with the Decision Tree CV scores and predictions. Therefore, although `models/decision_tree_v1.pkl` exists, its model object should be treated as incorrect until that cell uses `tree` and the artifact is regenerated. The other three save cells reference their expected model objects.

The fitted preprocessing pipeline is stored separately as `models/full_pipeline.pkl`. Keeping the preprocessing pipeline with the model artifacts supports reproducibility because future data must be transformed in the same way as the training data.

## Final Summary

- Linear Regression underfit: its training and CV errors were both high.
- Decision Tree overfit: its training RMSE was `0.0`, but its CV error was high.
- SVR performed poorly with the initial hyperparameters.
- Random Forest achieved the best mean cross-validation RMSE at **50,435.580921**.
- Random Forest was selected for fine-tuning even though it still showed overfitting.
- The next step was hyperparameter tuning and regularization to reduce overfitting and improve validation and final test performance.
