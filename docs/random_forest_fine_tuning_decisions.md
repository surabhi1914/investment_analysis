# Random Forest Fine-Tuning Decisions

This document explains how the Random Forest model was tuned, how the candidate models were compared, and why the final model was selected. The results come from the executed outputs in [`02_preprocess_baseline.ipynb`](../notebooks/02_preprocess_baseline.ipynb) and [`03_fine_tune.ipynb`](../notebooks/03_fine_tune.ipynb).

## Purpose of Fine-Tuning

Fine-tuning means changing a model's hyperparameters and measuring which combination works best on data that was not used to fit that version of the model. Hyperparameters are settings chosen before training. They control behavior such as how many trees the forest contains, how deep each tree may grow, and how much flexibility each tree has.

I tuned the Random Forest because its default settings are general-purpose choices, not settings designed specifically for this housing dataset. A default model may use too much complexity and memorize the training data, or it may use too little complexity and miss useful patterns. Tuning lets me search for a better balance.

Cross-validation was used to compare settings more reliably. In each search, five-fold cross-validation repeatedly trained a candidate on part of the training data and evaluated it on a held-out fold. This is more dependable than judging a model from one training score or one validation split.

The final test set was kept separate during training and hyperparameter selection. It was used only at the end to check how the shortlisted models performed on unseen data. After looking at test results, I should not continue tuning against that test set. Reusing it for model decisions would gradually turn it into another validation set and make its score less honest.

## Metrics Used

All errors below are root mean squared error (RMSE). RMSE measures the typical size of prediction errors in the target's units and gives extra weight to large mistakes. Lower RMSE is better.

- **Training RMSE** measures error on the training portions used during cross-validation. A very low value shows that the model fits its training data well, but it does not prove that the model will work well on new data.
- **Validation RMSE** measures error on held-out cross-validation folds. This was the main metric for choosing hyperparameters because it estimates performance on unseen data without using the final test set.
- **Test RMSE** measures error on the final test set. It is the last, independent check of the selected approach.
- **Train-validation gap** is `validation RMSE - training RMSE`. A large positive gap means the model performs much better on its training data than on held-out data, which is evidence of overfitting.

The practical interpretation rules were:

```text
Low train RMSE + much higher validation RMSE = overfitting
High train RMSE + high validation RMSE = possible underfitting
Low validation RMSE + reasonable gap = better generalization
Final test RMSE = final unbiased model check, if it is used only once at the end
```

These rules must be considered together. For example, the smallest gap is not automatically the best model if both errors are worse. Similarly, the lowest training RMSE is not automatically useful if validation performance does not improve.

## Baseline / Earlier Random Forest Result

The baseline notebook trained this default model:

```python
RandomForestRegressor(random_state=42)
```

Its executed output reported:

- Training RMSE: **18,650.698706**
- Mean 10-fold cross-validation RMSE: **50,435.580921**
- Cross-validation RMSE standard deviation: **2,203.338141**

The much lower training RMSE than cross-validation RMSE showed that the baseline Random Forest was overfitting. However, its cross-validation performance was still substantially better than the earlier Linear Regression, Decision Tree, and SVR results in the same notebook, so Random Forest remained a promising model to tune.

The first simple Grid Search then tested:

```python
param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]
```

It selected `n_estimators=30` and `max_features=8`, with a best five-fold validation RMSE of **49,898.989135**. Both selected values were at the upper edge of their ranges. That result motivated a wider second grid rather than assuming those boundaries were optimal.

## Grid Search 2: Expanding Random Forest Hyperparameters

The second grid expanded the number of trees and added controls for tree depth and leaf size:

```python
param_grid_2 = [
    {
        "n_estimators": [50, 100, 200],
        "max_features": [8, 10, 12],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 2, 4],
    },
]
```

Each setting had a specific purpose:

- `n_estimators` controls the number of trees. More trees usually make the forest's average prediction more stable, although they also increase training and prediction cost.
- `max_features` controls how many input features each split may consider. Using fewer features can make the trees less similar to one another and can improve the benefit of averaging them.
- `max_depth` limits how deep each tree can grow. Deep trees can learn detailed relationships, while shallower trees are a form of regularization.
- `min_samples_leaf` sets the minimum number of training examples allowed in a leaf. Larger leaves smooth predictions and reduce flexibility; a value of 1 allows the most detailed leaves in this grid.

The best parameters were:

```python
{
    "max_depth": 40,
    "max_features": 8,
    "min_samples_leaf": 1,
    "n_estimators": 200,
}
```

The notebook's best result row was:

- Training RMSE: **18,334.562817**
- Validation RMSE: **49,183.392802**
- Train-validation gap: **30,848.829985**
- Cross-validation rank: **1**

This improved validation RMSE by about **715.60** compared with the first simple Grid Search. The selected `n_estimators=200` was the largest tested value, suggesting that the next search should try more trees. The selected `max_features=8` was the smallest tested value, suggesting that fewer features per split should be explored. Depth 40 beat both depth 20 and unlimited depth for this grid, so deep trees helped, but removing the depth limit was not best. The explicit best-parameter output confirmed `min_samples_leaf=1`, meaning the model still preferred flexible leaves.

The gap remained large, so the model was still overfitting even though validation RMSE improved. I therefore made two follow-up decisions: search more closely around the promising performance boundaries, and separately test a more regularized grid designed to reduce the gap.

## Grid Search 3: Focused Search Around Better Ranges

The third grid followed the boundary signals from Grid Search 2:

```python
param_grid_3 = [
    {
        "n_estimators": [200, 300, 500],
        "max_features": [4, 6, 8],
        "max_depth": [30, 40, 50],
        "min_samples_leaf": [1, 2],
    }
]
```

I expanded `n_estimators` upward because 200 had been best at the upper edge. I lowered the `max_features` range because 8 had been best at the lower edge. I searched depths 30, 40, and 50 to examine the area around the previous best depth of 40. I kept `min_samples_leaf` at `[1, 2]` because leaf size 1 had won previously, while size 2 offered a nearby, slightly more regularized alternative.

The best parameters were:

```python
{
    "max_depth": 30,
    "max_features": 6,
    "min_samples_leaf": 1,
    "n_estimators": 500,
}
```

Its results were:

- Training RMSE: **18,109.880445**
- Validation RMSE: **48,997.196572**
- Train-validation gap: **30,887.316127**
- Test RMSE: **46,779.889360**

Validation RMSE improved by about **186.20** compared with Grid Search 2. The model chose more trees, fewer features, a somewhat shallower depth, and the flexible leaf size of 1. The improvement was real but modest, and the gap increased by about **38.49**, so the focused search improved predictive performance without solving the overfitting problem.

## Overfitting-Reduction Grid Search

I also ran a deliberately more regularized search:

```python
param_grid_reduce_overfit = [
    {
        "n_estimators": [200, 300],
        "max_features": [4, 6, 8],
        "max_depth": [20, 30, 40],
        "min_samples_leaf": [2, 4, 6],
    }
]
```

This grid reduced the maximum tree depth and removed `min_samples_leaf=1`. Starting leaf size at 2 forced every terminal leaf to represent more than one training example. Together, these settings limited the trees' ability to learn very specific patterns and noise from the training data.

The best parameters were:

```python
{
    "max_depth": 20,
    "max_features": 6,
    "min_samples_leaf": 2,
    "n_estimators": 300,
}
```

Its results were:

- Training RMSE: **24,870.728611**
- Validation RMSE: **49,231.836207**
- Train-validation gap: **24,361.107595**
- Test RMSE: **46,953.639299**

The higher training RMSE was expected because the model was less flexible. The gap fell by about **6,526.21** compared with Grid Search 3, showing that regularization made training and validation behavior more similar. However, validation RMSE became about **234.64** worse and test RMSE became about **173.75** worse.

This experiment taught me that reducing overfitting can improve stability without improving the final error. The regularized model had the smallest gap of the three finalists, but it also had the highest validation and test RMSE. It was a useful, more stable alternative, not the best-performing model.

## Randomized Search

After the focused grids, I used Randomized Search to explore a broader set of combinations without evaluating every possible combination. Grid Search tries every combination in a fixed list. Randomized Search instead samples a fixed number of combinations from ranges or distributions. This makes it practical to cover more hyperparameters and larger numeric ranges for a similar search budget.

The notebook used 50 sampled combinations, five-fold cross-validation, and `random_state=42`. Its search space was:

```python
param_distributions = {
    "n_estimators": randint(200, 801),
    "max_features": randint(3, 13),
    "max_depth": [None, 15, 20, 25, 30, 35, 40, 50, 60],
    "min_samples_leaf": randint(1, 7),
    "min_samples_split": randint(2, 13),
    "bootstrap": [True, False],
}
```

The full best parameters, confirmed from the notebook output, were:

```python
{
    "bootstrap": False,
    "max_depth": 60,
    "max_features": 5,
    "min_samples_leaf": 1,
    "min_samples_split": 3,
    "n_estimators": 537,
}
```

Its results were:

- Training RMSE: **2,439.249383**
- Validation RMSE: **48,306.409897**
- Train-validation gap: **45,867.160513**
- Test RMSE: **45,991.615914**

Randomized Search found a combination outside the exact manual grids, including 537 trees, `bootstrap=False`, and `min_samples_split=3`. It achieved the lowest validation and test RMSE in the final comparison. Its test RMSE was about **788.27** lower than Grid Search 3 and about **962.02** lower than the overfitting-reduction model.

The extremely low training RMSE and very large gap are still serious warning signs. With `bootstrap=False`, deep trees, and `min_samples_leaf=1`, the model fit the training data very strongly. Its held-out validation and test results nevertheless remained the best of the candidates, so the overfitting warning is a limitation rather than a reason to replace it with a demonstrably worse candidate.

## Final Model Comparison

| Model | Train RMSE | Validation RMSE | Gap | Test RMSE | Main Interpretation |
| ----- | ---------: | --------------: | --: | --------: | ------------------- |
| Randomized Search | 2,439.249383 | 48,306.409897 | 45,867.160513 | 45,991.615914 | Best validation and test error, but strongest overfitting and largest gap. |
| Grid Search 3 | 18,109.880445 | 48,997.196572 | 30,887.316127 | 46,779.889360 | Strongest focused-grid result; more accurate than the regularized model but less accurate than Randomized Search. |
| Overfitting-Reduction Grid Search | 24,870.728611 | 49,231.836207 | 24,361.107595 | 46,953.639299 | Smallest gap and most stable behavior, but slightly worse validation and test error. |

The differences between the finalists are much smaller on validation and test data than their training errors suggest. This is why the final decision was based primarily on held-out performance, while the gap was recorded as a risk.

## Final Decision

The **Randomized Search Random Forest** was selected as the final model because it achieved the lowest validation RMSE and the lowest final test RMSE. The validation result had already ranked it first before the test set was examined, and the final test result confirmed the same ordering.

Randomized Search won despite its overfitting because neither of the more regularized alternatives improved held-out performance. Its validation RMSE was **48,306.409897**, and its final test RMSE was **45,991.615914**. The close agreement between these two held-out scores is encouraging, even though the training RMSE of **2,439.249383** and gap of **45,867.160513** show that its training behavior does not generalize proportionally.

The final result means that, among the candidates tested, this model produced the best predictions on both cross-validation folds and the final unseen test data. It does not mean that the model is free from overfitting or that its error is acceptable for every business use.

No more hyperparameter tuning should be based on this test set. Doing so would leak information from the final evaluation into model development. Any later tuning should use training data and cross-validation only, followed by evaluation on a newly protected test set if another unbiased final comparison is required.

The repository already contains the Randomized Search experiment artifact at `models/random_forest_randomized_search.pkl`, its result table at `models/random_forest_randomized_search_results.csv`, and the fitted preprocessing pipeline at `models/full_pipeline.pkl`. These artifacts should be versioned and loaded together because the model expects data transformed in the same way as during training.

## What I Learned

- I learned that tuning is not only about getting the lowest training error. A model can fit the training data extremely well and still make much larger errors on new data.
- I learned that validation RMSE matters more than training RMSE when I choose hyperparameters because validation data better represents unseen examples.
- I learned that a smaller train-validation gap can mean better stability and less overfitting.
- I learned that the model with the smallest gap is not always the best final model. The regularized grid had the smallest gap but slightly worse validation and test RMSE.
- I learned to expand a search range when the winning value is at an edge. That rule led me to try more trees and fewer features after Grid Search 2.
- I learned that Randomized Search can find useful combinations that my manual grids do not contain exactly.
- I learned that I must record both accuracy and limitations. The selected model performed best on held-out data but still overfit the training data strongly.
- I learned that after I use the final test set, I should stop tuning against it so the result remains an honest final check.

## Practical Decision Rules for Future Projects

- If the best value is at the edge of a grid, expand the grid in that direction.
- If training RMSE is much lower than validation RMSE, investigate overfitting by reducing depth, increasing leaf size, adding data, or improving validation design.
- If both training and validation RMSE are high, the model may be underfitting, the features may be weak, or important relationships may be missing.
- Compare validation error and the train-validation gap together; neither metric tells the full story alone.
- If two models have similar test RMSE, prefer the simpler or more stable model when it meets the same practical accuracy needs.
- Do not repeatedly use the final test set for tuning or model selection.
- Save the final model, fitted preprocessing pipeline, metrics, hyperparameters, random seed, and decision notes together.
- Confirm that training, validation, and test metrics were calculated with the same target units and compatible preprocessing before comparing them.

## Next Steps

- Preserve and version the final Randomized Search artifact. Confirm that `models/random_forest_randomized_search.pkl` can be loaded and used for prediction in a clean session.
- Preserve and version the fitted preprocessing pipeline in `models/full_pipeline.pkl`, and document the required order for transforming data and calling the model.
- Create a small inference check that loads both artifacts and predicts from raw input data.
- Perform residual and error analysis to see whether errors are biased high or low and whether their size changes across the prediction range.
- Inspect examples with the largest prediction errors for data-quality issues, unusual districts, missing variables, or patterns the model does not capture.
- Compare feature importance, ideally including permutation importance on held-out data, to understand which inputs drive predictions.
- Try targeted feature engineering based on domain knowledge and the residual analysis.
- In a future experiment with a newly protected test set, compare Random Forest with Gradient Boosting, XGBoost, or `HistGradientBoostingRegressor`.
- Document limitations clearly, especially the large train-validation gap, the model's expected prediction error, and the conditions under which the training data may not represent future data.
