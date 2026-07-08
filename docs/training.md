# Model Training

## linear regression

    * RMSE - 68627.87390018745
    * not a great score: most districts’ median_housing_values range between $120,000 and $265,000, so a typical prediction error of $68,628 is not very satisfying.
    * model is underfitting - When this happens it can mean that the features do not provide enough information to make good predictions, or that the model is not powerful enough.

    * chosen solution : complex model

## DecisionTree

    * RMSE = 0.0
    * suspicious! Could be that the model is badly overfitting

## Random Forest

    * RMSE = 18866.85405671534
    * Better than decision tree and linear regression. Need to do CV to confirm this better score.

## Support Vector Regression

    * RMSE = 118578.69234925653
    * Worse than all the models. need to confirm using cv

## K-fold validations:

When you use 10-fold cross-validation, you force the Decision Tree to train on 90% of the data and make predictions on a 10% "validation slice" it has never seen before. It does this 10 times.

    * Decision Tree Mean RMSE: 71431.40907533008
    * Linear Regression Mean RMSE: 69104.07998247063
    * SVM Mean RMSE:: 118584.55594251942

    * Result: the Decision Tree's errors are larger.

Explanation:

- Decision Trees are incredibly flexible—almost too flexible. Left unconstrained, a Decision Tree will memorize every single quirk, outlier, and bit of noise in the training data until it gets a perfect score of 0.
- But because it memorized the training data instead of learning general patterns, the moment you show it new data (during cross-validation), it panics and makes poor predictions. This classic machine learning trap is called overfitting.
- SVR is incredibly sensitive to the scale of your target variable ($y$).By default, Scikit-Learn’s SVR sets a parameter called epsilon=0.1 and C=1.0.epsilon defines a "tolerance tube" around predictions where the model doesn't penalize errors.
- If your target values are house prices in the hundreds of thousands (like $300,000$), an epsilon of 0.1 and a regularization weight of C=1.0 are mathematically microscopic.Because the numbers are so mismatched, the SVR algorithm gives up and essentially draws a flat, straight line right through the average of your data.

Solution: Complex Model - Randomforest

    * Random Forest Mean RMSE: 50345.24055547834
    * Result : Better than decision tree and linear regression

Explanation:

- Random Forests look very promising. However, note that the score on the training set is still much lower than on the validation sets, meaning that the model is still overfitting the training set. Possible solutions for overfitting are to simplify the model, constrain it (i.e., regularize it), or get a lot more training data.

Solution: Regularization

# Save the model config for future

- By saving the actual predictions along with the model, you can build an "error dashboard" later to compare model types
- Packed all the experimental data into a single Python dictionary and save that dictionary as one .pkl file inside models/.
  - Linear Regression - lin_reg_v1.pkl
  - Decision tree - decision_tree_v1.pkl
  - Random Forest - random_forest_v1.pkl
  - SVR - svr_v1.pkl

# Fine tuning the model
