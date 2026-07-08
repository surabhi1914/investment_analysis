# Data cleaning action taken:

- seperated the dependant and independant label data
- simple imputer transformer with strategy applied on the numerical data of the stratified train data.
  - median because our data is skewed
- Tried labelencoder: Issue - This representation might cause ML algorithms will assume that two nearby values are more similar than two distant values. Hence, onehotencoding
- Custom Transformer helped in adding rooms_per_household, population_per_household
- Added pipeline for imputation, custom transformer, standardization
- Added column transformer for adding numerical pipeline and onehotencoder
