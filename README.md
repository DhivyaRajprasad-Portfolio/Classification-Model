# Classification-Model

German Credit dataset contains information about the default/non-default predictors across several industries.

We fit supervised and classification models to the dataset consisting of 1000 observations and 21 variables.

The dataset contains 20 predictor variables i.e. Status of checking account, credit history, purpose, savings accounts/bonds, present employment since, personal status and sex, other debtors/guarantors, property, other installment plans, housing, job, telephone, foreign worker, duration in month, credit amount, installment rate in terms of percentage of disposable income, present resident since, age in years, number of existing credits, number of people liable to provide maintenance and 1 response variable: default or no default status.

The dataset is randomly sampled into 75% training and 25% testing data. A comparison based on different models based on different flexibilities, misclassification rates, area under the curve and mean residual deviance are made. We choose an asymmetric cost of 5:1 which is False Negative: False Positive.