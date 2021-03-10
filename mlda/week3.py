# -*- coding: utf-8 -*-
"""
MLDA Week 3
@author: Jonathan Spencer

This week’s assignment involves running a lasso regression analysis. 
Lasso regression analysis is a shrinkage and variable selection method
for linear regression models. The goal of lasso regression is to obtain 
the subset of predictors that minimizes prediction error for a quantitative 
response variable. The lasso does this by imposing a constraint on the 
model parameters that causes regression coefficients for some variables 
to shrink toward zero. Variables with a regression coefficient equal to 
zero after the shrinkage process are excluded from the model. Variables 
with non-zero regression coefficients variables are most strongly 
associated with the response variable. Explanatory variables can be either 
quantitative, categorical or both. 

Your assignment is to run a lasso regression analysis using k-fold cross 
validation to identify a subset of predictors from a larger pool of predictor 
variables that best predicts a quantitative response variable. 

WHAT TO SUBMIT:

Following completion of the steps described above, create a blog
entry where you submit syntax used to run a lasso regression (copied and
pasted from your program) along with corresponding output and a brief 
written summary. Please note that your reviewers should NOT be required 
to download any files in order to complete the review.

If your data set has a relatively small number of observations, you do not 
need to split into training and test data sets. You can provide your rationale 
for not splitting your data set in your written summary.
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV


#Load the dataset
#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%f'%x)

data = pd.read_csv('dat/gapminder.csv', low_memory=False)
data = data.replace(r'^\s*$', np.NaN, regex=True)

# Unlike the previous assignments, we do not need to make a binary
# categorical response variable, but let's stick with income per person.

data_clean = data.dropna()

print('Information about our data:')
print(data_clean.dtypes)
print(data_clean.describe())
print()

#Split into training and testing sets
predictor_names = ['co2emissions','femaleemployrate','polityscore',
'alcconsumption','breastcancerper100th','employrate','hivrate','internetuserate',
'lifeexpectancy','oilperperson','relectricperperson','suicideper100th','urbanrate']
predvar = data_clean[predictor_names]

target = data_clean.incomeperperson

# standardize predictors to have mean=0 and sd=1
predictors=predvar.copy()
from sklearn import preprocessing
for n in predictor_names:
    predictors[str(n)]=preprocessing.scale(predictors[str(n)].astype('float64'))

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)

print('Prediction train shape:')
print(pred_train.shape)
print('Prediction test shape')
print(pred_test.shape)
print('Target train shape')
print(tar_train.shape)
print('Target test shape')
print(tar_test.shape)

# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
dict(zip(predictors.columns, model.coef_))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.mse_path_, ':')
plt.plot(m_log_alphascv, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
         

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
