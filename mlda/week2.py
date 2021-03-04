# -*- coding: utf-8 -*-
"""
MLDA Week 2
@author: Jonathan Spencer

The second assignment deals with Random Forests. Random forests are predictive 
models that allow for a data driven exploration of many explanatory variables 
in predicting a response or target variable. Random forests provide importance 
scores for each explanatory variable and also allow you to evaluate any increases 
in correct classification with the growing of smaller and larger number of trees.

Run a Random Forest. 

You will need to perform a random forest analysis to evaluate the importance of 
a series of explanatory variables in predicting a binary, categorical response variable.

WHAT TO SUBMIT:

Following completion of the steps described above, create a blog entry where you 
submit syntax used to run a Random Forest (copied and pasted from your program) 
along with corresponding output and a few sentences of interpretation. Please note 
that your reviewers should NOT be required to download any files in order to complete 
the review.
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.metrics
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus

#Load the dataset
#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%f'%x)

data = pd.read_csv('dat/gapminder.csv', low_memory=False)
data = data.replace(r'^\s*$', np.NaN, regex=True)

# For this assignment, we need a binary categorical response variable.  
# The Gapminder dataset does not have one, so let's make one out of
# income per person.  Let's also figure out what predictors we'll need
# and get rid of the observations for which we do not have data across
# the board.
data['incomeperperson'] = pd.to_numeric(data['incomeperperson'])
data_clean = data.dropna()

def HIGHINCOME(row):
    if row['incomeperperson'] > 25000:
        return 1
    else:
        return 0

data_clean['highincome'] = data_clean.apply(lambda row: HIGHINCOME(row), axis=1)

print('Information about our data:')
print(data_clean.dtypes)
print(data_clean.describe())
print()

#Split into training and testing sets
predictor_names = ['co2emissions','femaleemployrate','polityscore',
'alcconsumption','breastcancerper100th','employrate','hivrate','internetuserate',
'lifeexpectancy','oilperperson','relectricperperson','suicideper100th','urbanrate']
predictors = data_clean[predictor_names]

targets = data_clean.highincome

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)

print('Prediction train shape:')
print(pred_train.shape)
print('Prediction test shape')
print(pred_test.shape)
print('Target train shape')
print(tar_train.shape)
print('Target test shape')
print(tar_test.shape)

#Build model on training data
classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

print('Confusion Matrix:')
print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print()
print('Accuracy Score:')
print(sklearn.metrics.accuracy_score(tar_test, predictions))
print()

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
print('Feature Importances')
for p, i in zip(predictors.columns, model.feature_importances_):
    print(p + ': ' + str(i))