# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 4 Assignment

This week's assignment is to test a logistic regression model. 

Data preparation for this assignment:

1) If your response variable is categorical with more than two categories, 
you will need to collapse it down to two categories, or subset your data to 
select observations from 2 categories.

2) If your response variable is quantitative, you will need to bin it 
into two categories.

The assignment:

Write a blog entry that summarize in a few sentences 

1) what you found, making sure you discuss the results for the 
associations between all of your explanatory variables and your 
response variable. Make sure to include statistical results (odds 
ratios, p-values, and 95% confidence intervals for the odds ratios) 
in your summary. 

2) Report whether or not your results supported your hypothesis for 
the association between your primary explanatory variable and your 
response variable. 

3) Discuss whether or not there was evidence of confounding for the 
association between your primary explanatory and the response variable 
(Hint: adding additional explanatory variables to your model one at a 
time will make it easier to identify which of the variables are 
confounding variables).  

What to Submit: Write a blog entry and submit the URL for your blog. 
Your blog entry should include 

1) the summary of your results that addresses parts 1-3 of the assignment, 

2) the output from your logistic regression model.

  Example of how to write logistic regression results:

After adjusting for potential confounding factors (list them), the odds 
of having nicotine dependence were more than two times higher for participants 
with major depression than for participants without major depression (OR=2.36, 
95% CI = 1.44-3.81, p=.0001). Age was also significantly associated with 
nicotine dependence, such that older older participants were significantly 
less likely to have nicotine dependence (OR= 0.81, 95% CI=0.40-0.93, p=.041).  
"""

import numpy
import pandas
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn
import matplotlib.pyplot as plt
import scipy

#Set PANDAS to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

data = pandas.read_csv('dat/gapminder.csv', low_memory=False)
data = data.replace(r'^\s*$', numpy.NaN, regex=True)

# For Week 4, let's look at the relationship between urban rate (explanatory) and polity score (response).
# Let's also check whether income per person and employment rate are confounding variables.

# H0: There is no significant relationship between urban rate and polity score
# H1: There is a significant relationship between urban rate and polity score
# explanatory: urbanrate
# response: polity score
# confounders:  incomeperperson and employrate

data['urbanrate'] = pandas.to_numeric(data['urbanrate'])
data['polityscore'] = pandas.to_numeric(data['polityscore'])
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'])
data['employrate'] = pandas.to_numeric(data['employrate'])

#subset data to remove rows where any of the variables contain missing data
data=data[['urbanrate','incomeperperson','employrate','polityscore']].dropna()

# make a binary categorical for polity score, the response variable
def POSIPOLI(row):
    if row['polityscore'] > 0:
        return 1
    else:
        return 0

data['posipoli'] = data.apply(lambda row: POSIPOLI(row), axis=1)

# check the new positive polity score variable
print('Check positive polity score counts:')
posipolicheck = data['posipoli'].value_counts(sort=False, dropna=False)
print (posipolicheck)
print()

# center quantitative IVs for regression analysis
data['incomeperperson_c'] = (data['incomeperperson'] - data['incomeperperson'].mean())
data['employrate_c'] = (data['employrate'] - data['employrate'].mean())
