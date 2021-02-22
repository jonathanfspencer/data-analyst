# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 3 Assignment

This week's assignment is to test a multiple regression model. 

Data preparation for this assignment:

1) If your response variable is categorical, you will need to identify a 
quantitative variable in the data set that you can use as a response variable 
for this assignment. Variables with response scales with 4-5 values that 
represent a change in magnitude (eg, "strongly disagree to strongly agree", 
"never to often") can be considered quantitative for the assignment.

The assignment:

Write a blog entry that summarize in a few sentences 
1) what you found in your multiple regression analysis. Discuss the results for 
the associations between all of your explanatory variables and your response 
variable. Make sure to include statistical results (Beta coefficients and 
p-values) in your summary. 
2) Report whether your results supported your hypothesis for the association 
between your primary explanatory variable and the response variable. 
3) Discuss whether there was evidence of confounding for the association 
between your primary explanatory and response variable (Hint: adding additional 
explanatory variables to your model one at a time will make it easier to 
identify which of the variables are confounding variables); and 
4) generate the following regression diagnostic plots:

a) q-q plot
b) standardized residuals for all observations
c) leverage plot

d) Write a few sentences describing what these plots tell you about your regression 
model in terms of the distribution of the residuals, model fit, influential observations, 
and outliers. 

What to Submit: Submit the URL for your blog entry. The blog entry should include 
1) the summary of your results that addresses parts 1-4 of the assignment, 
2) the output from your multiple regression model, and 
3) the regression diagnostic plots.
"""

import numpy
import pandas
import statsmodels.api
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

# For Week 3, let's look at the relationship between urban rate (explanatory) and female employment(response).
# Let's also check whether income per person, employment rate, and polity score are confounding variables.

# H0: There is no significant relationship between urban rate and female employment
# H1: There is a significant relationship between urban rate and female employment
# explanatory: urbanrate
# response: femaleemployrate
# confounders:  incomeperperson, employrate, polityscore

data['urbanrate'] = pandas.to_numeric(data['urbanrate'])
data['femaleemployrate'] = pandas.to_numeric(data['femaleemployrate'])
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'])
data['employrate'] = pandas.to_numeric(data['employrate'])
data['polityscore'] = pandas.to_numeric(data['polityscore'])

#subset data to remove rows where any of the variables contain missing data
data=data.dropna()

