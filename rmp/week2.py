# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 2 Assignment

This week's assignment asks you to test a basic linear regression model for 
the association between your primary explanatory variable and a response variable, 
and to create a blog entry describing your results. 

Data preparation for this assignment: 

1) If your explanatory variable is categorical with more than two categories, 
you will need to collapse it down to two categories, or subset your data to select 
observations from 2 categories (next week you'll learn how to analyze categorical 
explanatory variable with more than 2 categories). 

2) If your response variable is categorical, you will need to identify a quantitative 
variable in the data set that you can use as a response variable for this assignment. 
Variables with response scales with 4-5 values that represent a change in magnitude 
(for example, "strongly disagree to strongly agree", "never to often") can be 
considered quantitative for the assignment.

The assignment:

1) If you have a categorical explanatory variable, make sure one of your 
categories is coded "0" and generate a frequency table for this variable to 
check your coding. If you have a quantitative explanatory variable, center it 
so that the mean = 0 (or really close to 0) by subtracting the mean, and then 
calculate the mean to check your centering. 

2) Test a linear regression model and summarize the results in a couple of 
sentences. Make sure to include statistical results (regression coefficients 
and p-values) in your summary.
"""

import numpy
import pandas
import statsmodels.api
import statsmodels.formula.api as smf
import seaborn
import matplotlib.pyplot as plt

#Set PANDAS to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

data = pandas.read_csv('dat/gapminder.csv', low_memory=False)
data = data.replace(r'^\s*$', numpy.NaN, regex=True)

data['urbanrate'] = pandas.to_numeric(data['urbanrate'])
# data['oilperperson'] = pandas.to_numeric(data['oilperperson'])
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'])

# H0: There is no significant relationship between urban rate and income per person
# H1: There is a significant relationship between urban rate and income per person
# explanatory: urbanrate
# response: incomeperperson

#subset data to remove rows where any of the variables contain missing data
data=data.dropna()

# Data Management
# Use urban rate to create a two-category variable
def urbhilow (row):
    if row['urbanrate'] > 50:
        return 1
    else:
        return 0

data['urbhilow'] = data.apply(lambda row: urbhilow(row), axis=1)
data['urbhilow'] = data['urbhilow'].astype('category')
# Generate a frequency table for urbhilow
urbhilowcount = data['urbhilow'].value_counts(sort=False)
print('Urban Low and High Rate counts:')
print(urbhilowcount)
urbhilowpercent = data['urbhilow'].value_counts(sort=False, normalize=True)
print('Urban Low and High Rate percentages:')
print(urbhilowpercent)
print()

# Make a scatter plot to visualize the relationship
fig1, bar1 = plt.subplots()
bar1 = seaborn.barplot(x="urbhilow", y="incomeperperson", data=data, ax=bar1)
bar1.set_xlabel('2008 Urban Population Rate (0=Low, 1=High)')
bar1.set_ylabel('2000 Income Per Capita in $USD')
bar1.set_title('Urbanization Rate and Income Per Person')
fig1.savefig('rmp/urbanincome.png')

# Perform an OLS Regression
print ('Association between urban population rate and income per person')
urbincomereg = smf.ols('incomeperperson ~ C(urbhilow)', data=data).fit()
print(urbincomereg.summary())