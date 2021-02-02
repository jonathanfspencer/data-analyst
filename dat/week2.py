# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 2 Assignment
"""

import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 
import scipy.stats
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

# data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'])
# data['co2emissions'] = pandas.to_numeric(data['co2emissions'])
data['femaleemployrate'] = pandas.to_numeric(data['femaleemployrate'])
data['polityscore'] = pandas.to_numeric(data['polityscore'])

# For week 2, we need two categorical variables
# I got weird results last week, so I want to further test the relationship between polity score and female employment
# H0: There is no correlation between the percentage of women in a country who are employed and the polity score
# H1: There is a correlation between the percentage of women in a country who are employed and the polity score
# explanatory variable: polityscore

# create a categorical variable for whether most working age women are employed
def howfememployed (row):
   if row['femaleemployrate'] > 75 :
      return 3
   elif row['femaleemployrate'] > 50 :
      return 2
   elif row['femaleemployrate'] > 25 :
      return 1
   else :
      return 0
         
data['howfememployed'] = data.apply (lambda row: howfememployed (row),axis=1)
# set new variable type to categorical
data['howfememployed'] = data['howfememployed'].astype('category')


#subset data to remove rows where any of the variables contain missing data
sub1=data.dropna(how='any', subset=['femaleemployrate', 'polityscore', 'howfememployed'])
