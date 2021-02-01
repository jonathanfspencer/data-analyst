# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 1 Assignment
"""

import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

#Set PANDAS to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

data = pandas.read_csv('dat/gapminder.csv', low_memory=False)
data = data.replace(r'^\s*$', numpy.NaN, regex=True)

data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'])
data['co2emissions'] = pandas.to_numeric(data['co2emissions'])
data['femaleemployrate'] = pandas.to_numeric(data['femaleemployrate'])
data['polityscore'] = pandas.to_numeric(data['polityscore'])

# create a categorical variable for whether most working age women are employed
def mostfememployed (row):
   if row['femaleemployrate'] > 50 :
      return 1
   else :
      return 0
         
data['mostfememployed'] = data.apply (lambda row: mostfememployed (row),axis=1)
# set new variable type to categorical
data['mostfememployed'] = data['mostfememployed'].astype('category')

#subset data to remove rows where any of the variables contain missing data
sub1=data.dropna(how='any', subset=['incomeperperson', 'co2emissions', 'femaleemployrate', 'polityscore', 'mostfememployed'])

# H0: There is no correlation between whether most women in a country are employed and the amount of CO2 released
# H1: There is a correlation between whether most women in a country are employed and the amount of CO2 released
# using ols function for calculating the F-statistic and associated p value
model1 = smf.ols(formula='co2emissions ~ C(mostfememployed)', data=sub1)
results1 = model1.fit()
print (results1.summary())
