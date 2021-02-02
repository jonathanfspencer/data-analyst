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

# # set entries with 0 co2emissions to NaN
# def co2emissions (row):
#     if row['co2emissions'] == 0 :
#         return numpy.NaN

# data['co2emissions'] = data.apply (lambda row: co2emissions (row),axis=1)

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
sub1=sub1[sub1['co2emissions'] > 0]

# H0: There is no correlation between whether most women in a country are employed and the amount of CO2 released
# H1: There is a correlation between whether most women in a country are employed and the amount of CO2 released
# using ols function for calculating the F-statistic and associated p value
model1 = smf.ols(formula='co2emissions ~ C(mostfememployed)', data=sub1)
results1 = model1.fit()
print (results1.summary())
print()

sub2 = sub1[['co2emissions', 'mostfememployed']].dropna()

print ('means for co2emissions by female employment status')
m1= sub2.groupby('mostfememployed').mean()
print (m1)

print ('standard deviations for co2emissions by female employment status')
sd1 = sub2.groupby('mostfememployed').std()
print (sd1)
print()

# H0: There is no correlation between whether most women in a country are employed and the polity score
# H1: There is a correlation between whether most women in a country are employed and the polity score
# using ols function for calculating the F-statistic and associated p value
model2 = smf.ols(formula='polityscore ~ C(mostfememployed)', data=sub1)
results2 = model2.fit()
print (results2.summary())
print()

sub3 = sub1[['polityscore', 'mostfememployed']].dropna()

print ('means for polityscore by female employment status')
m2= sub3.groupby('mostfememployed').mean()
print (m2)

print ('standard deviations for polityscore by female employment status')
sd2 = sub3.groupby('mostfememployed').std()
print (sd2)
print(0)
print()

# multicomparison of polityscore and mostfememployed
mc1 = multi.MultiComparison(sub1['polityscore'], sub1['mostfememployed'])
res1 = mc1.tukeyhsd()
print(res1.summary())