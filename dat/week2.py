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
data['howfememployed'] = pandas.to_numeric(data['howfememployed'])
data['polityscore'] = data['polityscore'].astype('category')


#subset data to remove rows where any of the variables contain missing data
sub1=data.dropna(how='any', subset=['polityscore', 'howfememployed'])

# Run a Chi-Square Test of Independence.

# contingency table of observed counts
ct1=pandas.crosstab(sub1['howfememployed'], sub1['polityscore'])
print (ct1)

# column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

# chi-square
print ('chi-square value, p value, expected counts')
cs1= scipy.stats.chi2_contingency(ct1)
print (cs1)

# graph rate of female employment within each smoking polity score category
polfemfig, polfemax = plt.subplots()
seaborn.catplot(x="polityscore", y="howfememployed", data=sub1, kind="bar", ci=None)
plt.xlabel('Polity Score (-10 to 10)')
plt.ylabel('Rate of Female Employment (0 to 3)')
plt.title('Rate of Female Employment and Polity Score')
plt.tight_layout()
plt.savefig('dat/polfemfig.png')

# You will need to analyze and interpret post hoc paired comparisons in instances where your 
# original statistical test was significant, and you were examining more than two groups 
# (i.e. more than two levels of a categorical, explanatory variable). 

# Note: although it is possible to run large Chi-Square tables (e.g. 5 x 5, 4 x 6, etc.), 
# the test is really only interpretable when your response variable has only 2 levels 
# (see Graphing decisions flow chart in Bivariate Graphing chapter).
