# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 3 Assignment
"""

import pandas
import numpy
import seaborn
import scipy
import matplotlib.pyplot as plt

#Set PANDAS to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

data = pandas.read_csv('dat/gapminder.csv', low_memory=False)
data = data.replace(r'^\s*$', numpy.NaN, regex=True)

data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'])
data['femaleemployrate'] = pandas.to_numeric(data['femaleemployrate'])

# H0: There is no correlation between whether most women in a country are employed and the income per capita
# H1: There is a correlation between whether most women in a country are employed and the income per capita

#subset data to remove rows where any of the variables contain missing data
sub1=data.dropna(how='any', subset=['incomeperperson', 'femaleemployrate'])
sub1=sub1[sub1['incomeperperson'] > 0]

# Make a scatter plot to visualize the relationship
fig1, scat1 = plt.subplots()
scat1 = seaborn.regplot(x="femaleemployrate", y="incomeperperson", fit_reg=True, data=sub1, ax=scat1)
scat1.set_xlabel('Female Employment Rate')
scat1.set_ylabel('Income Per Capita')
scat1.set_title('Female Employment Rate and Income Per Capita')
fig1.savefig('dat/femgdp.png')

# Perform a Pearson Correlation Coefficient Test
print ('Association between female employment rate and income per capita')
print (scipy.stats.pearsonr(sub1['femaleemployrate'], sub1['incomeperperson']))

# I realized from the visualization that many countries have a severely depressed GDP relative to others
# H0: There is no correlation between whether most women in a country are employed and the income per capita in countries where the income per capita is greater than $1000USD
# H1: There is a correlation between whether most women in a country are employed and the income per capita in countries where the income per capita is greater than $1000USD

#subset data to remove rows where any of the variables contain missing data
sub2=data.dropna(how='any', subset=['incomeperperson', 'femaleemployrate'])
sub2=sub1[sub1['incomeperperson'] > 1000]

# Make a scatter plot to visualize the relationship
fig2, scat2 = plt.subplots()
scat2 = seaborn.regplot(x="femaleemployrate", y="incomeperperson", fit_reg=True, data=sub2, ax=scat2)
scat2.set_xlabel('Female Employment Rate')
scat2.set_ylabel('Income Per Capita')
scat2.set_title('Female Employment Rate and Income Per Capita')
fig2.savefig('dat/femgdp1000.png')

# Perform a Pearson Correlation Coefficient Test
print ('Association between female employment rate and income per capita in countries where the income per capita is greater than $1000USD')
print (scipy.stats.pearsonr(sub2['femaleemployrate'], sub2['incomeperperson']))

# There was a stronger correlation at $1000 min gdp, so let's try at $5000
# H0: There is no correlation between whether most women in a country are employed and the income per capita in countries where the income per capita is greater than $5000USD
# H1: There is a correlation between whether most women in a country are employed and the income per capita in countries where the income per capita is greater than $5000USD

#subset data to remove rows where any of the variables contain missing data
sub3=data.dropna(how='any', subset=['incomeperperson', 'femaleemployrate'])
sub3=sub1[sub1['incomeperperson'] > 5000]

# Make a scatter plot to visualize the relationship
fig3, scat3 = plt.subplots()
scat3 = seaborn.regplot(x="femaleemployrate", y="incomeperperson", fit_reg=True, data=sub3, ax=scat3)
scat3.set_xlabel('Female Employment Rate')
scat3.set_ylabel('Income Per Capita')
scat3.set_title('Female Employment Rate and Income Per Capita')
fig3.savefig('dat/femgdp5000.png')

# Perform a Pearson Correlation Coefficient Test
print ('Association between female employment rate and income per capita in countries where the income per capita is greater than $5000USD')
print (scipy.stats.pearsonr(sub3['femaleemployrate'], sub3['incomeperperson']))