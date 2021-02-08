# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 4 Assignment
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

data['urbanrate'] = pandas.to_numeric(data['urbanrate'])
data['oilperperson'] = pandas.to_numeric(data['oilperperson'])
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'])

# H0: There is no significant relantionship between urban rate and oil use per person
# H1: # H0: There is a significant relantionship between urban rate and oil use per person
# explanatory: urbanrate
# response: oilperperson
# moderator: incomeperperson

#subset data to remove rows where any of the variables contain missing data
clean_data=data.dropna(how='any', subset=['urbanrate', 'oilperperson', 'oilperpoerson'])

# Make a scatter plot to visualize the relationship
fig1, scat1 = plt.subplots()
scat1 = seaborn.regplot(x="urbanrate", y="oilperperson", fit_reg=True, data=clean_data, ax=scat1)
scat1.set_xlabel('2008 Urban Population Rate')
scat1.set_ylabel('2010 Oil Consumption Per Capita')
scat1.set_title('Urbanization Rate and Oil Consumption')
fig1.savefig('dat/urbanoil.png')

# Perform a Pearson Correlation Coefficient Test
print ('Association between urban population rate and oil consumption')
print (scipy.stats.pearsonr(clean_data['urbanrate'], clean_data['oilperperson']))
print()

# See how incomeperperson works as a moderator
# Using International Poverty Line of $1.90/day based on http://documents1.worldbank.org/curated/en/837051468184454513/pdf/Estimating-international-poverty-lines-from-comparable-national-thresholds.pdf
povertyline = 1.9 * 365

# subset data to those countries at or below International Poverty Line
sub2=clean_data(clean_data['incomeperperson'] <= povertyline)

# Make a scatter plot to visualize the relationship
fig2, scat2 = plt.subplots()
scat2 = seaborn.regplot(x="urbanrate", y="oilperperson", fit_reg=True, data=sub2, ax=scat2)
scat2.set_xlabel('2008 Urban Population Rate')
scat2.set_ylabel('2010 Oil Consumption Per Capita')
scat2.set_title('Urbanization Rate and Oil Consumption Below IPL')
fig2.savefig('dat/urbanoilbelow.png')

# Perform a Pearson Correlation Coefficient Test
print ('Association between urban population rate and oil consumption below IPL')
print (scipy.stats.pearsonr(sub2['urbanrate'], sub2['oilperperson']))
print()

# subset data to those countries above International Poverty Line
sub3=clean_data(clean_data['incomeperperson'] <= povertyline)

# Make a scatter plot to visualize the relationship
fig3, scat3 = plt.subplots()
scat3 = seaborn.regplot(x="urbanrate", y="oilperperson", fit_reg=True, data=sub3, ax=scat3)
scat3.set_xlabel('2008 Urban Population Rate')
scat3.set_ylabel('2010 Oil Consumption Per Capita')
scat3.set_title('Urbanization Rate and Oil Consumption Above IPL')
fig3.savefig('dat/urbanoilabove.png')

# Perform a Pearson Correlation Coefficient Test
print ('Association between urban population rate and oil consumption above IPL')
print (scipy.stats.pearsonr(sub3['urbanrate'], sub3['oilperperson']))
print()