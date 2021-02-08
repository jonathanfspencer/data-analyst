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

# H0: There is no significant relationship between urban rate and oil use per person
# H1: # H0: There is a significant relationship between urban rate and oil use per person
# explanatory: urbanrate
# response: oilperperson
# moderator: incomeperperson

#subset data to remove rows where any of the variables contain missing data
data=data.dropna()

# Make a scatter plot to visualize the relationship
fig1, scat1 = plt.subplots()
scat1 = seaborn.regplot(x="urbanrate", y="oilperperson", fit_reg=True, data=data, ax=scat1)
scat1.set_xlabel('2008 Urban Population Rate')
scat1.set_ylabel('2010 Oil Consumption Per Capita')
scat1.set_title('Urbanization Rate and Oil Consumption')
fig1.savefig('dat/urbanoil.png')

# Perform a Pearson Correlation Coefficient Test
print ('Association between urban population rate and oil consumption')
urbanoilp = scipy.stats.pearsonr(data['urbanrate'], data['oilperperson'])
print(urbanoilp)
if urbanoilp[1] < 0.05:
    print('This relationship IS statistically significant')
else:
    print('This relationship is NOT statistically significant')
print ()

# See how incomeperperson works as a moderator
# subset data to those countries at or below USD$2000 
def incomegrp (row):
   if row['incomeperperson'] < 2000:
      return 1
   else:
      return 2
   
data['incomegrp'] = data.apply (lambda row: incomegrp (row),axis=1)
print('Row counts above and below USD$2000:')
chk1 = data['incomegrp'].value_counts(sort=False, dropna=False)
print(chk1)
# create a subframe for those countries below USD$2000
sub2=data[(data['incomegrp'] == 1)]
# create a subframe for those countries above USD$2000
sub3=data[(data['incomegrp'] == 2)]

# Repeat for those countries below USD$2000
# Make a scatter plot to visualize the relationship
fig2, scat2 = plt.subplots()
scat2 = seaborn.regplot(x="urbanrate", y="oilperperson", fit_reg=True, data=sub2, ax=scat2)
scat2.set_xlabel('2008 Urban Population Rate')
scat2.set_ylabel('2010 Oil Consumption Per Capita')
scat2.set_title('Urbanization Rate and Oil Consumption Below USD$2000')
fig2.savefig('dat/urbanoilbelow.png')

# Perform a Pearson Correlation Coefficient Test
print ('Association between urban population rate and oil consumption below USD$2000')
urbanoilbelowp = scipy.stats.pearsonr(sub2['urbanrate'], sub2['oilperperson'])
print(urbanoilbelowp)
if urbanoilbelowp[1] < 0.05:
    print('This relationship IS statistically significant')
else:
    print('This relationship is NOT statistically significant')
print ()

# Repeat for those countries above USD$2000
# Make a scatter plot to visualize the relationship
fig3, scat3 = plt.subplots()
scat3 = seaborn.regplot(x="urbanrate", y="oilperperson", fit_reg=True, data=sub3, ax=scat3)
scat3.set_xlabel('2008 Urban Population Rate')
scat3.set_ylabel('2010 Oil Consumption Per Capita')
scat3.set_title('Urbanization Rate and Oil Consumption Above USD$2000')
fig3.savefig('dat/urbanoilabove.png')

# Perform a Pearson Correlation Coefficient Test
print ('Association between urban population rate and oil consumption above USD$2000')
urbanoilabovep = scipy.stats.pearsonr(sub3['urbanrate'], sub3['oilperperson'])
print(urbanoilabovep)
if urbanoilabovep[1] < 0.05:
    print('This relationship IS statistically significant')
else:
    print('This relationship is NOT statistically significant')
print ()