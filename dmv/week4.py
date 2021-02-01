# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 4 Assignment
"""

import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

#Set PANDAS to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

data = pandas.read_csv('gapminder.csv', low_memory=False)
data = data.replace(r'^\s*$', numpy.NaN, regex=True)

data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'])
data['co2emissions'] = pandas.to_numeric(data['co2emissions'])
data['femaleemployrate'] = pandas.to_numeric(data['femaleemployrate'])
data['polityscore'] = pandas.to_numeric(data['polityscore'])
#subset data to remove rows where any of the variables contain missing data
sub1=data.dropna(how='any', subset=['incomeperperson', 'co2emissions', 'femaleemployrate', 'polityscore'])

print()
print('Table with relevant columns (first 25 rows)')
datatable = sub1[['country', 'incomeperperson', 'co2emissions', 'femaleemployrate', 'polityscore']]

#print first 25 rows of datatable
print(datatable.head(n=25))
print()
print()
print ('Number of observations: ', len(datatable)) #number of observations (rows)
print ('Number of variables: ', len(datatable.columns)) # number of variables (columns)
print()

print('2010 Gross Domestic Product per capita in constant 2000 US$. The inflation but not the differences in the cost of living between countries has been taken into account.')
print(datatable['incomeperperson'].describe())
# univariate histogram for quantitative variable
incomefig, incomeax = plt.subplots()
seaborn.histplot(datatable['incomeperperson'], x=datatable['incomeperperson'], ax=incomeax)
incomeax.set_title('2010 Gross Domestic Product per capita in constant 2000 US$')
incomeax.set_xlabel('Income per capita')
incomeax.set_ylabel('Number of countries')
incomefig.tight_layout()
incomefig.savefig('incomefig.png')
print()

print('2006 cumulative CO2 emission (metric tons), Total amount of CO2 emission in metric tons since 1751.')
print(datatable['co2emissions'].describe())
# univariate histogram for quantitative variable
co2fig, co2ax = plt.subplots()
seaborn.histplot(datatable['co2emissions'], ax=co2ax)
co2ax.set_xlabel('2006 cumulative CO2 emission (metric tons)')
co2ax.set_ylabel('Number of countries')
co2ax.set_title('Total amount of CO2 emission in metric tons since 1751')
co2fig.tight_layout()
co2fig.savefig('co2fig.png')
print()

print('2007 female employees age 15+ (% of population) Percentage of female population, age above 15, that has been employed during the given year.')
print(datatable['femaleemployrate'].describe())
# univariate histogram for quantitative variable
femfig, femax = plt.subplots()
seaborn.histplot(datatable['femaleemployrate'], ax=femax)
femax.set_xlabel('2007 female employees age 15+ (% of population)')
femax.set_ylabel('Number of countries')
femax.set_title('Percentage of female population, age above 15, that has been employed during the given year')
femfig.tight_layout()
femfig.savefig('femfig.png')
print()

print('2009 Democracy score (Polity) Overall polity score from the Polity IV dataset, calculated by subtracting an autocracy score from a democracy score. The summary measure of a country''s democratic and free nature. -10 is the lowest value, 10 the highest.')
print(datatable['polityscore'].describe())
# univariate histogram for quantitative variable
polfig, polax = plt.subplots()
seaborn.histplot(datatable['polityscore'], ax=polax)
polax.set_xlabel('2009 Democracy score (Polity)')
polax.set_ylabel('Number of countries')
polax.set_title('Overall polity score from the Polity IV dataset')
polfig.tight_layout()
polfig.savefig('polfig.png')

# scatter plot of femaleemployrate (explanatory, quantitative) to co2emissions (response, quantitative)
femco2fig, femco2ax = plt.subplots()
seaborn.scatterplot(data=datatable, x='femaleemployrate', y='co2emissions', hue='polityscore', ax=femco2ax)
femco2ax.set_xlabel("2007 female employees age 15+ (% of population)")
femco2ax.set_ylabel('2006 cumulative CO2 emission (metric tons)')
femco2ax.set_title('Relationship between rate of female employment and CO2 emissions')
femco2fig.tight_layout()
femco2fig.savefig('femco2fig.png')

#scatter plot of femaleemployrate (explanatory, quantitative) to polityscore (response, quantitative)
fempolfig, fempolax = plt.subplots()
seaborn.scatterplot(data=datatable, x='femaleemployrate', y='polityscore', hue='incomeperperson', ax=fempolax)
fempolax.set_xlabel("2007 female employees age 15+ (% of population)")
fempolax.set_ylabel('2009 Democracy score (Polity)')
fempolax.set_title('Relationship between rate of female employment and polity score')
fempolfig.tight_layout()
fempolfig.savefig('fempolfig.png')

# scatter plot of polity score (explanatory, quantitative) to co2emissions (response, quantitative)
polco2fig, polco2ax = plt.subplots()
seaborn.scatterplot(data=datatable, x='polityscore', y='co2emissions', hue='femaleemployrate', ax=polco2ax)
polco2ax.set_xlabel("2009 Democracy score (Polity)")
polco2ax.set_ylabel('2006 cumulative CO2 emission (metric tons)')
polco2ax.set_title('Relationship between polity score and CO2 emissions')
polco2fig.tight_layout()
polco2fig.savefig('polco2fig.png')
