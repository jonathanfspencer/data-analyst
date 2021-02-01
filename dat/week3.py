# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 3 Assignment
"""

import pandas
import numpy

data = pandas.read_csv('gapminder.csv', low_memory=False)
#subset data to remove rows where any of the variables contain missing data
sub1=data[(data['incomeperperson'] != ' ') & (data['co2emissions'] != ' ') & (data['femaleemployrate'] != ' ') & (data['polityscore']!= ' ')]

print()
print('Table with relevant columns')
datatable = sub1[['country', 'incomeperperson', 'co2emissions', 'femaleemployrate', 'polityscore']]
#print first 25 rows of datatable
print(datatable.head(n=25))
print()
print()
print ('Number of observations: ', len(datatable)) #number of observations (rows)
print ('Number of variables: ', len(datatable.columns)) # number of variables (columns)

print('List of country names')
print('Count for country names: country')
countrycount = datatable['country'].value_counts(sort=True)
print (countrycount)
print()

print('2010 Gross Domestic Product per capita in constant 2000 US$. The inflation but not the differences in the cost of living between countries has been taken into account.')
print('Counts for Income Per Person: incomeperperson')
incomeperpersoncount = datatable['incomeperperson'].value_counts(sort=False)
print (incomeperpersoncount)
print('Percentages for Income Per Person: incomeperperson')
incomeperpersonpercent = datatable['incomeperperson'].value_counts(sort=False, normalize=True)
print (incomeperpersonpercent)
print()

print('2006 cumulative CO2 emission (metric tons), Total amount of CO2 emission in metric tons since 1751.')
print('Counts for CO2 Emissions Per Person: co2emissions')
co2emissionscount = datatable['co2emissions'].value_counts(sort=False)
print (co2emissionscount)
print('Percentages for CO2 Emissions Per Person: co2emissions')
co2emissionspercent = datatable['co2emissions'].value_counts(sort=False, normalize=True)
print (co2emissionspercent)
print()

print('2007 female employees age 15+ (% of population) Percentage of female population, age above 15, that has been employed during the given year.')
print('Counts for Female Employment Rate: femaleemployrate')
femaleemployratecount = datatable['femaleemployrate'].value_counts(sort=False)
print (femaleemployratecount)
print('Percentages for Female Employment Rate: femaleemployrate')
femaleemployratepercent = datatable['femaleemployrate'].value_counts(sort=False, normalize=True)
print (femaleemployratepercent)
print()

print('2009 Democracy score (Polity) Overall polity score from the Polity IV dataset, calculated by subtracting an autocracy score from a democracy score. The summary measure of a country''s democratic and free nature. -10 is the lowest value, 10 the highest.')
print('Counts for Polity Score: polityscore')
polityscorecount = datatable['polityscore'].value_counts(sort=False)
print (polityscorecount)
print('Percentages for Polity Score: polityscore')
polityscorepercent = datatable['polityscore'].value_counts(sort=False, normalize=True)
print (polityscorepercent)


