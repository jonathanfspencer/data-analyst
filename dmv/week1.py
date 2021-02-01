# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 1 Assignment
"""

import pandas
import numpy

data = pandas.read_csv('gapminder.csv', low_memory=False)

print ('Number of observations: ', len(data)) #number of observations (rows)
print ('Number of variables: ', len(data.columns)) # number of variables (columns)

print()
print()
print('2010 Gross Domestic Product per capita in constant 2000 US$. The inflation but not the differences in the cost of living between countries has been taken into account.')
print('Counts for Income Per Person: incomeperperson')
incomeperpersoncount = data['incomeperperson'].value_counts(sort=False)
print (incomeperpersoncount)
print('Percentages for Income Per Person: incomeperperson')
incomeperpersonpercent = data['incomeperperson'].value_counts(sort=False, normalize=True)
print (incomeperpersonpercent)
print()

print('2006 cumulative CO2 emission (metric tons), Total amount of CO2 emission in metric tons since 1751.')
print('Counts for CO2 Emissions Per Person: co2emissions')
co2emissionscount = data['co2emissions'].value_counts(sort=False)
print (co2emissionscount)
print('Percentages for CO2 Emissions Per Person: co2emissions')
co2emissionspercent = data['co2emissions'].value_counts(sort=False, normalize=True)
print (co2emissionspercent)
print()

print('2007 female employees age 15+ (% of population) Percentage of female population, age above 15, that has been employed during the given year.')
print('Counts for Female Employment Rate: femaleemployrate')
femaleemployratecount = data['femaleemployrate'].value_counts(sort=False)
print (femaleemployratecount)
print('Percentages for Female Employment Rate: femaleemployrate')
femaleemployratepercent = data['femaleemployrate'].value_counts(sort=False, normalize=True)
print (femaleemployratepercent)
print()

print('2009 Democracy score (Polity) Overall polity score from the Polity IV dataset, calculated by subtracting an autocracy score from a democracy score. The summary measure of a country''s democratic and free nature. -10 is the lowest value, 10 the highest.')
print('Counts for Polity Score: polityscore')
polityscorecount = data['polityscore'].value_counts(sort=False)
print (polityscorecount)
print('Percentages for Polity Score: polityscore')
polityscorepercent = data['polityscore'].value_counts(sort=False, normalize=True)
print (polityscorepercent)