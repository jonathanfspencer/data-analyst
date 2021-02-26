# -*- coding: utf-8 -*-
"""
Jonathan Spencer
Week 4 Assignment

This week's assignment is to test a logistic regression model. 

Data preparation for this assignment:

1) If your response variable is categorical with more than two categories, 
you will need to collapse it down to two categories, or subset your data to 
select observations from 2 categories.

2) If your response variable is quantitative, you will need to bin it 
into two categories.

The assignment:

Write a blog entry that summarize in a few sentences 

1) what you found, making sure you discuss the results for the 
associations between all of your explanatory variables and your 
response variable. Make sure to include statistical results (odds 
ratios, p-values, and 95% confidence intervals for the odds ratios) 
in your summary. 

2) Report whether or not your results supported your hypothesis for 
the association between your primary explanatory variable and your 
response variable. 

3) Discuss whether or not there was evidence of confounding for the 
association between your primary explanatory and the response variable 
(Hint: adding additional explanatory variables to your model one at a 
time will make it easier to identify which of the variables are 
confounding variables).  

What to Submit: Write a blog entry and submit the URL for your blog. 
Your blog entry should include 

1) the summary of your results that addresses parts 1-3 of the assignment, 

2) the output from your logistic regression model.

  Example of how to write logistic regression results:

After adjusting for potential confounding factors (list them), the odds 
of having nicotine dependence were more than two times higher for participants 
with major depression than for participants without major depression (OR=2.36, 
95% CI = 1.44-3.81, p=.0001). Age was also significantly associated with 
nicotine dependence, such that older older participants were significantly 
less likely to have nicotine dependence (OR= 0.81, 95% CI=0.40-0.93, p=.041).  
"""

import numpy
import pandas
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn
import matplotlib.pyplot as plt
import scipy

#Set PANDAS to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

data = pandas.read_csv('dat/gapminder.csv', low_memory=False)
data = data.replace(r'^\s*$', numpy.NaN, regex=True)

# For Week 3, let's look at the relationship between urban rate (explanatory) and female employment(response).
# Let's also check whether income per person, employment rate, and polity score are confounding variables.

# H0: There is no significant relationship between urban rate and female employment
# H1: There is a significant relationship between urban rate and female employment
# explanatory: urbanrate
# response: femaleemployrate
# confounders:  incomeperperson, employrate, polityscore

data['urbanrate'] = pandas.to_numeric(data['urbanrate'])
data['femaleemployrate'] = pandas.to_numeric(data['femaleemployrate'])
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'])
data['employrate'] = pandas.to_numeric(data['employrate'])
data['polityscore'] = pandas.to_numeric(data['polityscore'])

#subset data to remove rows where any of the variables contain missing data
data=data[['urbanrate','femaleemployrate','incomeperperson','employrate','polityscore']].dropna()

# scatterplot for urban rate and female employment
urbfemfiglinear, urbfemaxlinear = plt.subplots()
urbfemaxlinear = seaborn.regplot(x="urbanrate", y="femaleemployrate", scatter=True, data=data, ax=urbfemaxlinear)
urbfemaxlinear.set_xlabel('Urbanization Rate')
urbfemaxlinear.set_ylabel('Female Employment Rate')
urbfemaxlinear.set_title('Urbanization Rate and Female Employment Rate (First Order)')
urbfemfiglinear.savefig('rmp/urbfemfiglinear.png')

# second order polynomial fit
urbfemfigsecond, urbfemaxsecond = plt.subplots()
urbfemaxsecond = seaborn.regplot(x="urbanrate", y="femaleemployrate", scatter=True, data=data, order=2, ax=urbfemaxsecond)
urbfemaxsecond.set_xlabel('Urbanization Rate')
urbfemaxsecond.set_ylabel('Female Employment Rate')
urbfemaxsecond.set_title('Urbanization Rate and Female Employment Rate (Second Order)')
urbfemfigsecond.savefig('rmp/urbfemfigsecond.png')

# center the variables
data['urbanrate_c'] = (data['urbanrate'] - data['urbanrate'].mean())
data['femaleemployrate_c'] = (data['femaleemployrate'] - data['femaleemployrate'].mean())
data['incomeperperson_c'] = (data['incomeperperson'] - data['incomeperperson'].mean())
data['employrate_c'] = (data['employrate'] - data['employrate'].mean())
data['polityscore_c'] = (data['polityscore'] - data['polityscore'].mean())

# do a linear regression analysis
print('Linear Regression for female employment rate and urbanization rate:')
reg1 = smf.ols('femaleemployrate ~ urbanrate_c', data=data).fit()
print(reg1.summary())
print()


# do a multiple regression with femaleemployrate and centered incomeperperson
print('Multiple regression with urbanrate and incomeperperson:')
reg2 = smf.ols('femaleemployrate ~ urbanrate_c + incomeperperson_c', data=data).fit()
print(reg2.summary())
print()

# do a polynomial regression analysis
print('Polynomial Regression for female employment rate and urbanization rate:')
reg3 = smf.ols('femaleemployrate ~ urbanrate_c + I(urbanrate_c**2)', data=data).fit()
print(reg3.summary())
print()

# do a polynomial regression analysis adding income per person
print('Polynomial Regression for female employment rate and urbanization rate with income per person:')
reg4 = smf.ols('femaleemployrate ~ urbanrate_c + I(urbanrate_c**2) + incomeperperson_c', data=data).fit()
print(reg4.summary())
print()

# do a Q-Q plot
qqfig, qqax = plt.subplots()
qqfig = sm.qqplot(reg4.resid, line='r')
qqfig.savefig('rmp/qqfig.png')

# plot the residuals
residuals = pandas.DataFrame(reg4.resid_pearson)
residualsfig, residualsax = plt.subplots()
plt.plot(residuals, 'o', ls='None')
residualline = plt.axhline(y=0, color='r')
residualsax.set_ylabel('Standardized Residual')
residualsax.set_xlabel('Observation Number')
residualsfig.savefig('rmp/residuals.png')

# do a leverage plot
levfig = sm.graphics.influence_plot(reg4, size=8)
levfig.savefig('rmp/levfig.png')