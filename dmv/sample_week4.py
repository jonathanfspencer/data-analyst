# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:51:26 2015

@author: ldierker
"""

import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

# any additional libraries would be imported here

#Set PANDAS to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

data = pandas.read_csv('nesarc_pds.csv', low_memory=False)

print(len(data)) #number of observations (rows)
print(len(data.columns)) # number of variables (columns)

# checking the format of your variables
data['ETHRACE2A'].dtype

#setting variables you will be working with to numeric (updated)
data['TAB12MDX'] = pandas.to_numeric(data['TAB12MDX'])
data['CHECK321'] = pandas.to_numeric(data['CHECK321'])
data['S3AQ3B1'] = pandas.to_numeric(data['S3AQ3B1'])
data['S3AQ3C1'] = pandas.to_numeric(data['S3AQ3C1'])
data['AGE'] = pandas.to_numeric(data['AGE'])

#subset data to young adults age 18 to 25 who have smoked in the past 12 months
sub1=data[(data['AGE']>=18) & (data['AGE']<=25) & (data['CHECK321']==1)]

#make a copy of my new subsetted data
sub2 = sub1.copy()

#SETTING MISSING DATA
# recode missing values to python missing (NaN)
sub2['S3AQ3B1']=sub2['S3AQ3B1'].replace(9, numpy.nan)
# recode missing values to python missing (NaN)
sub2['S3AQ3C1']=sub2['S3AQ3C1'].replace(99, numpy.nan)

recode1 = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
sub2['USFREQ']= sub2['S3AQ3B1'].map(recode1)

recode2 = {1: 30, 2: 22, 3: 14, 4: 5, 5: 2.5, 6: 1}
sub2['USFREQMO']= sub2['S3AQ3B1'].map(recode2)

# A secondary variable multiplying the number of days smoked/month and the approx number of cig smoked/day
sub2['NUMCIGMO_EST']=sub2['USFREQMO'] * sub2['S3AQ3C1']

#univariate bar graph for categorical variables
# First hange format from numeric to categorical
sub2["TAB12MDX"] = sub2["TAB12MDX"].astype('category')

seaborn.countplot(x="TAB12MDX", data=sub2)
plt.xlabel('Nicotine Dependence past 12 months')
plt.title('Nicotine Dependence in the Past 12 Months Among Young Adult Smokers in the NESARC Study')

#Univariate histogram for quantitative variable:
seaborn.distplot(sub2["NUMCIGMO_EST"].dropna(), kde=False)
plt.xlabel('Number of Cigarettes per Month')
plt.title('Estimated Number of Cigarettes per Month among Young Adult Smokers in the NESARC Study')
###############################################################################
# Code for Week 4 Python Lesson 3 - Measures of Center & Spread  

# standard deviation and other descriptive statistics for quantitative variables
print ('describe number of cigarettes smoked per month')
desc1 = sub2['NUMCIGMO_EST'].describe()
print (desc1)

c1= sub2.groupby('NUMCIGMO_EST').size()
print (c1)

print ('describe nicotine dependence')
desc2 = sub2['TAB12MDX'].describe()
print (desc2)

c1= sub2.groupby('TAB12MDX').size()
print (c1)

print ('mode')
mode1 = sub2['TAB12MDX'].mode()
print (mode1)

print ('mean')
mean1 = sub2['NUMCIGMO_EST'].mean()
print (mean1)

print ('std')
std1 = sub2['NUMCIGMO_EST'].std()
print (std1)

print ('min')
min1 = sub2['NUMCIGMO_EST'].min()
print (min1)

print ('max')
max1 = sub2['NUMCIGMO_EST'].max()
print (max1)

print ('median')
median1 = sub2['NUMCIGMO_EST'].median()
print (median1)

print ('mode')
mode1 = sub2['NUMCIGMO_EST'].mode()
print (mode1)


c1= sub2.groupby('TAB12MDX').size()
print (c1)

p1 = sub2.groupby('TAB12MDX').size() * 100 / len(data)
print (p1)


c2 = sub2.groupby('NUMCIGMO_EST').size()
print (c2)

p2 = sub2.groupby('NUMCIGMO_EST').size() * 100 / len(data)
print (p2)

# A secondary variable multiplying the number of days smoked per month and the approx number of cig smoked per day
# A secondary variable multiplying the number of days smoked per month and the approx number of cig smoked per day
sub2['PACKSPERMONTH']=sub2['NUMCIGMO_EST'] / 20

c2= sub2.groupby('PACKSPERMONTH').size()
print (c2)

sub2['PACKCATEGORY'] = pandas.cut(sub2.PACKSPERMONTH, [0, 5, 10, 20, 30, 147])

# change format from numeric to categorical
sub2['PACKCATEGORY'] = sub2['PACKCATEGORY'].astype('category')

print ('pack category counts')
c7 = sub2['PACKCATEGORY'].value_counts(sort=False, dropna=True)
print(c7)

print ('describe PACKCATEGORY')
desc3 = sub2['PACKCATEGORY'].describe()
print (desc3)


sub2['TAB12MDX'] = pandas.to_numeric(sub2['TAB12MDX'])

# bivariate bar graph C->Q
seaborn.catplot(x="PACKCATEGORY", y="TAB12MDX", data=sub2, kind="bar", ci=None)
plt.xlabel('Packs per Month')
plt.ylabel('Proportion Nicotine Dependent')

#creating 3 level smokegroup variable
def SMOKEGRP (row):
   if row['TAB12MDX'] == 1 :
      return 1
   elif row['USFREQMO'] == 30 :
      return 2
   else :
      return 3
         
sub2['SMOKEGRP'] = sub2.apply (lambda row: SMOKEGRP (row),axis=1)

c3= sub2.groupby('SMOKEGRP').size()
print (c3)

#creating daily smoking variable
def DAILY (row):
   if row['USFREQMO'] == 30 :
      return 1
   elif row['USFREQMO'] != 30 :
      return 0
      
sub2['DAILY'] = sub2.apply (lambda row: DAILY (row),axis=1)
      
c4= sub2.groupby('DAILY').size()
print (c4)


seaborn.catplot(x='ETHRACE2A', y='DAILY', data=sub2, kind="bar", ci=None)
plt.xlabel('Ethnic Group')
plt.ylabel('Proportion Daily Smokers')

# you can rename categorical variable values for graphing if original values are not informative 
# first change the variable format to categorical if you havenâ€™t already done so
sub2['ETHRACE2A'] = sub2['ETHRACE2A'].astype('category')
# second create a new variable (PACKCAT) that has the new variable value labels
sub2['ETHRACE2A']=sub2['ETHRACE2A'].cat.rename_categories(["White", "Black", "NatAm", "Asian", "Hispanic"])

# bivariate bar graph C->C
seaborn.catplot(x='ETHRACE2A', y='DAILY', data=sub2, kind="bar", ci=None)
plt.xlabel('Ethnic Group')
plt.ylabel('Proportion Daily Smokers')

#check to see if missing data were set to NaN 
print ('counts for S3AQ3C1 with 99 set to NAN and number of missing requested')
c4 = sub2['S3AQ3C1'].value_counts(sort=False, dropna=False)
print(c4)

print ('counts for TAB12MDX - past 12 month nicotine dependence')
c5 = sub2['TAB12MDX'].value_counts(sort=False)
print(c5)


#ADDHEALTH DATA EXAMPLE

import pandas
import numpy

data = pandas.read_csv('addhealth_pds.csv', low_memory=False)

#making individual ethnicity variables numeric
data['H1GI4'] = pandas.to_numeric(data['H1GI4'])
data['H1GI6A'] = pandas.to_numeric(data['H1GI6A'])
data['H1GI6B'] = pandas.to_numeric(data['H1GI6B'])
data['H1GI6C'] = pandas.to_numeric(data['H1GI6C'])
data['H1GI6D'] = pandas.to_numeric(data['H1GI6D'])


#Set missing data to NAN
data['H1GI4']=data['H1GI4'].replace(6, numpy.nan)
data['H1GI4']=data['H1GI4'].replace(8, numpy.nan)
data['H1GI6A']=data['H1GI6A'].replace(6, numpy.nan)
data['H1GI6A']=data['H1GI6A'].replace(8, numpy.nan)
data['H1GI6B']=data['H1GI6B'].replace(6, numpy.nan)
data['H1GI6B']=data['H1GI6B'].replace(8, numpy.nan)
data['H1GI6C']=data['H1GI6C'].replace(6, numpy.nan)
data['H1GI6C']=data['H1GI6C'].replace(8, numpy.nan)
data['H1GI6D']=data['H1GI6D'].replace(6, numpy.nan)
data['H1GI6D']=data['H1GI6D'].replace(8, numpy.nan)

#count of number of ethnicity categories endorsed, NUMETHNIC
data['NUMETHNIC']=data['H1GI4'] + data['H1GI6A'] + data['H1GI6B'] + data['H1GI6C'] + data['H1GI6D'] 

print ('counts for NUMETHNIC')
c10 = data['NUMETHNIC'].value_counts(sort=False)
print(c10)

#new ETHNICITY variable, categorical 1 through 6
def ETHNICITY (row):
   if row['NUMETHNIC'] > 1 :
      return 1
   if row['H1GI4'] == 1 :
      return 2
   if row['H1GI6A'] == 1:
      return 3
   if row['H1GI6B'] == 1:
      return 4
   if row['H1GI6C'] == 1:
      return 5
   if row['H1GI6D'] == 1:
      return 6
data['ETHNICITY'] = data.apply (lambda row: ETHNICITY (row),axis=1)

# subset variables in new data frame, sub1
sub1=data[['AID','H1GI4', 'H1GI6A', 'H1GI6B', 'H1GI6C', 'H1GI6D', 'NUMETHNIC', 'ETHNICITY']]
a = sub1.head (n=25)
print(a)

#frequency distributions for primary and secondary ethinciity variables
print ('counts for Hispanic/Latino')
c10 = sub1['H1GI4'].value_counts(sort=False)
print(c10)

print ('percentages for Hispanic/Latino')
p10 = sub1['H1GI4'].value_counts(sort=False, normalize=True)
print (p10)

print ('counts for Black/African American')
c11 = sub1['H1GI6A'].value_counts(sort=False)
print(c11)

print ('percentages for Black/African American')
p11= sub1['H1GI6A'].value_counts(sort=False, normalize=True)
print (p11)

print ('counts for American Indian/Native American')
c12 = sub1['H1GI6B'].value_counts(sort=False)
print(c12)

print ('percentages for American Indian/Native American')
p12 = sub1['H1GI6B'].value_counts(sort=False, normalize=True)
print (p12)

print ('counts for Asian/Pacific Islander')
c13 = sub1['H1GI6C'].value_counts(sort=False)
print(c13)

print ('percentages for Asian/Pacific Islander')
p13 = sub1['H1GI6C'].value_counts(sort=False, normalize=True)
print (p13)

print ('counts for White')
c14 = sub1['H1GI6D'].value_counts(sort=False)
print(c14)

print ('percentages for White')
p14 = sub1['H1GI6D'].value_counts(sort=False, normalize=True)
print (p14)

print ('counts for number of races/ethnicities endorsed')
c15 = sub1['NUMETHNIC'].value_counts(sort=False)
print(c15)

print ('percentages for number of races/ethnicities endorsed')
p16 = sub1['NUMETHNIC'].value_counts(sort=False, normalize=True)
print (p16)


# GAPMINDER DATA EXAMPLE

import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

# any additional libraries would be imported here
data = pandas.read_csv('gapminder.csv', low_memory=False)

data = data.replace(r'^\s*$', numpy.NaN, regex=True)

#setting variables you will be working with to numeric
data['internetuserate'] = pandas.to_numeric(data['internetuserate'])
data['urbanrate'] = pandas.to_numeric(data['urbanrate'])

desc1 = data['urbanrate'].describe()
print (desc1)

desc2 = data['internetuserate'].describe()
print (desc2)

#basic scatterplot:  Q->Q
scat1 = seaborn.regplot(x="urbanrate", y="internetuserate", fit_reg=False, data=data)
plt.xlabel('Urban Rate')
plt.ylabel('Internet Use Rate')
plt.title('Scatterplot for the Association Between Urban Rate and Internet Use Rate')

scat2 = seaborn.regplot(x="urbanrate", y="internetuserate", data=data)
plt.xlabel('Urban Rate')
plt.ylabel('Internet Use Rate')
plt.title('Scatterplot for the Association Between Urban Rate and Internet Use Rate')

scat3 = seaborn.regplot(x="incomeperperson", y="internetuserate", data=data)
plt.xlabel('Income per Person')
plt.ylabel('Internet Use Rate')
plt.title('Scatterplot for the Association Between Income per Person and Internet Use Rate')

scat4 = seaborn.regplot(x="incomeperperson", y="hivrate", data=data)
plt.xlabel('Income per Person')
plt.ylabel('HIV Rate')
plt.title('Scatterplot for the Association Between Income per Person and HIV Rate')

# quartile split (use qcut function & ask for 4 groups - gives you quartile split)
print ('Income per person - 4 categories - quartiles')
data['INCOMEGRP4']=pandas.qcut(data.incomeperperson, 4, labels=["1=25th%tile","2=50%tile","3=75%tile","4=100%tile"])
c10 = data['INCOMEGRP4'].value_counts(sort=False, dropna=True)
print(c10)

# bivariate bar graph C->Q
seaborn.catplot(x='INCOMEGRP4', y='hivrate', data=data, kind="bar", ci=None)
plt.xlabel('income group')
plt.ylabel('mean HIV rate')


c11= data.groupby('INCOMEGRP4').size()
print (c11)

result = data.sort(['INCOMEGRP4'], ascending=[1])
print(result)


