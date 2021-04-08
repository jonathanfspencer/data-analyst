# -*- coding: utf-8 -*-
"""
MLDA Week 4
@author: Jonathan Spencer

This week’s assignment involves running a k-means cluster analysis. 
Cluster analysis is an unsupervised machine learning method that 
partitions the observations in a data set into a smaller set of 
clusters where each observation belongs to only one cluster. 
The goal of cluster analysis is to group, or cluster, observations 
into subsets based on their similarity of responses on multiple 
variables. Clustering  variables should be primarily quantitative 
variables, but binary variables may also be included.

Your assignment is to run a k-means cluster analysis to identify 
subgroups of observations in your data set that have similar patterns 
of response on a set of clustering variables. 

WHAT TO SUBMIT:

Following completion of the steps described above, create a blog 
entry where you submit syntax used to run a k-means cluster analysis 
(copied and pasted from your program) along with corresponding output 
and a brief written summary. Please note that your reviewers should NOT
be required to download any files in order to complete the review.

This assignment does NOT require you to run your cluster analysis 
again on a test data set. You are welcome to do so, but you are only 
required to run your cluster analysis once on your training data set. 
If your data set has a relatively small number of observations, you 
do not need to split into training and test data sets. You can provide 
your rationale for not splitting your data set in your written summary.  
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans


#Load the dataset
#Set PANDAS to show all columns in DataFrame
pd.set_option('display.max_columns', None)
#Set PANDAS to show all rows in DataFrame
pd.set_option('display.max_rows', None)

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%f'%x)

data = pd.read_csv('dat/gapminder.csv', low_memory=False)
data = data.replace(r'^\s*$', np.NaN, regex=True)

data_clean = data.dropna()

# data prep
cluster_names = ['incomeperperson','co2emissions','femaleemployrate','polityscore',
'alcconsumption','breastcancerper100th','employrate','hivrate','internetuserate',
'lifeexpectancy','oilperperson','relectricperperson','suicideper100th','urbanrate']
cluster = data_clean[cluster_names]

print('Information about our data:')
print(data_clean.dtypes)
print(cluster.describe())
print()


# standardize predictors to have mean=0 and sd=1
clustervar=cluster.copy()
from sklearn import preprocessing
for n in cluster_names:
    clustervar[str(n)]=preprocessing.scale(clustervar[str(n)].astype('float64'))

# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

print('Cluster train shape:')
print(clus_train.shape)
print('Cluster test shape')
print(clus_test.shape)
print()

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

# Plot average distance from observations from the cluster centroid
# to use the Elbow Method to identify number of clusters to choose

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
# plt.show()

# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
# plt.show()

# BEGIN multiple steps to merge cluster assignment with clustering variables to examine
# cluster variable means by cluster


# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=pd.DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
print('merged training data:')
print(merged_train.head(n=100))
# cluster frequencies
merged_train.cluster.value_counts()

# END multiple steps to merge cluster assignment with clustering variables to examine
# cluster variable means by cluster

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)


# validate clusters in training data by examining cluster differences in income using ANOVA
# first have to merge income with clustering variables and cluster assignment data 
income_data=data_clean['incomeperperson']
print('Just income data:')
print(income_data)
print()
# split income data into train and test sets
income_train, income_test = train_test_split(income_data, test_size=.3, random_state=123)
income_train1=pd.DataFrame(income_train)
income_train1.reset_index(level=0, inplace=True)
income_train1.reindex_like(merged_train)
#income_train1 = income_train1.set_index('index')
print('income train 1:')
print(income_train1)
merged_train_all=pd.merge(income_train1, merged_train, on='index')
# merged_train_all=income_train1.join(merged_train.set_index('index'), on='index')
print('merged train all after merge:')
print(merged_train_all)
#columns = ['incomeperperson', 'cluster']
#merged_train_all=merged_train_all.reindex(columns=columns)
#print('merged train all after reindex:')
#print(merged_train_all)
sub1 = merged_train_all[['incomeperperson_x', 'cluster']].dropna()
print('data before OLS:')
print(sub1)

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

sub1['incomeperperson_x'] = pd.to_numeric(sub1['incomeperperson_x'])
incomemod = smf.ols(formula='incomeperperson_x ~ C(cluster)', data=sub1).fit()
print (incomemod.summary())
print()

print ('means for income by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)
print()

print ('standard deviations for income by cluster')
m2= sub1.groupby('cluster').std()
print (m2)
print()

mc1 = multi.MultiComparison(sub1['incomeperperson_x'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())
