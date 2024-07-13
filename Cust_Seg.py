#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[65]:


df = pd.read_csv(r'E:\Dataset\ifood_df.csv')


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


df.isna().sum()
sns.heatmap(df.isna())


# In[9]:


df.duplicated().sum()


# In[10]:


df.drop_duplicates()


# In[11]:


df.duplicated().sum()


# In[13]:


df.nunique()


# In[15]:


df1=df.drop_duplicates()
df1.to_csv('ifood_copy.csv',index=False)


# In[16]:


df1.duplicated().sum()


# In[17]:


df.info()


# In[18]:


stats = ['Income','Age','Recency','NumDealsPurchases','MntTotal','MntRegularProds','AcceptedCmpOverall']
df2 = df1[stats].copy()


# In[19]:


df2.describe()


# In[20]:


sns.heatmap(df2.corr(),annot=True)


# In[21]:


plt.figure(figsize=(6, 4))
sns.boxplot(data=df1, y='MntTotal')
plt.title('Box Plot for MntTotal')
plt.ylabel('MntTotal')
plt.show()


# In[25]:


plt.figure(figsize=(6, 4))
sns.boxplot(data=df1, y='Income', color='red')
plt.title('Box Plot for Income')
plt.ylabel('MntTotal')
plt.show()


# In[26]:


plt.figure(figsize=(6, 6))
sns.histplot(data=df, x='Income', bins=30, kde=True)
plt.title('Histogram for Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()


# In[27]:


plt.figure(figsize=(8, 6))  
sns.histplot(data=df, x='Age', bins=30, kde=True, color = 'green')
plt.title('Histogram for Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[28]:


def get_relationship(row):
    if row['marital_Married'] ==1:
        return 1
    elif row['marital_Together'] == 1:
        return 1
    else:
        return 0
df['In_relationship'] = df.apply(get_relationship, axis=1)
df.head()


# In[39]:


print(df2.columns)


# In[46]:


def get_marital_status(row):
    if row['marital_Divorced'] == 1:
        return 'Divorced'
    elif row['marital_Married'] == 1:
        return 'Married'
    elif row['marital_Single'] == 1:
        return 'Single'
    elif row['marital_Together'] == 1:
        return 'Together'
    elif row['marital_Widow'] == 1:
        return 'Widow'
    else:
        return 'Unknown'
df['Marital'] = df.apply(get_marital_status, axis=1)


# In[47]:


plt.figure(figsize=(8, 6))
sns.barplot(x='Marital', y='MntTotal', data=df, palette='viridis')
plt.title('MntTotal by marital status')
plt.xlabel('Marital status')
plt.ylabel('MntTotal')


# In[48]:


x1 = df.loc[:, ['AcceptedCmpOverall', 'In_relationship']].values
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10)
    kmeans.fit(x1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker = '*')
plt.title('Elbow graph')
plt.xlabel('AcceptedCmpOverall')
plt.ylabel('In_relationship')
plt.show()


# In[49]:


from sklearn.metrics import silhouette_score
silhouette_list = []
for K in range(2,10):
    model = KMeans(n_clusters = K, n_init = 10)
    clusters = model.fit_predict(x1)
    s_avg = silhouette_score(x1, clusters)
    silhouette_list.append(s_avg)

plt.figure(figsize=[7,5])
plt.plot(range(2,10), silhouette_list, color='b', marker = '*')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()


# In[50]:


kmeans = KMeans(n_clusters = 4)
labels = kmeans.fit_predict(x1)
print(labels)


# In[51]:


print(kmeans.cluster_centers_)


# In[52]:


plt.figure(figsize = (14, 8))

plt.scatter(x1[:, 0], x1[:, 1], c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red')
plt.title('Clusters of Customers\n', fontsize = 20)
plt.xlabel('aceepted overall campaign')
plt.ylabel('In_relationship')
plt.show()


# In[53]:


import pandas as pd

cluster_df = pd.DataFrame(x1, columns=['AcceptedCmpOverall', 'In_relationship'])
cluster_df['Cluster-1'] = labels
cluster_percentage = cluster_df['Cluster-1'].value_counts(normalize=True) * 100
cluster_percentage = cluster_percentage.rename('Percentage').reset_index()

print(cluster_percentage)


# In[54]:


x2 = df.loc[:, ['NumDealsPurchases', 'MntRegularProds']].values
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init = 10)
    kmeans.fit(x2)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker = 'o')
plt.title('elbow graph')
plt.xlabel('NumDealsPurchases')
plt.ylabel('MntRegularProds')
plt.show()


# In[55]:


kmeans = KMeans(n_clusters = 4)
labels = kmeans.fit_predict(x2)
print(labels)


# In[56]:


print(kmeans.cluster_centers_)


# In[57]:


plt.figure(figsize = (14,8))

plt.scatter(x2[:, 0], x2[:, 1], c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red')
plt.title('Clusters of Customers\n', fontsize = 20)
plt.xlabel('deals purchase')
plt.ylabel('regular products')
plt.show()


# In[58]:


import pandas as pd

cluster_df = pd.DataFrame(x2, columns=['NumDealsPurchases', 'MntRegularProds'])
cluster_df['Cluster-2'] = labels
cluster_percentage = cluster_df['Cluster-2'].value_counts(normalize=True) * 100
cluster_percentage = cluster_percentage.rename('Percentage').reset_index()

print(cluster_percentage)


# In[59]:


x3 = df.loc[:, ['Income', 'MntTotal']].values
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init = 10)
    kmeans.fit(x3)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker = 'o')
plt.title('elbow graph')
plt.xlabel('Income')
plt.ylabel('MntTotal')
plt.show()


# In[60]:


# k means prediction for cluster 3
kmeans = KMeans(n_clusters = 4)
labels = kmeans.fit_predict(x3)
print(labels)


# In[61]:


print(kmeans.cluster_centers_)


# In[62]:


plt.figure(figsize = (14, 8))

plt.scatter(x3[:, 0], x3[:, 1], c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red')
plt.title('Clusters of Customers\n', fontsize = 20)
plt.xlabel('Income')
plt.ylabel('MntTotal')
plt.show()


# In[63]:


import pandas as pd

cluster_df = pd.DataFrame(x3, columns=['Income', 'MntTotal'])
cluster_df['Cluster-3'] = labels
cluster_percentage = cluster_df['Cluster-3'].value_counts(normalize=True) * 100
cluster_percentage = cluster_percentage.rename('Percentage').reset_index()

print(cluster_percentage)

