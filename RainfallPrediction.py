#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing


# In[2]:


data=pd.read_csv("C:\\Users\\prade\\Documents\\MiniProject-II\\rainfall.csv")
print("Data heads:")
print(data.head())
print("Null values in the dataset before preprocessing:")
print(data.isnull().sum())
print("Filling null values with mean of that particular column")
data=data.fillna(np.mean(data))
print("Mean of data:")
print(np.mean(data))
print("Null values in the dataset after preprocessing:")
print(data.isnull().sum())
print("\n\nShape: ",data.shape)


# In[3]:


print ("Info")
print(data.info())


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


data.hist(figsize=(24,24))


# In[7]:


print("Annual rainfall from Year 1900 to 2015")
data.groupby("YEAR").sum()['ANNUAL'].plot(figsize=(12,8))


# In[8]:


data[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("YEAR").sum().plot(figsize=(13,8))


# In[9]:


data[['YEAR','Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].groupby("YEAR").sum().plot(figsize=(13,8));


# In[10]:


data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").mean().plot.barh(stacked=True,figsize=(15,15))


# In[11]:


data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").mean().plot.barh(stacked=True,figsize=(15,15))


# In[12]:


plt.figure(figsize=(13,6)) 
sb.heatmap(data[['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec','ANNUAL']].corr(),annot=True) 
plt.show()


# In[13]:


plt.figure(figsize=(14,9)) 
sb.heatmap(data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].corr(),annot=True) 
plt.show()


# In[14]:


#Function to plot the graph
def plot_graphs(groundtruth,prediction,title):
    N=9
    ind = np.arange(N)
    width=0.27
    
    fig = plt.figure()
    fig.suptitle(title, fontsize=12) 
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, groundtruth, width, color='r') 
    rects2 = ax.bar(ind+width, prediction, width, color='g')
    
    ax.set_ylabel("Amount of rainfall") 
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC') ) 
    ax.legend( (rects1[0], rects2[0]), ('Ground truth', 'Prediction') )
 
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')

    plt.show()


# In[15]:


# seperation of training and testing data 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error
division_data = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])
X = None; y = None
for i in range(division_data.shape[1]-3):
    if X is None:
        X = division_data[:, i:i+3] 
        y = division_data[:, i+3]
    else:
        X = np.concatenate((X, division_data[:, i:i+3]), axis=0) 
        y = np.concatenate((y, division_data[:, i+3]), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[16]:


#test 2010 
temp = data[['SUBDIVISION','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2010]
data_2010 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == 'TAMIL NADU'])
X_year_2010 = None; y_year_2010 = None 
for i in range(data_2010.shape[1]-3): 
    if X_year_2010 is None: 
        X_year_2010 = data_2010[:, i:i+3] 
        y_year_2010 = data_2010[:, i+3] 
    else: 
        X_year_2010 = np.concatenate((X_year_2010, data_2010[:, i:i+3]), axis=0) 
        y_year_2010 = np.concatenate((y_year_2010, data_2010[:, i+3]), axis=0)


# In[17]:


#test 2005 
temp = data[['SUBDIVISION','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2005]
data_2005 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == 'TAMIL NADU'])
X_year_2005 = None; y_year_2005 = None 
for i in range(data_2005.shape[1]-3): 
    if X_year_2005 is None: 
        X_year_2005 = data_2005[:, i:i+3] 
        y_year_2005 = data_2005[:, i+3] 
    else: 
        X_year_2005 = np.concatenate((X_year_2005, data_2005[:, i:i+3]), axis=0) 
        y_year_2005 = np.concatenate((y_year_2005, data_2005[:, i+3]), axis=0)


# In[18]:


#test 2015 
temp = data[['SUBDIVISION','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2015]
data_2015 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == 'TAMIL NADU'])
X_year_2015 = None; y_year_2015 = None 
for i in range(data_2015.shape[1]-3):
    if X_year_2015 is None: 
        X_year_2015 = data_2015[:, i:i+3] 
        y_year_2015 = data_2015[:, i+3] 
    else: 
        X_year_2015 = np.concatenate((X_year_2015, data_2015[:, i:i+3]), axis=0) 
        y_year_2015 = np.concatenate((y_year_2015, data_2015[:, i+3]), axis=0)


# In[19]:


from sklearn import linear_model
# linear model 
reg = linear_model.ElasticNet(alpha=0.5) 
reg.fit(X_train, y_train) 
y_pred = reg.predict(X_test) 
print(mean_absolute_error(y_test, y_pred))


# In[20]:


#2005 
y_year_pred_2005 = reg.predict(X_year_2005)

#2010 
y_year_pred_2010 = reg.predict(X_year_2010)

#2015
y_year_pred_2015 = reg.predict(X_year_2015)

print("MEAN 2005") 
print(np.mean(y_year_2005),np.mean(y_year_pred_2005)) 
print("Standard deviation 2005") 
print(np.sqrt(np.var(y_year_2005)),np.sqrt(np.var(y_year_pred_2005)))

print("MEAN 2010")
print(np.mean(y_year_2010),np.mean(y_year_pred_2010)) 
print("Standard deviation 2010") 
print(np.sqrt(np.var(y_year_2010)),np.sqrt(np.var(y_year_pred_2010)))

print("MEAN 2015") 
print(np.mean(y_year_2015),np.mean(y_year_pred_2015)) 
print("Standard deviation 2015") 
print(np.sqrt(np.var(y_year_2015)),np.sqrt(np.var(y_year_pred_2015)))

plot_graphs(y_year_2005,y_year_pred_2005,"Year-2005") 
plot_graphs(y_year_2010,y_year_pred_2010,"Year-2010") 
plot_graphs(y_year_2015,y_year_pred_2015,"Year-2015")


# In[21]:


# spliting training and testing data only for Tamil Nadu 
tamilNadu = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['SUBDIVISION'] == 'TAMIL NADU'])
X = None; y = None 
for i in range(tamilNadu.shape[1]-3): 
    if X is None: 
        X = tamilNadu[:, i:i+3] 
        y = tamilNadu[:, i+3] 
    else: 
        X = np.concatenate((X, tamilNadu[:, i:i+3]), axis=0) 
        y = np.concatenate((y, tamilNadu[:, i+3]), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)


# In[22]:


from sklearn import linear_model
# linear model 
reg = linear_model.ElasticNet(alpha=0.5) 
reg.fit(X_train, y_train) 
y_pred = reg.predict(X_test) 
print (mean_absolute_error(y_test, y_pred))


# In[23]:


#2005 
y_year_pred_2005 = reg.predict(X_year_2005)

#2010 
y_year_pred_2010 = reg.predict(X_year_2010)

#2015 
y_year_pred_2015 = reg.predict(X_year_2015)

print ("MEAN 2005") 
print (np.mean(y_year_2005),np.mean(y_year_pred_2005))
print ("Standard deviation 2005")
print (np.sqrt(np.var(y_year_2005)),np.sqrt(np.var(y_year_pred_2005)))

print ("MEAN 2010") 
print (np.mean(y_year_2010),np.mean(y_year_pred_2010)) 
print ("Standard deviation 2010") 
print (np.sqrt(np.var(y_year_2010)),np.sqrt(np.var(y_year_pred_2010)))

print ("MEAN 2015") 
print (np.mean(y_year_2015),np.mean(y_year_pred_2015)) 
print ("Standard deviation 2015") 
print (np.sqrt(np.var(y_year_2015)),np.sqrt(np.var(y_year_pred_2015)))

plot_graphs(y_year_2005,y_year_pred_2005,"Year-2005")
plot_graphs(y_year_2010,y_year_pred_2010,"Year-2010") 
plot_graphs(y_year_2015,y_year_pred_2015,"Year-2015")


# In[ ]:




