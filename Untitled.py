#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset =pd.read_csv("Data.csv")


# In[24]:


from sklearn.preprocessing import LabelEncoder
dataset.isnull().any()
lb=LabelEncoder()
dataset['country'] = lb.fit_transform(dataset['Country'])
lb1 = LabelEncoder()
dataset['Purchased'] = lb1.fit_transform(dataset['Purchased'])
dataset.head()


# In[29]:


x = dataset.iloc[:,0:3].values
y=  dataset.iloc[:,3].values


# In[27]:


from sklearn.model_selection import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[ ]:





# In[44]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from sklearn import preprocessing 

   
data_set = pd.read_csv('Data.csv') 
data_set.head() 
  
 #slicing
x = data_set.iloc[:, 1:3].values 
print (" values : \n",  x) 

from sklearn import preprocessing 
   #feature selection
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
  
# Scaled feature 
x_after_min_max_scaler = min_max_scaler.fit_transform(x) 
  
print (" Scaling : \n", x_after_min_max_scaler) 
Standardisation = preprocessing.StandardScaler() 
  
# standardisation
x_after_Standardisation = Standardisation.fit_transform(x) 
  
print (" Standardisation : \n", x_after_Standardisation)


# In[43]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[("oh",OneHotEncoder(),[0])], remainder="passthrough")
x =ct.fit_transform(x)

