#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[36]:


salary_dataset = pd.read_csv('salary.csv')
salary_dataset.head()


# In[3]:


X = np.array(salary_dataset['YearsExperience'])
y = np.array(salary_dataset['Salary'])


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)
lx = len(X_train)


# In[22]:


m = 1
c = 0.001
alpha = 0.01
n = 10000

for i in range(n):
    slope = 0
    intercept = 0
    for j in range(lx):
        intercept = intercept+((m*X_train[j]+c)-y_train[j])
        slope = slope+((m*X_train[j]+c)-y_train[j])*X_train[j]
    
    c = c-alpha*(intercept/lx)
    m = m-alpha*(slope/lx)
    


# In[23]:


print("Slope: %f" % m)
print("Intercept: %f" %c)


# In[15]:


y_pred = m*X_test + c
y_pred


# In[16]:


plt.plot(X_test, y_pred, color="blue", marker ="o",markerfacecolor="red")
plt.scatter(X,y, color="red", marker ="o")


# In[38]:


from sklearn.metrics import r2_score, mean_absolute_error


# In[39]:


r2_score = r2_score(y_pred, y_test)
print("R2_Score : %.2f" %(r2_score*100)+"%")

