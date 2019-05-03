#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.externals import joblib
import numpy as np


# In[2]:


# Load the model from the file 
mlp_from_joblib = joblib.load('accel_gyro.pkl')


# In[3]:


Activity_list = ['stand',
                 'sit',
                 'walk',
                 'stairsup',
                 'stairsdown',
                 'bike']


# In[4]:


X_ip = np.array([list(map(float,input().split(',')))])


# In[5]:


# Use the loaded model to make predictions 
print(Activity_list[int(mlp_from_joblib.predict(X_ip)[0]-1)])

