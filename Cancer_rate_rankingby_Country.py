#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[28]:


dataset=pd.read_csv("Cancer_rate_by_countries.csv")


# In[29]:


print(type(dataset))


# In[49]:


dataset.head()


# In[51]:


country_data = pd.get_dummies(dataset.Country)
print(country_data)


# In[52]:


new_data = dataset.drop(['Country'], axis=1)
new_data = pd.concat((new_data, country_data), axis=1)
print(new_data)


# In[53]:


x1=new_data.iloc[:,2:]


# In[54]:


x1


# In[55]:


y1=new_data.Rank


# In[56]:


y1


# In[57]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x1,y1,test_size=0.2)


# In[68]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[70]:


y_pred=model.predict(X_test)


# In[74]:


y_pred


# In[75]:


from sklearn.metrics import r2_score


# In[76]:


r2_score(y_pred,y_test)


# In[ ]:




