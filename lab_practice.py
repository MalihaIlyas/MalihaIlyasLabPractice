#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[37]:


df1 = pd.read_csv('text.csv')


# In[38]:


df1


# In[39]:


df1['Class'] = df1.Class.map({'cinema':0, 'education':1})
df1


# In[40]:


X= df1['Document'].to_numpy()
Y = df1['Class'].to_numpy()
y = Y.astype('int')


# In[41]:


print("X")
print(X)
print("y")
print(y)


# In[42]:


vec = CountVectorizer( )


# In[43]:


vec.fit(X)
vec.vocabulary_


# In[44]:


vec = CountVectorizer(stop_words='english' )
vec.fit(X)
vec.vocabulary_


# In[45]:


print(vec.get_feature_names())
print(len(vec.get_feature_names()))


# In[46]:


X_transformed=vec.transform(X)
X_transformed


# In[47]:


X=X_transformed.toarray()
X


# In[48]:


pd.DataFrame(X, columns=vec.get_feature_names())


# In[49]:


df2 = pd.read_csv('test_text.csv') 
df2


# In[53]:


test_numpy_array = df2.values
X_test = test_numpy_array[:, 0]
Y_test = test_numpy_array[:, 1]

y_test = Y_test.astype('int')
print("X_test")

print(X_test)
print("Y_test")
print(y_test)


# In[54]:


X_test_transformed=vec.transform(X_test)
X_test_transformed


# In[55]:


X_test=X_test_transformed.toarray()
X_test


# In[56]:


mnb=MultinomialNB()

mnb.fit(X,y)

mnb.predict_proba(X_test)


# In[58]:


y_prediction = mnb.predict(X_test)


acc = accuracy_score(y_test, y_prediction)


print("Accuracy:", acc)


# In[ ]:




