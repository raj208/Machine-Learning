#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[9]:


train = pd.read_csv('titanic_train.csv')


# In[10]:


train.head()


# In[11]:


#seaching for the null values
train.info()


# In[12]:


train.isnull()


# In[13]:


sns.heatmap(train.isnull())


# In[ ]:





# In[18]:


sns.countplot(x = 'Survived', data = train, hue = 'Sex')


# In[19]:


sns.countplot(x = 'Survived', data = train, hue = 'Pclass')


# In[20]:


sns.distplot(train['Pclass'],kde = False)


# In[21]:


sns.displot(train['Age'].dropna(), kde = True)


# In[22]:


sns.countplot(x = 'SibSp', data = train)


# In[23]:


train['Fare'].hist(bins = 40, figsize = (8,12))


# In[ ]:


#Preprocessing data (cleaning data)


# In[24]:


sns.boxplot(x='Pclass', y = 'Age', data = train)


# In[25]:


#filling null value of age with its median
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 25
    else:
        return Age


# In[26]:


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)


# In[30]:


sns.heatmap(train.isnull())


# In[ ]:


train.drop('Cabin', axis = 1, inplace = True) #dropping cabin column


# In[36]:


pd.get_dummies(train['Sex']) #assigning string into binary


# In[39]:


pd.get_dummies(train['Embarked'])


# In[41]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)


# In[46]:


train.drop(['Sex','Embarked','Name','Ticket'], axis = 1, inplace = True)


# In[47]:


train.head()


# In[48]:


train = pd.concat([train,sex, embark],axis = 1)


# In[49]:


train.head()


# In[56]:


#Logistic Regression MOdel
from sklearn.model_selection import train_test_split


# In[58]:


#dropping Survived column and using all other columns as inputs
x_train, x_test, y_train, y_test = train_test_split(train.drop('Survived', axis = 1),train['Survived'], test_size = 0.30, random_state = 101)


# In[59]:


from sklearn.linear_model import LogisticRegression


# In[60]:


logmodel = LogisticRegression()


# In[61]:


logmodel.fit(x_train, y_train)


# In[62]:


predictions = logmodel.predict(x_test)


# In[64]:


from sklearn.metrics import classification_report,confusion_matrix


# In[67]:


confusion_matrix(y_test, predictions)


# In[70]:


print(classification_report(y_test, predictions))

