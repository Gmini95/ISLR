
# coding: utf-8

# In[413]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

get_ipython().run_line_magic('matplotlib', 'inline')


# In[414]:


auto=pd.read_csv('Auto.csv',engine='python',encoding='949')


# In[415]:


auto[auto['horsepower']=='?']


# In[416]:


df=auto[auto['horsepower']!='?']


# # The Validation Set Approach

# In[433]:


df['horsepower'] = pd.to_numeric(df['horsepower'])


# In[434]:


df_train,df_test=train_test_split(df,test_size=0.5,random_state=1)
print('train data size : {}'.format(df_train.shape))
print('test data size : {}'.format(df_test.shape))


# In[439]:


lm=smf.ols(formula='mpg ~ horsepower',data=df_train).fit()
y_pred=lm.predict(df_test)
print(mean_squared_error(df_test['mpg'],y_pred).round(3))


# In[440]:


poly=pd.DataFrame()
poly['mpg']=df['mpg']
poly['horsepower']=df['horsepower']
poly['horsepower2']=df['horsepower']*df['horsepower']
poly['horsepower3']=df['horsepower']*df['horsepower']*df['horsepower']


# In[441]:


poly_x=poly[['horsepower','horsepower2']]
poly_y=poly[['mpg']]
x2_train,x2_test,y2_train,y2_test=train_test_split(poly_x,poly_y,test_size=0.5,random_state=1)


# In[444]:


lm2=LinearRegression()
lm2.fit(x2_train,y2_train)
y2_pred=lm2.predict(x2_test)
print(mean_squared_error(y2_test,y2_pred).round(3))


# In[447]:


poly_x=poly[['horsepower','horsepower2','horsepower3']]
poly_y=poly[['mpg']]
x3_train,x3_test,y3_train,y3_test=train_test_split(poly_x,poly_y,test_size=0.5,random_state=1)


# In[449]:


lm3=LinearRegression()
lm3.fit(x3_train,y3_train)
y3_pred=lm3.predict(x3_test)
print(mean_squared_error(y3_test,y3_pred).round(3))


# # Leave-One-Out Cross-Validation

# In[452]:


formula='mpg ~ horsepower'
glm = smf.glm(formula = formula, data=df).fit()
print(glm.summary())


# In[453]:


lm=smf.ols(formula='mpg ~ horsepower',data=df).fit()
print(lm.summary())


# In[454]:


p_order = np.arange(1,6)
r_state = np.arange(0,10)

# LeaveOneOut CV
lm = LinearRegression()
loo = LeaveOneOut()
loo.get_n_splits(df)
scores = list()

for i in p_order:
    poly = PolynomialFeatures(i)
    x_poly = poly.fit_transform(df['horsepower'].values.reshape(-1,1))
    score = cross_val_score(lm, x_poly, df['mpg'], cv=loo, scoring='neg_mean_squared_error').mean()
    print(-score.round(2))


# # k-fold

# In[455]:


p_order = np.arange(1,11)

lm = LinearRegression()


for i in p_order:
    poly = PolynomialFeatures(i)
    x_poly = poly.fit_transform(df['horsepower'].values.reshape(-1,1))
    score = cross_val_score(lm, x_poly, df['mpg'], cv=10, scoring='neg_mean_squared_error').mean()
    print(-score.round(2))


# # The Bootstrap

# In[456]:


data=pd.DataFrame()
data['mpg']=df['mpg']
data['horsepower']=df['horsepower']


# In[457]:


def boot(data,i):
    df=data.sample(n=i,replace=True,random_state=1)
    lm=smf.ols(formula='mpg ~ horsepower',data=df).fit()
    print(lm.params.round(3))
    return lm.summary()
    


# In[458]:


boot(data,392)


# In[459]:


boot(data,1000)


# In[468]:


def boot2(data,i):
    df=data.sample(n=i,replace=True,random_state=1)
    df['horsepower2']=df['horsepower']*df['horsepower']
    lm=smf.ols(formula='mpg ~ horsepower+horsepower2',data=df).fit()
    print(lm.params.round(3))
    return lm.summary()


# In[469]:


boot2(data,1000)

