#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
banner_promo = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/ecommerce_banner_promo.csv')

#Exploration
print('5 data teratas:\n',banner_promo.head())
print('Informasi dataset:\n',banner_promo.info())
print('Statistik diskriptif dataset:',banner_promo.describe())
print('Ukuran dataset:\n',banner_promo.shape)
print('Korelasi dataset:\n',banner_promo.corr())
print('Distribusi label:\n',banner_promo.groupby('Clicked on Ad').size())


# In[4]:


#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#Setting packaged
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
#Visualization count of customer based on age
plt.figure(figsize=(10,5))
plt.hist(banner_promo['Age'],bins=banner_promo.Age.nunique())
plt.title('Histogram of Customer Age', color='blue' )
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
#using pairplot() from seaborn (sns) module for describe relationship each feature
plt.figure()
sns.pairplot(banner_promo)
plt.show()


# In[5]:


#Chenking missing value

print(banner_promo.isnull().sum().sum())


# In[8]:


#Modelling with Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Split data to be x (variable independent) and y(variable dependent)
x=banner_promo.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'],axis=1)
y=banner_promo['Clicked on Ad']
#split x and y to be training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)
reglog=LogisticRegression()
model_reglog = reglog.fit(x_train,y_train)
y_pred =reglog.predict(x_test)

#Evaluation Model Performance
print('Training performance:',model_reglog.score(x_train,y_train))
print('Testing performance:',model_reglog.score(x_test,y_test))


# In[9]:


#Confusion matrix and classification report
from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))
print('Classification Report:\n',classification_report(y_test,y_pred))

