#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.linear_model import LinearRegression 


# In[2]:


# Load dataset 
dataset = pd.read_csv('Salary_Data.csv')  # Ensure correct closing quote 
print(dataset) 


# In[3]:


# Extract independent variable (Years of Experience) and dependent variable 
X = dataset.iloc[:, :-1].values  # All rows, all columns except the last 
y = dataset.iloc[:, 1].values    # All rows, second column (Salary) 

print(y)


# In[4]:


# Split the dataset into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0) 
print(X_train) 


# In[5]:


print(y_train) 


# In[6]:


# Create and train the Linear Regression model 
regressor = LinearRegression() 
regressor.fit(X_train, y_train) 


# In[7]:


# Visualising the Training set results 
# Predicting the Test set results 
y_pred = regressor.predict(X_test) 
print(y_pred)
plt.scatter(X_train, y_train, color='red')  # Actual training data points 
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Regression line 
plt.title('Salary vs Experience (Training set)') 
plt.xlabel('Years of Experience') 
plt.ylabel('Salary') 
plt.show() 
# Visualising the Test set results 
plt.scatter(X_test, y_test, color='green')  # Actual test data points 
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Same regression line 
plt.title('Salary vs Experience (Test set)') 
plt.xlabel('Years of Experience') 
plt.ylabel('Salary') 
plt.show() 

