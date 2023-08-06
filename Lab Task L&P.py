#!/usr/bin/env python
# coding: utf-8

# # Predicting Student Marks
# Objective: The objective of this lab task is to build a linear and polynomial regression model to predict student marks based on the number of courses taken and the number of study hours.
# Also calculate MSE, RMSE, MAE.

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[22]:


#load the dataset step 1
df = pd.read_csv('C:/Users/Farman vcds/Desktop/Nust/Student_Marks.csv')
df


# In[24]:


# Display the first few rows of the dataset
print(df.head())


# In[25]:


# Get information about the dataset
print(df.info())


# In[27]:


# Statistical summary of the dataset
print(df.describe())


# In[28]:


# Separate the features and target variable step 2
X = df[['number_courses', 'time_study']]
y = df['Marks']


# In[29]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


#  Build the linear regression model
from sklearn.linear_model import LinearRegression

# Create a linear regression model
linear_model = LinearRegression()

# Fit the model to the training data
linear_model.fit(X_train, y_train)


# In[31]:


# Build the polynomial regression model
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Split the transformed features into training and testing sets (if required)
X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Create a linear regression model with polynomial features
poly_model = LinearRegression()

# Fit the model to the training data with polynomial features
poly_model.fit(X_poly_train, y_train)


# In[32]:


# Make predictions
# Predict using the linear regression model
y_linear_pred = linear_model.predict(X_test)
# Predict using the polynomial regression model
y_poly_pred = poly_model.predict(poly_features.transform(X_test))


# In[33]:


# Evaluate the models Calculate the MSE, RMSE, and MAE to evaluate the performance of the models.
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Calculate MSE for linear regression
mse_linear = mean_squared_error(y_test, y_linear_pred)
# Calculate RMSE for linear regression
rmse_linear = mean_squared_error(y_test, y_linear_pred, squared=False)
# Calculate MAE for linear regression
mae_linear = mean_absolute_error(y_test, y_linear_pred)
# Calculate MSE for polynomial regression
mse_poly = mean_squared_error(y_test, y_poly_pred)
# Calculate RMSE for polynomial regression
rmse_poly = mean_squared_error(y_test, y_poly_pred, squared=False) # Calculate MAE for polynomial regression
mae_poly = mean_absolute_error(y_test, y_poly_pred)
print("Linear Regression:")
print("MSE:", mse_linear)
print("RMSE:", rmse_linear)
print("MAE:", mae_linear)
print("\nPolynomial Regression:")
print("MSE:", mse_poly)
print("RMSE:", rmse_poly)
print("MAE:", mae_poly)


# In[ ]:





# In[ ]:





# In[ ]:




