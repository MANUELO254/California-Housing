import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_excel('C:/Projects/CaliHouse/1553768847_housing.xlsx')

# Extract the 'median_income' column as the independent variable
X = data['median_income'].values.reshape(-1, 1)
Y = data['median_house_value'].values

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Perform Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, Y_train)

# Predict output for test dataset
Y_pred = linear_reg.predict(X_test)

# Plot the fitted model for training data and test data
plt.scatter(X_train, Y_train, color='blue', label='Training Data')
plt.scatter(X_test, Y_test, color='green', label='Test Data')
plt.plot(X_train, linear_reg.predict(X_train), color='red', linewidth=2, label='Fitted Model')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Linear Regression - Median Income vs Median House Value')
plt.legend()
plt.show()
