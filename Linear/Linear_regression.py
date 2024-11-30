import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import random as rn

# Set random seed for reproducibility
rn.seed(1)

# File path
file_path = 'heart.csv'  # Ensure the file name is consistent

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"Error: {file_path} not found in {os.getcwd()}")

# Load the dataset
data = pd.read_csv(file_path)
print("File loaded successfully!")
print(data.head())  # Display the first few rows
print("Column Names:", data.columns)

# Convert categorical columns to numeric
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Display correlation matrix for numeric columns
print("Correlation Matrix (Numeric Columns):")
print(data_encoded.corr().to_string())

# Select features and target variable
X = data_encoded[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']]
Y = data_encoded['HeartDisease']

# Visualize pair plots
sns.pairplot(data, x_vars=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'], 
             y_vars='HeartDisease', kind='scatter')
plt.show()

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)
Y_pred_rounded = np.round(Y_pred)  # Round the predictions to 0 or 1 for accuracy

# Calculate and print the accuracy
accuracy = accuracy_score(Y_test, Y_pred_rounded)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

# Scatter Plot for Actual vs Predicted values
plt.figure(figsize=(8,6))
plt.scatter(Y_test, Y_pred_rounded, color='blue', label="Actual vs Predicted")
plt.title("Linear Regression: Actual vs Predicted (Heart Disease)")
plt.xlabel("Actual Heart Disease (0 or 1)")
plt.ylabel("Predicted Heart Disease (0 or 1)")
plt.legend()
plt.show()
