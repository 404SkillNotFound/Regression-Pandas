import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random as rn

# Load dataset
data = pd.read_csv('heart.csv')

# Print dataset columns and their data types
print("Columns in the dataset:")
print(data.columns)
print("\nData types:")
print(data.dtypes)

# Convert categorical variables to numeric using one-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

# Define features and target variable
X = data[["Age", "Cholesterol", "RestingBP", "MaxHR", "Oldpeak"] + [col for col in data.columns 
 if 'Sex_' in col or 'ChestPainType_' in col or 'RestingECG_' in col or 'ExerciseAngina_' in col or 'ST_Slope_' in col]]
Y = data["HeartDisease"]

# Scatter plots to visualize the relationship between features and the target variable (Heart Disease)
sns.scatterplot(x=data["Age"], y=data["HeartDisease"], color="blue", label="Age vs Heart Disease")
plt.title("Age vs Heart Disease")
plt.xlabel("Age")
plt.ylabel("Heart Disease (0 or 1)")
plt.show()

sns.scatterplot(x=data["Cholesterol"], y=data["HeartDisease"], color="green", label="Cholesterol vs Heart Disease")
plt.title("Cholesterol vs Heart Disease")
plt.xlabel("Cholesterol")
plt.ylabel("Heart Disease (0 or 1)")
plt.show()

sns.scatterplot(x=data["RestingBP"], y=data["HeartDisease"], color="red", label="RestingBP vs Heart Disease")
plt.title("RestingBP vs Heart Disease")
plt.xlabel("Resting BP")
plt.ylabel("Heart Disease (0 or 1)")
plt.show()

# Split the dataset into training and testing sets
rn.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

# Initialize and train the KNN model
model = KNeighborsClassifier(n_neighbors=5)  # You can adjust 'n_neighbors' as needed
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Print accuracy score
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")

# Optional: Visualize the results
plt.figure(figsize=(10, 6))
sns.countplot(x=Y_test, palette='Set2')
plt.title('Distribution of Heart Disease in Test Set')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()
