import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
import random as rn

# Load dataset
data = pd.read_csv('heart.csv')

# Print dataset columns
print("Columns in the dataset:", data.columns)

# Define features and target variable
X = data[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]]
Y = data["HeartDisease"]  # Binary target variable (0 or 1)

# Scatter plot to visualize the relationship between features and target variable
sns.scatterplot(x=data["Age"], y=data["HeartDisease"], color="blue", label="Age vs Heart Disease")
plt.title("Age vs Heart Disease")
plt.xlabel("Age")
plt.ylabel("Heart Disease (0 or 1)")
plt.show()

sns.scatterplot(x=data["RestingBP"], y=data["HeartDisease"], color="green", label="RestingBP vs Heart Disease")
plt.title("RestingBP vs Heart Disease")
plt.xlabel("Resting BP")
plt.ylabel("Heart Disease (0 or 1)")
plt.show()

sns.scatterplot(x=data["Cholesterol"], y=data["HeartDisease"], color="red", label="Cholesterol vs Heart Disease")
plt.title("Cholesterol vs Heart Disease")
plt.xlabel("Cholesterol")
plt.ylabel("Heart Disease (0 or 1)")
plt.show()

# Split the dataset into training and testing sets
rn.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=200)  # Increase iterations if needed
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)
Y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# Print accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print coefficients and intercept for feature importance
print("Feature Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"  {feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()
