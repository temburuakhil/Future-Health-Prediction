import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
data = pd.read_csv('updated_health_data.csv')

# Check and fill missing values
if data.isnull().sum().any():
    print("Missing values detected. Filling missing values with the mean...")
    data.fillna(data.mean(), inplace=True)

# Prepare features (X) and labels (y)
X = data[['Age', 'Gender', 'BMI', 'BloodPressure', 'CholesterolLevel', 'Smoking', 'Diabetes', 'AlcoholConsumption']].values
y = data[['HeartDisease', 'Diabetes', 'Stroke', 'LungDisease', 'LiverDisease']].apply(lambda x: (x > 0).astype(int)).values


# Convert 'Gender' feature to binary encoding (Male=1, Female=0)
X[:, 1] = np.where(X[:, 1] == 'Male', 1, 0)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Random Forest model for each target
n_estimators = 100  # Number of trees in the forest
forest_models = {}
for i, disease in enumerate(['HeartDisease', 'Diabetes', 'Stroke', 'LungDisease', 'LiverDisease']):
    print(f"Training Random Forest model for {disease}...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train[:, i])
    forest_models[disease] = model

# Evaluate each model and calculate accuracy on the test set
for disease, model in forest_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test[:, list(forest_models.keys()).index(disease)], y_pred)
    print(f"Test Accuracy for {disease}: {accuracy * 100:.2f}%")

# Example Prediction
new_person = scaler.transform([[65, 1, 28, 130, 220, 1, 0, 1]])
predictions = {}
for disease, model in forest_models.items():
    prob = model.predict_proba(new_person)[:, 1][0]  # Get probability of the positive class
    predictions[disease] = prob * 100
    print(f"Risk of {disease}: {predictions[disease]:.2f}%")

# Save models and scaler
for disease, model in forest_models.items():
    joblib.dump(model, f'{disease}_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Models and scaler saved successfully.")