# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("diabetes.csv")

X = data.drop(columns="Outcome", axis=1)
Y = data["Outcome"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
model = SVC(kernel="rbf", class_weight="balanced", random_state=42)
model.fit(X_train, Y_train)

# Evaluate
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

print(f"✅ Model trained successfully!")
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save model and scaler
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("🧠 Model and scaler saved successfully!")
