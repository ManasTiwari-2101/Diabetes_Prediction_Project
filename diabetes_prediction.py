# diabetes_prediction.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# 2. Separate features and labels
X = diabetes_dataset.drop(columns="Outcome", axis=1)
Y = diabetes_dataset["Outcome"]

# 3. Split into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

# 4. Standardize (fit only on training)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train the model (SVM with class balancing)
classifier = SVC(kernel="rbf", class_weight="balanced", random_state=42)
classifier.fit(X_train, Y_train)

# 6. Evaluate accuracy
train_acc = accuracy_score(classifier.predict(X_train), Y_train)
test_acc = accuracy_score(classifier.predict(X_test), Y_test)

print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# 7. Predictive system
def predict_diabetes(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_array)
    prediction = classifier.predict(std_data)
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

# Example input (try changing these values)
sample_input = (6, 148, 72, 35, 0, 33.6, 0.627, 50)
result = predict_diabetes(sample_input)
print("Prediction for sample input:", result)
