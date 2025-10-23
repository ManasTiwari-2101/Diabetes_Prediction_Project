
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_diabetes(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_array)
    prediction = model.predict(std_data)
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

# Example input
sample_input = (6, 148, 72, 35, 0, 33.6, 0.627, 50)
result = predict_diabetes(sample_input)

print("Prediction for sample input:", result)
