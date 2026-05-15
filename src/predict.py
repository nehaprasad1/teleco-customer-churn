# src/predict.py

import joblib
import pandas as pd
import os

from preprocess import load_data, preprocess_data

# -------------------------
# Load trained model
# -------------------------
# Path updated to match the name used in train.py
model_path = "models/churn_xgb_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Could not find the model at {model_path}. Did you run train.py first?")

model = joblib.load(model_path)
print("Model loaded successfully!")

# -------------------------
# Load new data
# -------------------------
df = load_data("data/raw/Telco_customer_churn.csv")

# -------------------------
# Preprocess data
# -------------------------
df = preprocess_data(df)

# Remove target column if present
# Ensure this matches the target name used in your preprocessing
target_col = "Churn_Value" 
if target_col in df.columns:
    X = df.drop(target_col, axis=1)
else:
    X = df

# -------------------------
# Predict probabilities
# -------------------------
predictions = model.predict_proba(X)[:, 1]

# Add predictions to dataframe
results = X.copy()
results["churn_probability"] = predictions

# High-risk customers
results["high_risk"] = results["churn_probability"] > 0.7

# -------------------------
# Save predictions
# -------------------------
os.makedirs("data/processed", exist_ok=True)
results.to_csv(
    "data/processed/churn_predictions.csv",
    index=False
)

print("Predictions saved successfully at data/processed/churn_predictions.csv!")