from fastapi import FastAPI
import joblib
import pandas as pd
import os
import sys

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import preprocess_data
from app.schema import ChurnInput

app = FastAPI(title="ibm customer Churn API")

# Load model and the list of columns the model was trained on
MODEL_PATH = "models/churn_xgb_model.pkl"
model = joblib.load(MODEL_PATH)
model_features = model.get_booster().feature_names # Get the exact columns XGBoost wants
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "project": "Customer Churn Pipeline",
        "api_docs": "http://127.0.0.1:8000/docs"
    }
@app.post("/predict")
def predict(payload: ChurnInput):
    # 1. Convert input to DataFrame
    raw_df = pd.DataFrame([payload.dict()])
    
    # 2. Run your existing preprocessing logic
    processed_df = preprocess_data(raw_df)
    
    # 3. ALIGNMENT: Add missing columns that get_dummies might have skipped
    # This ensures your API doesn't crash if a category is missing
    for col in model_features:
        if col not in processed_df.columns:
            processed_df[col] = 0
            
    # 4. Ensure column order matches the model exactly
    final_df = processed_df[model_features]
    
    # 5. Predict
    prob = model.predict_proba(final_df)[:, 1][0]
    
    return {
        "churn_probability": round(float(prob), 4),
        "prediction": "Churn" if prob > 0.5 else "Stay",
        "high_risk_alert": prob > 0.75
    }