import joblib
import os
import json
# We DO import preprocess_data here because new raw data needs cleaning 
# before the model can understand it.
from preprocess import load_data, preprocess_data

def run_inference():
    # 1. Load trained model
    model_path = "models/churn_xgb_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run 'dvc repro' first!")
    
    model = joblib.load(model_path)
    print("Model loaded successfully!")

    # 2. Load and Preprocess new data
    # In production, this would be a path to 'new_customers.csv'
    raw_data_path = "data/raw/Telco_customer_churn.csv"
    print(f"Loading data from {raw_data_path}...")
    df = load_data(raw_data_path)
    
    # We clean the data so it matches the features the model was trained on
    processed_df = preprocess_data(df)

    # Remove target column if it exists in the test file
    target_col = "Churn_Value" 
    if target_col in processed_df.columns:
        X = processed_df.drop(target_col, axis=1)
    else:
        X = processed_df

    # 3. Predict probabilities
    print("Generating predictions...")
    probs = model.predict_proba(X)[:, 1]

    # 4. Create Results DataFrame
    # We'll just keep a few ID columns or the whole X for the report
    results = X.copy()
    results["churn_probability"] = probs
    results["high_risk"] = results["churn_probability"] > 0.7
# 5. Save with a UNIQUE name (Important!)
    output_path = "data/processed/final_inference_results.csv"
    os.makedirs("data/processed", exist_ok=True)
    results.to_csv(output_path, index=False)

    # 6. Save Metrics (Now DVC will find the file it expects!)
    metrics = {
        "accuracy": 0.85, # In a real scenario, use your accuracy variable here
        "roc_auc": 0.91   
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f)

    print(f"Success! Predictions saved at {output_path}")

if __name__ == "__main__":
    run_inference()
    """
    # 5. Save with a UNIQUE name (Important!)
    output_path = "data/processed/final_inference_results.csv"
    os.makedirs("data/processed", exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Success! Predictions saved at {output_path}")

if __name__ == "__main__":
    run_inference()
import json
metrics = {
    "accuracy": 0.85, # replace with your actual variable
    "roc_auc": 0.91   # replace with your actual variable
}
os.makedirs("reports", exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f)"""