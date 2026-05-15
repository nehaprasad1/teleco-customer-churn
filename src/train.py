import os
import joblib
import numpy as np
import random
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)
random.seed(42)

def run_training():
    # -------------------------
    # Load Preprocessed dataset
    # -------------------------
    # We load the output from the PREPROCESS stage
    input_path = "data/processed/churn_predictions.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Processed data not found at {input_path}. Run 'dvc repro' to generate it.")
        
    df = pd.read_csv(input_path)

    # Note: We NO LONGER call preprocess_data(df) here. 
    # The data is already clean!

    # -------------------------
    # Features & target
    # -------------------------
    target_col = "Churn_Value"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -------------------------
    # Train-test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # XGBoost Model
    # -------------------------
    clf_xgb = xgb.XGBClassifier(
        random_state=42,
        objective='binary:logistic',
        gamma=0.25,
        learning_rate=0.1,
        reg_lambda=10,
        scale_pos_weight=3,
        subsample=0.9,
        colsample_bytree=0.5,
        eval_metric='aucpr',
        n_estimators=1000,
        early_stopping_rounds=10
    )

    # -------------------------
    # TRAINING
    # -------------------------
    print("Starting training...")
    clf_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred = clf_xgb.predict(X_test)
    y_prob = clf_xgb.predict_proba(X_test)[:, 1]

    print("\n========================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("========================\n")

    # -------------------------
    # Save model
    # -------------------------
    os.makedirs("models", exist_ok=True)
    model_path = "models/churn_xgb_model.pkl"
    joblib.dump(clf_xgb, model_path)
    print(f"Model saved successfully at {model_path}")

if __name__ == "__main__":
    run_training()