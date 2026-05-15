import os
import joblib
import numpy as np
import random
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from preprocess import load_data, preprocess_data

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)
random.seed(42)

# -------------------------
# Load dataset
# -------------------------
df = load_data("data/raw/Telco_customer_churn.csv")

# -------------------------
# Preprocess dataset
# -------------------------
df = preprocess_data(df)

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
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# XGBoost Model
# -------------------------
# Note: early_stopping_rounds and eval_metric are now defined here 
# to comply with XGBoost 2.0+ API standards.
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
# We pass eval_set here, but early_stopping is handled by the constructor above.
clf_xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# -------------------------
# Predictions
# -------------------------
y_pred = clf_xgb.predict(X_test)
y_prob = clf_xgb.predict_proba(X_test)[:, 1]

# -------------------------
# Evaluation metrics
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n========================")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print("========================\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------
# Save model
# -------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(clf_xgb, "models/churn_xgb_model.pkl")

print("\nModel saved successfully at models/churn_xgb_model.pkl")