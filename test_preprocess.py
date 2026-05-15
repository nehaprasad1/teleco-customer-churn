from src.preprocess import load_data, preprocess_data, split_features_target

# Load data
df = load_data(r"data\raw\Telco_customer_churn.csv")

print("Original shape:", df.shape)

# Preprocess
processed_df = preprocess_data(df)

print("\nProcessed shape:", processed_df.shape)
print("\nColumns:", processed_df.columns)

# Split
X, y = split_features_target(processed_df)

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("\nSample X:")
print(X.head())