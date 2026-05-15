import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data"""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean + preprocess churn dataset in a production-safe way
    """

    # -----------------------------
    # Step 1: Safe column dropping
    # -----------------------------
    columns_to_drop = [
        'CustomerID',
        'Churn Label',
        'Churn Score',
        'CLTV',
        'Churn Reason',
        'Count',
        'Country',
        'State',
        'Lat Long',
        'City',        # high cardinality
        'Zip_Code'     # not meaningful for ML
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')

    # -----------------------------
    # Step 2: Fix numeric columns
    # -----------------------------
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

    # -----------------------------
    # Step 3: Clean column names
    # -----------------------------
    df.columns = df.columns.str.replace(' ', '_')

    # -----------------------------
    # Step 4: Identify target column
    # -----------------------------
    target_col = "Churn_Value"

    # -----------------------------
    # Step 5: Split features & target
    # -----------------------------
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # -----------------------------
    # Step 6: One-hot encode categorical features
    # -----------------------------
    categorical_cols = X.select_dtypes(include=["object"]).columns

    X = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    # -----------------------------
    # Step 7: Reattach target
    # -----------------------------
    processed_df = pd.concat([X, y], axis=1)

    return processed_df


def split_features_target(df: pd.DataFrame):
    """Utility for training"""
    X = df.drop(columns=["Churn_Value"])
    y = df["Churn_Value"]
    return X, y