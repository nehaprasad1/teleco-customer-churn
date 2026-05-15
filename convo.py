import pandas as pd

df = pd.read_excel("Telco_customer_churn.xlsx")

df.to_csv(
    "data/raw/Telco_customer_churn.csv",
    index=False
)

print("CSV created successfully!")