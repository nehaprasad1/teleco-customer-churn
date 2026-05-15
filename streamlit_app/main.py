import streamlit as st
import requests

st.set_page_config(page_title="Churn Predictor", page_icon="📊")

# Apply your Dark Academia aesthetic here via custom CSS if you like!
st.title("🍂 Customer Churn Analytics")
st.markdown("Predict customer behavior using our real-time XGBoost model.")

with st.form("prediction_form"):
    st.subheader("Customer Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", value=50.0)

    with col2:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        total_charges = st.number_input("Total Charges ($)", value=600.0)

    submit = st.form_submit_button("Predict Churn Risk")

if submit:
    # 1. Prepare payload (Match your FastAPI Pydantic schema)
    payload = {
        "Gender": gender,
        "Senior_Citizen": "No", # Defaulting for simplicity
        "Partner": "No",
        "Dependents": "No",
        "Tenure_Months": tenure,
        "Phone_Service": "Yes",
        "Multiple_Lines": "No",
        "Internet_Service": internet,
        "Online_Security": "No",
        "Online_Backup": "No",
        "Device_Protection": "No",
        "Tech_Support": "No",
        "Streaming_TV": "No",
        "Streaming_Movies": "No",
        "Contract": contract,
        "Paperless_Billing": "Yes",
        "Payment_Method": payment,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges
    }

    # 2. Call your FastAPI backend
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        st.divider()
        prob = result["churn_probability"]
        
        if prob > 0.7:
            st.error(f"High Risk! Churn Probability: {prob:.2%}")
        elif prob > 0.4:
            st.warning(f"Moderate Risk. Churn Probability: {prob:.2%}")
        else:
            st.success(f"Low Risk. Churn Probability: {prob:.2%}")

    except Exception as e:
        st.error("Could not connect to FastAPI. Is the server running?")