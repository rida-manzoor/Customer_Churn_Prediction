# app/streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import os

# Path to model
MODEL_PATH = os.path.join("models", "churn_model.pkl")

@st.cache_resource
def load_model():
    """Load trained churn model pipeline."""
    return joblib.load(MODEL_PATH)

def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
    st.title("📊 Customer Churn Prediction App")

    st.markdown(
        """
        This app predicts whether a customer is likely to churn based on their attributes.  
        Enter customer details below and click **Predict**.
        """
    )

    # Load model
    model = load_model()

    # --- Input fields (adjust according to dataset features) ---
    st.header("Enter Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    # --- Prediction ---
    if st.button("🔮 Predict"):
        # Collect inputs
        input_data = pd.DataFrame(
            {
                "tenure": [tenure],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges],
                "gender": [gender],
                "SeniorCitizen": [senior_citizen],
                "Partner": [partner],
                "Dependents": [dependents],
                "PhoneService": [phone_service],
                "InternetService": [internet_service],
                "Contract": [contract],
                "PaymentMethod": [payment_method],
            }
        )

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Customer is likely to CHURN (probability: {proba:.2f})")
        else:
            st.success(f"✅ Customer is NOT likely to churn (probability: {proba:.2f})")

if __name__ == "__main__":
    main()
