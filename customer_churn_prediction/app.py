import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------
# 1. LOAD THE MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model_data = joblib.load('churn_model_final.pkl')
    return model_data['pipeline'], model_data['threshold']

pipeline, threshold = load_model()

# ---------------------------
# 2. DEFINE INPUT FIELDS (matching your training data)
# ---------------------------
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Predictor")
st.markdown("Enter customer details below to estimate churn probability.")

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0, step=0.1)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=1.0)
    SeniorCitizen = st.selectbox("Senior Citizen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    Partner = st.selectbox("Partner", options=["Yes", "No"])
    Dependents = st.selectbox("Dependents", options=["Yes", "No"])

with col2:
    PhoneService = st.selectbox("Phone Service", options=["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", options=["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", options=["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", options=["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", options=["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", options=["Yes", "No", "No internet service"])

# Second row for remaining features
st.divider()
col3, col4 = st.columns(2)

with col3:
    StreamingTV = st.selectbox("Streaming TV", options=["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", options=["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", options=["Yes", "No"])

with col4:
    PaymentMethod = st.selectbox("Payment Method", options=[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

# ---------------------------
# 3. PREDICT BUTTON
# ---------------------------
if st.button("🔮 Predict Churn", type="primary"):
    # Create a DataFrame with the exact column order as training (X_train)
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }
    df_input = pd.DataFrame(input_data)

    # Predict probability
    proba = pipeline.predict_proba(df_input)[0][1]  # Probability of churn = 1
    prediction = int(proba >= threshold)

    # Display results
    st.divider()
    st.subheader("🧾 Prediction Result")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.metric("Churn Probability", f"{proba:.2%}")
    with col_res2:
        if prediction == 1:
            st.error("🚨 **Customer is likely to churn**")
        else:
            st.success("✅ **Customer is likely to stay**")

    # Optional: Show a gauge or progress bar for probability
    st.progress(proba, text=f"Confidence: {proba:.1%}")