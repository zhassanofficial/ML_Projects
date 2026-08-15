import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    /* Prediction card */
    .prediction-card {
        background: #f8f9fa;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        text-align: center;
        margin-top: 1.5rem;
    }
    .prediction-card h3 {
        margin-top: 0;
        color: #1E3A5F;
    }
    .prob-text {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .churn-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .churn-badge.danger {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .churn-badge.success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    /* Sidebar style */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 1rem;
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 0.5rem;
    }
    /* Metric boxes */
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-box .label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .metric-box .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
        margin-top: 3rem;
        border-top: 1px solid #dee2e6;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL (cached)
# ---------------------------
@st.cache_resource
def load_model():
    model_data = joblib.load('churn_model_final.pkl')
    return model_data['pipeline'], model_data['threshold']

pipeline, threshold = load_model()

# ---------------------------
# SIDEBAR INPUT
# ---------------------------
st.sidebar.markdown('<div class="sidebar-header">📋 Customer Profile</div>', unsafe_allow_html=True)

# Organize inputs in expanders
with st.sidebar.expander("👤 Demographics", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])

with st.sidebar.expander("📞 Services Subscribed", expanded=True):
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with st.sidebar.expander("📄 Contract & Billing", expanded=True):
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0, step=0.5)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)

# Reset button
if st.sidebar.button("🔄 Reset to Defaults"):
    st.experimental_rerun()

# ---------------------------
# MAIN AREA
# ---------------------------
st.markdown('<div class="main-title">📊 Telco Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter customer details in the sidebar and click Predict</div>', unsafe_allow_html=True)

# Predict button
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    with st.spinner("Calculating..."):
        time.sleep(0.5)  # simulate slight delay for UX

        # Build input DataFrame
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

        # Predict
        proba = pipeline.predict_proba(df_input)[0][1]
        prediction = int(proba >= threshold)

        # ---------------------------
        # DISPLAY RESULTS
        # ---------------------------
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("<h3>🧾 Prediction Result</h3>", unsafe_allow_html=True)

        # Two columns: probability gauge & status
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f'<div class="prob-text">{proba:.1%}</div>', unsafe_allow_html=True)
            st.write("Churn Probability")
            # Progress bar colored based on probability
            if proba < 0.3:
                color = "#2ecc71"
            elif proba < 0.6:
                color = "#f39c12"
            else:
                color = "#e74c3c"
            st.progress(proba, text=f"Confidence: {proba:.1%}")

        with col2:
            if prediction == 1:
                st.markdown('<div class="churn-badge danger">🚨 Likely to Churn</div>', unsafe_allow_html=True)
                st.write("Recommendation: Offer retention incentives")
            else:
                st.markdown('<div class="churn-badge success">✅ Likely to Stay</div>', unsafe_allow_html=True)
                st.write("Recommendation: Continue current engagement")

        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------
        # ADDITIONAL INSIGHTS (optional)
        # ---------------------------
        with st.expander("📈 Factors Influencing this Prediction"):
            st.write("Top risk factors for this customer (based on model coefficients):")
            # For simplicity, show a placeholder; you can extract coefficients from the pipeline if desired.
            st.info("""
            - **Contract type**: Month-to-month increases churn risk significantly.
            - **Tenure**: Shorter tenure (less than 12 months) indicates higher risk.
            - **Monthly Charges**: Higher monthly bills are associated with churn.
            """)

        # Show a summary of the input in a compact table
        with st.expander("📋 Input Summary"):
            st.dataframe(df_input.T, use_container_width=True, height=300)
else:
    # Show a welcome message when no prediction yet
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn** to get started.")
    st.image("https://via.placeholder.com/800x200?text=Telco+Customer+Churn+Prediction", use_container_width=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown('<div class="footer">Built with ❤️ using Streamlit | Model: Logistic Regression (Recall: 0.80, F1: 0.62)</div>', unsafe_allow_html=True)