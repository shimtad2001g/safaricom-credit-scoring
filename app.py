import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Safaricom Credit Scoring Demo", page_icon="💳", layout="centered")

st.title("💳 Credit Scoring AI Prototype")
st.markdown("""
This is a demonstration of an end-to-end Machine Learning pipeline built for **Credit Scoring**. 
It simulates how Telco/GSM data can be used to predict loan default risk.
""")

# Load model
@st.cache_resource
def load_model():
    with open('credit_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    
    st.subheader("Customer Data Input")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Monthly Income (ETB)", min_value=1000, value=15000)
        loan = st.number_input("Requested Loan Amount (ETB)", min_value=500, value=5000)
        
    with col2:
        tenure = st.number_input("Sim Card Tenure (Months)", min_value=1, value=24)
        late = st.number_input("Number of Late Payments", min_value=0, max_value=20, value=0)
        data_usage = st.number_input("Monthly Data Usage (GB)", min_value=1.0, value=10.0)

    if st.button("Calculate Credit Risk", type="primary"):
        # Prepare input
        input_df = pd.DataFrame([{
            'age': age,
            'monthly_income': income,
            'loan_amount': loan,
            'tenure_months': tenure,
            'num_late_payments': late,
            'data_usage_gb': data_usage
        }])
        
        # Predict
        prediction = model.predict(input_data)[0] if 'input_data' in locals() else model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        st.markdown("---")
        st.subheader("Results")
        
        if prediction == 1:
            st.error(f"🔴 **HIGH RISK**: Probability of Default is **{probability*100:.1f}%**")
            st.markdown("**Recommendation:** Reject or require collateral.")
        else:
            st.success(f"🟢 **LOW RISK**: Probability of Default is **{probability*100:.1f}%**")
            st.markdown("**Recommendation:** Approve loan.")
            
except Exception as e:
    st.error(f"Error loading model. Please run `python train_model.py` first. Details: {e}")

st.markdown("---")
st.caption("Developed by Shimeles Tadesse Zelelew | Proof of Work for Safaricom ML Engineer Role")
