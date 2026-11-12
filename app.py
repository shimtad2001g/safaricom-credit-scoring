import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Safaricom Credit Scoring Demo", page_icon="💳", layout="centered")

st.title("💳 Credit Scoring AI Prototype")
st.markdown("""
This is a demonstration of an end-to-end Machine Learning pipeline built for **Credit Scoring**. 
It simulates how Telco/GSM data can be used to predict loan default risk.
""")

# Auto-train model if it doesn't exist (for cloud deployment)
@st.cache_resource
def load_or_train_model():
    model_path = 'credit_model.pkl'
    
    if not os.path.exists(model_path):
        st.info("🔄 First run detected. Training model on the fly (this takes ~5 seconds)...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 10000
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'monthly_income': np.random.randint(1000, 50000, n_samples),
            'loan_amount': np.random.randint(500, 20000, n_samples),
            'tenure_months': np.random.randint(1, 120, n_samples),
            'num_late_payments': np.random.randint(0, 10, n_samples),
            'data_usage_gb': np.random.uniform(1.0, 50.0, n_samples)
        }
        df = pd.DataFrame(data)
        
        risk_score = (df['num_late_payments'] * 0.5) + (df['loan_amount'] / df['monthly_income']) - (df['tenure_months'] * 0.01)
        df['default'] = (risk_score > 1.5).astype(int)
        
        X = df.drop('default', axis=1)
        y = df['default']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model for future runs
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
    else:
        with open(model_path, 'rb') as f:
            return pickle.load(f)

try:
    model = load_or_train_model()
    
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
        input_df = pd.DataFrame([{
            'age': age,
            'monthly_income': income,
            'loan_amount': loan,
            'tenure_months': tenure,
            'num_late_payments': late,
            'data_usage_gb': data_usage
        }])
        
        prediction = model.predict(input_df)[0]
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
    st.error(f"Error: {e}")

st.markdown("---")
st.caption("Developed by Shimeles Tadesse Zelelew | Proof of Work for Safaricom ML Engineer Role")

# Forced rebuild to clear Streamlit Cloud routing cache - 11/12/2026 15:21:24
