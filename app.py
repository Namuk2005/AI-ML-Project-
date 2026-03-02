import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.set_page_config(page_title="Loan Approval System", page_icon="🏦", layout="wide")

st.title("🏦 Loan Approval Prediction System")
st.markdown("### AI/ML Based Loan Eligibility Checker")

# Load model
try:
    model = pickle.load(open("loan_model.pkl", "rb"))
    st.success("Model loaded successfully!")
except:
    st.error("Model not found!")
    st.stop()

st.markdown("---")

# Create 2 columns layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    app_income = st.number_input("Applicant Income", min_value=0)
    coapp_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (in days)", min_value=0)
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")

# Convert values
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

if st.button("🔍 Predict Loan Status"):

    input_data = np.array([[gender, married, dependents, education,
                            self_employed, app_income, coapp_income,
                            loan_amount, loan_term, credit_history,
                            property_area]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.info(f"Approval Probability: {probability[0][1]*100:.2f}%")

st.markdown("---")
st.caption("Developed using Random Forest & Streamlit | AI/ML Project")
