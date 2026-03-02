// model training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

print("Starting Loan Model Training...")

# Load dataset
df = pd.read_csv("train.csv")

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

# Print columns to check
print("Columns in dataset:")
print(df.columns)

# Drop Loan_ID if exists
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Find correct target column automatically
target_column = None
for col in df.columns:
    if "loan" in col.lower() and "status" in col.lower():
        target_column = col
        break

if target_column is None:
    print("Loan Status column not found. Check column names above.")
    exit()

print("Target column detected:", target_column)

# Convert target
df[target_column] = df[target_column].map({"Y": 1, "N": 0})

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("loan_model.pkl", "wb"))

print("Model saved successfully!")



//app.py
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
