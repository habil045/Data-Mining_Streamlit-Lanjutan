import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

preproc = joblib.load('preproc_telco.joblib')
model = joblib.load('voting_telco.joblib')

st.title("📡 Telco Churn Prediction App")
st.write("Aplikasi ini memprediksi apakah pelanggan akan melakukan *churn* atau tidak menggunakan Voting Classifier (Random Forest + XGBoost).")

st.subheader("Input Data Pelanggan")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (Lama Berlangganan Bulan)", 0, 100, 1)
phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
paymentmethod = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
monthlycharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
totalcharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 100.0)

input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phoneservice,
    'MultipleLines': multiplelines,
    'InternetService': internetservice,
    'OnlineSecurity': onlinesecurity,
    'OnlineBackup': onlinebackup,
    'DeviceProtection': deviceprotection,
    'TechSupport': techsupport,
    'StreamingTV': streamingtv,
    'StreamingMovies': streamingmovies,
    'Contract': contract,
    'PaperlessBilling': paperlessbilling,
    'PaymentMethod': paymentmethod,
    'MonthlyCharges': monthlycharges,
    'TotalCharges': totalcharges
}])

if st.button("Predict Churn"):
    try:
        processed = preproc.transform(input_data)
        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]

        st.subheader("Hasil Prediksi")
        if pred == 1:
            st.error(f"⚠️ Pelanggan **berpotensi churn** (Probabilitas {prob:.2f})")
        else:
            st.success(f"✅ Pelanggan **tidak churn** (Probabilitas {prob:.2f})")

    except Exception as e:
        st.write("Error:", e)
