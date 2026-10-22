"""
app.py
------
Streamlit front-end for the AI Loan Approval & Risk Prediction System.
Loads the model trained by train_model.py and serves real-time
predictions from user-entered applicant details.

Run:
  streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "model"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    metrics = {}
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return model, encoders, scaler, target_encoder, feature_cols, metrics


CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

if not os.path.exists(os.path.join(MODEL_DIR, "loan_model.pkl")):
    st.error(
        "No trained model found. Run these first in your terminal:\n\n"
        "```\npython generate_data.py\npython train_model.py\n```\n"
        "then restart this app."
    )
    st.stop()

model, encoders, scaler, target_encoder, feature_cols, metrics = load_artifacts()

st.title("🏦 AI Loan Approval & Risk Prediction System")
st.caption("Logistic Regression model · trained on applicant financial & demographic data")

with st.sidebar:
    st.header("Model performance")
    if metrics:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        c1, c2 = st.columns(2)
        c1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        c2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    st.divider()
    st.caption(
        "This is a personal/educational project using a synthetic dataset. "
        "Predictions are illustrative only and not financial advice."
    )

st.subheader("Applicant details")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Monthly Income", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Monthly Income", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=120, step=5)
    loan_term = st.selectbox("Loan Term (days)", [360, 240, 180, 120, 60], index=0)
    credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])

if st.button("Predict Loan Approval", type="primary", use_container_width=True):
    raw = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Property_Area": property_area,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1 if credit_history.startswith("Good") else 0,
    }

    row = pd.DataFrame([raw])

    # apply the SAME encoders/scaler fit during training
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        # guard against a category the encoder has never seen
        val = raw[col] if raw[col] in le.classes_ else le.classes_[0]
        row[col] = le.transform([val])

    row[NUMERIC_COLS] = scaler.transform(row[NUMERIC_COLS])

    X_input = row[feature_cols]
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    label = target_encoder.inverse_transform([pred])[0]
    approve_idx = list(target_encoder.classes_).index("Y")

    st.divider()
    if label == "Y":
        st.success(f"✅ Loan likely to be APPROVED  ·  confidence {proba[approve_idx] * 100:.1f}%")
    else:
        st.error(f"❌ Loan likely to be REJECTED  ·  confidence {(1 - proba[approve_idx]) * 100:.1f}%")

    st.progress(float(proba[approve_idx]), text=f"Approval probability: {proba[approve_idx] * 100:.1f}%")
