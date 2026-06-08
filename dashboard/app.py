"""
Streamlit monitoring dashboard.
"""

import streamlit as st
import requests

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📊", layout="wide")

API_URL = "http://localhost:8000"
st.title("Customer Churn Prediction Dashboard")

try:
    stats = requests.get(f"{API_URL}/stats", timeout=3).json()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", stats.get("total_predictions", 0))
    c2.metric("Avg Churn Probability", f"{stats.get('avg_churn_probability', 0):.1%}")
    dist = stats.get("risk_distribution", {})
    c3.metric("High Risk", dist.get("HIGH", 0))
    c4.metric("Medium Risk", dist.get("MEDIUM", 0))
except Exception:
    st.warning("API offline — start uvicorn first.")

st.divider()
st.subheader("Single customer prediction")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure          = st.number_input("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly charges ($)", 0.0, 200.0, 65.0)
        contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        internet     = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        payment      = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with col3:
        gender     = st.selectbox("Gender", ["Male", "Female"])
        partner    = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        senior     = st.selectbox("Senior citizen", [0, 1])

    explain_toggle = st.checkbox("Show SHAP explanation", value=True)
    submitted = st.form_submit_button("Predict churn")

if submitted:
    payload = {
        "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": internet,
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": tech_support,
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": contract, "PaperlessBilling": "Yes",
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": float(tenure * monthly_charges),
    }
    with st.spinner("Predicting..."):
        res = requests.post(
            f"{API_URL}/predict",
            json=payload,
            params={"explain": explain_toggle},
            timeout=10,
        ).json()

    prob  = res["churn_probability"]
    tier  = res["risk_tier"]
    color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[tier]
    st.metric("Churn probability", f"{prob:.1%}")
    st.write(f"**Risk tier:** {color} {tier}")

    if explain_toggle and res.get("top_reasons"):
        st.subheader("Top contributing factors")
        for r in res["top_reasons"]:
            icon = "⬆️" if r["direction"].startswith("increases") else "⬇️"
            st.write(f"{icon} **{r['feature']}** — SHAP value: `{r['shap_value']}`")
