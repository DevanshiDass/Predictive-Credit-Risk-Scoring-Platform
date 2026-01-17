import streamlit as st
import requests

st.set_page_config(
    page_title="Predictive Credit Risk Scoring Platform",
    layout="wide",
    page_icon="🏦"
)

API_BASE = "http://127.0.0.1:5000"

# -------------------------
# API helpers
# -------------------------
def call_explain(payload):
    r = requests.post(f"{API_BASE}/explain", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def get_model_info():
    r = requests.get(f"{API_BASE}/model-info", timeout=5)
    r.raise_for_status()
    return r.json()

# -------------------------
# Finance helpers
# -------------------------
def compute_emi(P, annual_rate_percent, N_months):
    if annual_rate_percent <= 0:
        return P / N_months
    r = annual_rate_percent / 100 / 12
    return (P * r * (1 + r)**N_months) / ((1 + r)**N_months - 1)

def dti_from_income(emi, monthly_income):
    if monthly_income <= 0:
        return None
    return (emi / monthly_income) * 100

def interest_suggestion(score):
    if score >= 750:
        return 8.0
    elif score >= 650:
        return 11.0
    elif score >= 550:
        return 16.0
    else:
        return 24.0

# =========================
# UI
# =========================
st.title("🏦 Predictive Credit Risk Scoring Platform")
st.caption("PD + LTV + LGD + Explainability")

left, right = st.columns([2.3, 1])

with left:
    st.subheader("Applicant Details")

    status = st.selectbox("Account Status", ["< 100 DM", "0 <= ... < 200 DM", "no checking account"])
    duration = st.number_input("Loan Duration (months)", 1, 120, 12)
    credit_history = st.selectbox(
        "Credit History",
        [
            "critical account/other credits existing",
            "existing credits paid back duly till now",
            "delay in paying off in the past",
            "no credits/all credits paid back duly",
        ],
    )
    purpose = st.selectbox("Purpose", ["domestic appliances", "radio/television", "retraining", "car (new)"])
    amount = st.number_input("Loan Amount (₹)", 100, 2_000_000, 2000, step=100)
    savings = st.selectbox("Savings", ["< 100 DM", "100 <= ... < 500 DM", "unknown"])
    employment_duration = st.selectbox(
        "Employment Duration",
        ["unemployed", "< 1 year", "1 <= ... < 4 years", "4 <= ... < 7 years"],
    )
    installment_rate = st.number_input("Installment Rate", 1, 10, 2)
    housing = st.selectbox("Housing", ["own", "rent", "for free"])
    age = st.number_input("Age", 18, 100, 30)

    st.subheader("Collateral Appraisals")
    a1 = st.number_input("Appraisal 1 (₹)", 0, 10_000_000, 5_000_000, step=50_000)
    a2 = st.number_input("Appraisal 2 (₹)", 0, 10_000_000, 4_800_000, step=50_000)
    a3 = st.number_input("Appraisal 3 (₹)", 0, 10_000_000, 5_200_000, step=50_000)
    collateral_type = st.selectbox("Collateral Type", ["secured", "unsecured"])

    st.subheader("Income")
    monthly_income = st.number_input("Monthly Income (₹)", 0, 2_000_000, 30000, step=500)

    if st.button("Evaluate Applicant"):
        payload = {
            "status": status,
            "duration": int(duration),
            "credit_history": credit_history,
            "purpose": purpose,
            "amount": float(amount),
            "savings": savings,
            "employment_duration": employment_duration,
            "installment_rate": int(installment_rate),
            "housing": housing,
            "age": int(age),
            "property_appraisals": [a1, a2, a3],
            "collateral_type": collateral_type,
        }

        resp = call_explain(payload)

        pd_default = resp["probability_default"]
        score = resp["credit_score"]

        rate = interest_suggestion(score)
        emi = compute_emi(amount, rate, duration)
        dti = dti_from_income(emi, monthly_income)

        st.markdown("---")
        st.subheader("Risk Output")

        st.metric("Probability of Default", f"{pd_default:.3f}")
        st.metric("Credit Score", f"{score} / 900")
        st.write(f"**Interest Rate:** {rate:.2f}%")
        st.write(f"**EMI:** ₹{emi:,.2f}")
        st.write(f"**LGD:** {resp['lgd']}")
        st.write(f"**Expected Loss:** ₹{resp['expected_loss']:,.2f}")

        if resp["ltv_metrics"]:
            st.subheader("LTV Metrics")
            st.json(resp["ltv_metrics"])

        st.subheader("Why this decision?")
        for r in resp["adverse_action_reasons"]:
            st.warning(r)

with right:
    st.subheader("Model Info")
    st.json(get_model_info())
