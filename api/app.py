from flask import Flask, request, jsonify
import pandas as pd
import joblib
import shap

from common.scoring import pd_to_score
from common.reason_codes import generate_reason_codes
from common.ltv import compute_ltv
from common.lgd import estimate_lgd

# ============================
# LOAD MODEL
# ============================
cal_model = joblib.load("credit_model_v2.pkl")
base_pipeline = cal_model.calibrated_classifiers_[0].estimator

preprocess = base_pipeline.named_steps["preprocess"]
clf = base_pipeline.named_steps["model"]

FEATURES = [
    "status", "duration", "credit_history", "purpose",
    "amount", "savings", "employment_duration",
    "installment_rate", "housing", "age"
]

# ============================
# SHAP BACKGROUND DATA
# ============================
url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df_bg = pd.read_csv(url)

X_bg = df_bg[FEATURES].sample(100, random_state=42)
X_bg_transformed = preprocess.transform(X_bg)

explainer = shap.LinearExplainer(
    clf,
    X_bg_transformed,
    feature_perturbation="interventional"
)

# ============================
# FLASK APP
# ============================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "Credit Risk API running"

# ============================
# EXPLAIN ENDPOINT
# ============================
@app.route("/explain", methods=["POST"])
def explain():
    data = request.json
    X = pd.DataFrame([data])

    loan_amount = data.get("amount", 0)
    appraisals = data.get("property_appraisals", [])
    collateral_type = data.get("collateral_type", "secured")

    # PD + SCORE
    pd_default = cal_model.predict_proba(X)[:, 1][0]
    score = pd_to_score(pd_default)

    # LTV
    ltv_metrics = compute_ltv(loan_amount, appraisals)
    ltv_used = ltv_metrics["ltv_worst"] if ltv_metrics else None

    # LGD + EXPECTED LOSS
    lgd = estimate_lgd(ltv_used, collateral_type)
    expected_loss = pd_default * lgd * loan_amount

    # SHAP
    X_transformed = preprocess.transform(X)
    shap_values = explainer.shap_values(X_transformed)

    shap_df = pd.DataFrame({
        "feature": preprocess.get_feature_names_out(),
        "shap_value": shap_values[0]
    })

    shap_df["abs"] = shap_df["shap_value"].abs()
    top_features = shap_df.sort_values("abs", ascending=False).head(10)

    reasons = generate_reason_codes(top_features)

    return jsonify({
        "probability_default": round(pd_default, 4),
        "credit_score": score,
        "ltv_metrics": ltv_metrics,
        "lgd": round(lgd, 3),
        "expected_loss": round(expected_loss, 2),
        "adverse_action_reasons": reasons
    })

@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_version": "v2",
        "risk_framework": "PD × LGD × EAD",
        "lgd_type": "Rule-based",
        "ltv_method": "Multi-appraisal conservative",
        "explainability": "SHAP (LinearExplainer)",
        "calibration": "Isotonic"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
