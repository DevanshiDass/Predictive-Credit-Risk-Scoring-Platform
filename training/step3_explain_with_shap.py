import shap
import joblib
import pandas as pd
import numpy as np

# ============================
# LOAD CALIBRATED MODEL
# ============================
cal_model = joblib.load("credit_model_v2.pkl")
base_pipeline = cal_model.calibrated_classifiers_[0].estimator

preprocess = base_pipeline.named_steps["preprocess"]
clf = base_pipeline.named_steps["model"]

# ============================
# LOAD DATA FOR BACKGROUND
# ============================
url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv(url)

features = [
    "status", "duration", "credit_history", "purpose",
    "amount", "savings", "employment_duration",
    "installment_rate", "housing", "age"
]

X_background = df[features].sample(100, random_state=42)

X_background_transformed = preprocess.transform(X_background)

# ============================
# SAMPLE APPLICANT
# ============================
sample = {
    "status": "< 100 DM",
    "duration": 12,
    "credit_history": "critical account/other credits existing",
    "purpose": "domestic appliances",
    "amount": 2000,
    "savings": "< 100 DM",
    "employment_duration": "1 <= ... < 4 years",
    "installment_rate": 2,
    "housing": "own",
    "age": 30
}

X = pd.DataFrame([sample])
X_transformed = preprocess.transform(X)

feature_names = preprocess.get_feature_names_out()

# ============================
# SHAP LINEAR EXPLAINER (CORRECT)
# ============================
explainer = shap.LinearExplainer(
    clf,
    X_background_transformed,
    feature_perturbation="interventional"
)

shap_values = explainer.shap_values(X_transformed)

# ============================
# CONTRIBUTIONS
# ============================
contrib = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values[0]
})

contrib["abs"] = contrib["shap_value"].abs()

top_reasons = contrib.sort_values("abs", ascending=False).head(10)

# ============================
# OUTPUT
# ============================
print("\nTOP RISK DRIVERS (SHAP):\n")
print(top_reasons)
from common.reason_codes import generate_reason_codes

reasons = generate_reason_codes(top_reasons)

print("\nADVERSE ACTION REASONS:")
for r in reasons:
    print("-", r)
