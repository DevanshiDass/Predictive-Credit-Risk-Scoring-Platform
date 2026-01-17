import pandas as pd
import joblib
from common.scoring import pd_to_score

# Load model
model = joblib.load("credit_model.pkl")

# Test sample
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

df = pd.DataFrame([sample])

# Predict PD
pd_default = model.predict_proba(df)[:, 1][0]

# Convert to score
score = pd_to_score(pd_default)

print("Probability of Default:", pd_default)
print("Credit Score:", score)
