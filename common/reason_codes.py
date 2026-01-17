REASON_MAP = {
    "num__amount": "High loan amount",
    "num__duration": "Long loan duration",
    "num__installment_rate": "High installment burden",
    "num__age": "Applicant age risk factor",

    "cat__savings_< 100 DM": "Low savings balance",
    "cat__status_no checking account": "No checking account history",
    "cat__credit_history_critical account/other credits": "Adverse credit history",
    "cat__credit_history_delay in paying off": "Past payment delays",
    "cat__employment_duration_unemployed": "Unstable employment",
    "cat__employment_duration_1 <= ... < 4 years": "Short employment history",
    "cat__housing_rent": "Rented housing",
}

def generate_reason_codes(shap_df, top_n=3):
    reasons = []

    for _, row in shap_df.iterrows():
        if row["shap_value"] > 0 and row["feature"] in REASON_MAP:
            reasons.append(REASON_MAP[row["feature"]])
        if len(reasons) == top_n:
            break

    return reasons
