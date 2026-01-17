import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------
# KS STATISTIC
# ---------------------------
def ks_stat(y_true, y_prob):
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df = df.sort_values("p")
    df["cum_good"] = ((df["y"] == 0).cumsum()) / (df["y"] == 0).sum()
    df["cum_bad"] = ((df["y"] == 1).cumsum()) / (df["y"] == 1).sum()
    return np.max(np.abs(df["cum_good"] - df["cum_bad"]))

# ---------------------------
# LOAD DATA
# ---------------------------
url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv(url)

target = "credit_risk"

features = [
    "status", "duration", "credit_history", "purpose",
    "amount", "savings", "employment_duration",
    "installment_rate", "housing", "age"
]

X = df[features]
y = df[target].map({1: 0, 0: 1})  # 1 = default

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# MODELS TO COMPARE
# ---------------------------
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
}

results = {}

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", clf)
    ])

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    ks = ks_stat(y_test, probs)

    results[name] = {
        "pipeline": pipe,
        "auc": auc,
        "ks": ks
    }

    print(f"{name.upper()}  AUC={auc:.4f}  KS={ks:.4f}")

# ---------------------------
# PICK BEST MODEL (BY KS)
# ---------------------------
best_name = max(results, key=lambda k: results[k]["ks"])
best_pipe = results[best_name]["pipeline"]

print("\nSelected model:", best_name)

# ---------------------------
# PROBABILITY CALIBRATION
# ---------------------------
calibrated_model = CalibratedClassifierCV(
    best_pipe,
    method="isotonic",
    cv=3
)

calibrated_model.fit(X_train, y_train)

cal_probs = calibrated_model.predict_proba(X_test)[:, 1]
cal_auc = roc_auc_score(y_test, cal_probs)
cal_ks = ks_stat(y_test, cal_probs)

print(f"CALIBRATED  AUC={cal_auc:.4f}  KS={cal_ks:.4f}")

# ---------------------------
# SAVE FINAL MODEL
# ---------------------------
joblib.dump(calibrated_model, "credit_model_v2.pkl")
joblib.dump(features, "model_features_v2.pkl")

print("Saved credit_model_v2.pkl")
