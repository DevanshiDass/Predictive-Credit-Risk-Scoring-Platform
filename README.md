# 🚀 Predictive Credit Risk Scoring Platform

This project is an **end-to-end Machine Learning–based Credit Risk Scoring Platform** that simulates how banks and fintech companies evaluate loan applications in the real world.

It combines **Data Science**, **Finance**, and **Explainable AI (XAI)** to make **transparent, risk-aware lending decisions**.

---

## 📌 Business Problem (Finance Context)

When a bank issues a loan, it faces **credit risk** — the risk that the borrower may fail to repay the loan.

Banks must answer:
- How likely is the borrower to default?
- If default occurs, how much money will be lost?
- Can this decision be explained to regulators?

This platform addresses these questions using **industry-standard credit risk metrics**:
- Probability of Default (PD)
- Loss Given Default (LGD)
- Exposure at Default (EAD)
- Expected Loss

---

## 🧠 End-to-End Solution Overview

For each loan applicant, the system:

1. Predicts **Probability of Default (PD)** using a calibrated ML model  
2. Converts PD into a **credit score (300–900)**  
3. Evaluates collateral to compute **Loan-to-Value (LTV)**  
4. Estimates **Loss Given Default (LGD)**  
5. Calculates **Expected Loss**  
6. Generates **explainable adverse action reasons** using SHAP  
7. Serves results via API and dashboard  

---

## 🏗️ System Architecture
```
User
    ↓
Streamlit Dashboard (UI)
    ↓
Flask API (Inference Layer)
    ↓
ML Model + Credit Risk Logic
    ↓
PD | Credit Score | LGD | Expected Loss | Reason Codes
```



---

## 📊 Dataset

- Based on the **German Credit Risk Dataset**
- Binary classification:
  - `1` → Default
  - `0` → No Default

### Feature groups:
- Credit history
- Loan amount and duration
- Employment stability
- Savings and income
- Installment burden
- Housing status and age
- Collateral details

---

## 🔬 Machine Learning Pipeline (Data Science)

### Data Preprocessing
- Categorical encoding
- Numerical scaling
- Implemented using **scikit-learn pipelines**
- Prevents data leakage

### Model Training
- Logistic / Linear classification model
- Chosen for:
  - Stable probability outputs
  - Interpretability
  - Compatibility with SHAP

### Probability Calibration
- **Isotonic Regression**
- Aligns predicted PD with real default frequencies
- Critical for financial decision-making

---

## 💰 Credit Risk & Finance Calculations

### Probability of Default (PD)

- Probability that a borrower will default
- Output of the calibrated ML model
- Value between 0 and 1

---

### Credit Score

- Human-friendly representation of risk
- Derived from PD
- Scale used: **300–900**
- Lower PD → Higher credit score

---

### Exposure at Default (EAD)

- Total amount exposed to loss if default occurs
- Approximated as the **loan amount**

---

### Loan-to-Value (LTV)

Loan-to-Value measures how much of the loan is covered by collateral.
LTV = Loan Amount / Collateral Value



- Higher LTV → Higher risk
- Worst-case LTV is used for conservative decisions

---

### Loss Given Default (LGD)

- Percentage of the loan expected to be lost if default occurs
- Depends on:
  - LTV
  - Collateral type
- Higher LTV → Higher LGD

---

### Expected Loss (Core Banking Metric)

Expected Loss represents the **average financial loss** the bank expects from a loan.
Expected Loss = PD × LGD × EAD


Used for:
- Risk provisioning
- Loan pricing
- Portfolio risk management

---

## 🔍 Explainable AI (XAI)

- Uses **SHAP (SHapley Additive exPlanations)**
- Explains how each feature affects PD
- SHAP values are converted into **adverse action reason codes**

This ensures:
- Regulatory compliance
- Business transparency
- Customer trust

---

## 🔴 High-Risk Applicant Scenario

This scenario demonstrates how the platform **identifies a high-risk borrower**.

### 📸 Applicant Input Profile
![High Risk Input](dashboard/high_risk_input.png)

![High Risk Input Continued](dashboard/high_risk_input_2.png)

### 📸 Risk Output & Expected Loss
![High Risk Output](dashboard/high_risk_expected.png)

## 🔴 High-Risk Applicant Scenario

### Observations
- High PD
- Low credit score
- High LTV
- High LGD due to weak collateral
- Large Expected Loss

### Generated Reasons
- High loan amount
- Long loan duration
- Poor credit history
- Low income and savings

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- SHAP
- Flask
- Streamlit
- Joblib

---

## 🎯 What This Project Demonstrates

- End-to-end data science pipeline
- Real-world credit risk modeling
- Explainable ML for regulated finance
- Production-style system design

---

## 👩‍💻 Author

Devanshi Das  
AI / ML | Data Science | Credit Risk Modeling

---

## 🚀 Future Enhancements

- Model monitoring and drift detection
- Policy-based approval logic
- Cloud deployment
- Real-time data ingestion
- Authentication and access control