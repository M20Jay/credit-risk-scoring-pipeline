# api/main.py
# Credit Risk Scoring API
# Three endpoints: health, assess, segment

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os

# Load artifacts
model = joblib.load('models/credit_risk_model.pkl')
threshold = joblib.load('models/optimal_threshold.pkl')
imputer = joblib.load('models/imputer.pkl')

# Feature list — must match train.py order
FEATURES = [
    'loan_amnt', 'int_rate', 'installment',
    'annual_inc', 'dti_clean', 'delinq_2yrs',
    'inq_last_6mths', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc',
    'credit_history_years', 'loan_to_income',
    'high_dti', 'high_revol_util', 'has_delinquency'
]

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Scores loan applicants on credit risk, propensity and RFM segment",
    version="1.0.0"
)

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost Credit Risk",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Assessing endpoint
@app.post("/assess")
def assess(application: LoanApplication):
    try:
        # Feature engineering on input
        issue_d = pd.to_datetime(application.issue_d, format='%b-%Y')
        earliest_cr_line = pd.to_datetime(
            application.earliest_cr_line, format='%b-%Y'
        )
        credit_history_years = (
            (issue_d - earliest_cr_line).days / 365
        )
        loan_to_income = application.loan_amnt / application.annual_inc
        dti_clean = min(application.dti, 100)
        high_dti = int(dti_clean > 20)
        high_revol_util = int(application.revol_util > 80)
        has_delinquency = int(application.delinq_2yrs > 0)

        # Build feature array in correct order
        features = np.array([[
            application.loan_amnt,
            application.int_rate,
            application.installment,
            application.annual_inc,
            dti_clean,
            application.delinq_2yrs,
            application.inq_last_6mths,
            application.open_acc,
            application.pub_rec,
            application.revol_bal,
            application.revol_util,
            application.total_acc,
            credit_history_years,
            loan_to_income,
            high_dti,
            high_revol_util,
            has_delinquency
        ]])

        # Impute and predict
        features_imputed = imputer.transform(features)
        default_prob = model.predict_proba(features_imputed)[0][1]
        propensity = 1 - default_prob
        decision = "Decline" if default_prob >= threshold else "Approve"
        return{
            "default_probability" : round(float(default_prob),3),
            "credit_risk_score" : round(1-float(default_prob)*100,1),
            "propensity_score" : round(float(propensity),3),
            "recommendation" : decision,
            "threshold_used" : round(float(threshold), 3),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))