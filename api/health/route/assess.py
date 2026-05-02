# api/routes/assess.py
# POST /assess — credit risk and propensity scoring

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from datetime import datetime
from api.schemas.request import LoanApplication
from api.schemas.response import AssessResponse
from api.utils import engineer_features

router = APIRouter()


@router.post("/assess", response_model=AssessResponse)
def assess(application: LoanApplication):
    try:
        from api.main import model, threshold, imputer
        features_eng = engineer_features(application)

        features = np.array([[
            application.loan_amnt,
            application.int_rate,
            application.installment,
            application.annual_inc,
            features_eng['dti_clean'],
            application.delinq_2yrs,
            application.inq_last_6mths,
            application.open_acc,
            application.pub_rec,
            application.revol_bal,
            application.revol_util,
            application.total_acc,
            features_eng['credit_history_years'],
            features_eng['loan_to_income'],
            features_eng['high_dti'],
            features_eng['high_revol_util'],
            features_eng['has_delinquency']
        ]])

        # Impute and predict
        features_imputed = imputer.transform(features)
        default_prob = model.predict_proba(
            features_imputed
        )[0][1]
        propensity = 1 - default_prob
        decision = (
            "Decline" if default_prob >= threshold
            else "Approve"
        )

        return AssessResponse(
            default_probability=round(float(default_prob), 3),
            credit_risk_score=round(
                (1 - float(default_prob)) * 100, 1
            ),
            propensity_score=round(float(propensity), 3),
            recommendation=decision,
            threshold_used=round(float(threshold), 3),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))