# api/routes/assess.py
# POST /assess — credit risk and propensity scoring

import numpy as np
from fastapi import APIRouter, HTTPException
from datetime import datetime
from api.schemas.request import LoanApplication
from api.schemas.response import AssessResponse
from api.utils import engineer_features
from src.data.database import save_prediction

router = APIRouter()


@router.post("/assess", response_model=AssessResponse)
def assess(application: LoanApplication):
    try:
        from api.main import model, threshold, imputer, feature_names

        features = engineer_features(application, feature_names)
        features_imputed = imputer.transform(features)
        default_prob = model.predict_proba(features_imputed)[0][1]
        propensity = 1 - default_prob
        decision = (
            "Decline" if default_prob >= threshold
            else "Approve"
        )

        response = AssessResponse(
            default_probability=round(float(default_prob), 3),
            credit_risk_score=round(
                (1 - float(default_prob)) * 100, 1
            ),
            propensity_score=round(float(propensity), 3),
            recommendation=decision,
            threshold_used=round(float(threshold), 3),
            timestamp=datetime.now().isoformat()
        )

        # Calculate RFM values
        rfm = {
            'recency': float(features[0][12]),
            'frequency': float(application.total_acc),
            'monetary': float(application.loan_amnt)
        }

        save_prediction(application, response, rfm)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))