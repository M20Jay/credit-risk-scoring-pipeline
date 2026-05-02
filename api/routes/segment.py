# api/routes/segment.py
# POST /segment — RFM proxy values

import pandas as pd
from fastapi import APIRouter, HTTPException
from datetime import datetime
from api.schemas.request import LoanApplication
from api.schemas.response import SegmentResponse
from api.utils import engineer_features

router = APIRouter()


@router.post("/segment", response_model=SegmentResponse)
def segment(application: LoanApplication):
    try:
        features_eng = engineer_features(application)

        return SegmentResponse(
            rfm_recency=round(
                features_eng['credit_history_years'], 1
            ),
            rfm_frequency=application.total_acc,
            rfm_monetary=application.loan_amnt,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))