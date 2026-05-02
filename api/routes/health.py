# api/routes/health.py
from fastapi import APIRouter
from datetime import datetime
from api.schemas.response import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model="XGBoost Credit Risk v1.0",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )