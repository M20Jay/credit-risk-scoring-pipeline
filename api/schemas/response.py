# api/schemas/response.py
# Response models — what the API returns

from pydantic import BaseModel


class AssessResponse(BaseModel):
    default_probability: float
    credit_risk_score: float
    propensity_score: float
    recommendation: str
    threshold_used: float
    timestamp: str


class SegmentResponse(BaseModel):
    rfm_recency: float
    rfm_frequency: float
    rfm_monetary: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    timestamp: str