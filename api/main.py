# api/main.py
# Entry point — creates app, loads artifacts, registers routes

import joblib
from fastapi import FastAPI
from api.routes import health, assess, segment
from src.data.database import create_table

# Load artifacts once at startup
model = joblib.load('models/credit_risk_model.pkl')
threshold = joblib.load('models/optimal_threshold.pkl')
imputer = joblib.load('models/imputer.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Create FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="Scores loan applicants on credit risk, propensity and RFM segment",
    version="1.0.0"
)

# Create database table on startup
@app.on_event("startup")
async def startup_event():
    create_table()

# Register routes
app.include_router(health.router)
app.include_router(assess.router)
app.include_router(segment.router)