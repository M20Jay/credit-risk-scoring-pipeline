# src/data/database.py
# PostgreSQL connection and predictions storage

import psycopg2
import os
from datetime import datetime


def get_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "credit_risk"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        port=os.getenv("DB_PORT", "5432")
    )


def create_table():
    """Create predictions table if not exists"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                loan_amnt FLOAT,
                int_rate FLOAT,
                annual_inc FLOAT,
                grade VARCHAR(10),
                purpose VARCHAR(100),
                default_probability FLOAT,
                credit_risk_score FLOAT,
                propensity_score FLOAT,
                recommendation VARCHAR(20),
                threshold_used FLOAT,
                rfm_recency FLOAT,
                rfm_frequency FLOAT,
                rfm_monetary FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("Predictions table ready ✅")
    except Exception as e:
        print(f"Database not available: {e}")


def save_prediction(application, response, rfm=None):
    """Save prediction to PostgreSQL"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                loan_amnt, int_rate, annual_inc,
                grade, purpose,
                default_probability, credit_risk_score,
                propensity_score, recommendation, threshold_used,
                rfm_recency, rfm_frequency, rfm_monetary
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            application.loan_amnt,
            application.int_rate,
            application.annual_inc,
            application.grade,
            application.purpose,
            response.default_probability,
            response.credit_risk_score,
            response.propensity_score,
            response.recommendation,
            response.threshold_used,
            rfm['recency'] if rfm else None,
            rfm['frequency'] if rfm else None,
            rfm['monetary'] if rfm else None,
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")