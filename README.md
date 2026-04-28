# Week 5 — Credit Risk Scoring Pipeline

**Author:** Martin James Ng'ang'a | [github.com/M20Jay](https://github.com/M20Jay)  
**Status:** ⏳ Week 5 of 15 — Building Now  
**Stack:** XGBoost · SHAP · DVC · RFM · FastAPI · PostgreSQL · Grafana · Docker · Render

---

## Business Problem

A bank needs to answer three questions simultaneously for every customer:

1. **Will they default on a loan?** — Credit risk score
2. **Will they accept a loan offer?** — Propensity score
3. **How valuable are they?** — RFM customer segment

This API answers all three in one call — with SHAP explainability for every decision.

---

## Dataset

📁 LendingClub — 421MB — real loan application data

---

## Week 5 Daily Progress

| Day | Task | Status |
|-----|------|--------|
| Day 1 | EDA — loan.csv, target variable, distributions | ✅ Complete |
| Day 2 | XGBoost model + calibration + optimal threshold | ✅ Complete |
| Day 3 | SHAP explainability + src/ modular structure | ⏳ |
| Day 4 | Propensity scoring + RFM segmentation + DVC | ⏳ |
| Day 5 | FastAPI endpoints + PostgreSQL predictions storage | ⏳ |
| Day 6 | Grafana dashboard + Docker + deployment | ⏳ |
| Day 7 | README complete + LinkedIn post + GitHub pushed | ⏳ |

## Day 1 Key Findings

| Finding | Detail |
|---------|--------|
| Total dataset | 887,379 loans — 74 features |
| Modelling dataset | 252,971 loans — Fully Paid and Charged Off only |
| Default rate | 17.9% — moderate class imbalance |
| Strongest predictor | Loan grade — A=5% default, G=42% default |
| File format | Converted CSV 421MB → Parquet 36MB — 11x smaller |
| New features | dti_clean · credit_history_years · loan_to_income |

---
## Day 2 Key Findings

| Metric | Result |
|--------|--------|
| Best model | XGBoost — ROC-AUC 0.701 |
| Optimal threshold | 0.187 — business driven |
| Recall on defaulters | 65% — met requirement |
| False negatives | 3,164 missed defaulters |
| False positives | 14,760 wrongly declined |
| Model saved | models/credit_risk_model.pkl |
| Next improvement | Better features + ADASYN |

---
## Stack

| Tool | Purpose |
|------|---------|
| XGBoost | Credit risk and propensity models |
| SHAP | Explainability — why each decision was made |
| DVC | Data version control |
| RFM | Customer segmentation — Recency, Frequency, Monetary |
| FastAPI | Serves all three models in one API call |
| PostgreSQL | Stores every prediction with timestamp |
| Grafana | Live monitoring dashboard |
| Docker | Containerisation |
| Render | Deployment |

---

## Why SHAP Matters for Banking

Regulators require banks to explain every loan decision.  
Feature importance tells you what matters globally.  
SHAP tells you **why this specific customer was declined**.

---

*Building from Nairobi. Deployed to the world. 🇰🇪*