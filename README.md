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
| Day 1 | EDA — loan.csv, target variable, distributions | ⏳ |
| Day 2 | XGBoost credit risk model + calibration | ⏳ |
| Day 3 | SHAP explainability — summary, waterfall, force plots | ⏳ |
| Day 4 | Propensity scoring + RFM segmentation + DVC | ⏳ |
| Day 5 | FastAPI endpoints + PostgreSQL predictions storage | ⏳ |
| Day 6 | Grafana dashboard + Docker + deployment | ⏳ |
| Day 7 | README complete + LinkedIn post + GitHub pushed | ⏳ |

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