# Week 5 — Credit Risk Scoring + Propensity + RFM
**Martin James Ng'ang'a · MLOps Engineer · Nairobi, Kenya 🇰🇪**
`github.com/M20Jay` · Week 5 of 15

---

## Overview

Production credit risk pipeline trained on 252,971 real LendingClub loan decisions.
Answers three questions simultaneously for every applicant:

1. **Will they default?** → XGBoost · ROC-AUC 0.703 · 65% recall on defaulters
2. **Will they accept the offer?** → Propensity scoring from default probability
3. **How valuable are they?** → RFM proxy segmentation · KMeans clustering

Full MLOps stack: FastAPI · PostgreSQL · Grafana · Docker · AWS EC2 Frankfurt.

---

## Final Results

| Metric | Result |
|--------|--------|
| Dataset | 252,971 LendingClub loans · 17.9% default rate |
| Best Model | XGBoost · ROC-AUC 0.703 |
| Recall on defaulters | 65% · met business requirement |
| Optimal threshold | 0.183 · business-driven not accuracy-driven |
| ADASYN balancing | 27,149 → 135,353 minority samples |
| Mean propensity | 0.821 on test set |
| Live API | http://18.184.3.203:8005/docs |
| Endpoints | GET /health · POST /assess · POST /segment |
| Grafana panels | 6 panels · total predictions, decline rate, pie chart, avg prob, loan amount, time series |

---

## Project Structure

```
credit-risk-scoring-pipeline/
├── configs/
│   └── model.yaml              ALL hyperparameters in one place
├── src/
│   ├── data/
│   │   ├── preprocessing.py    Load, clean, encode categoricals
│   │   └── database.py         PostgreSQL connection, save predictions
│   ├── features/
│   │   └── feature_engineering.py  RFM proxy, KMeans segments
│   └── models/
│       ├── train.py            ADASYN, XGBoost, calibration, threshold, save artifacts
│       └── evaluate.py         Load model, evaluate, SHAP plots
├── api/
│   ├── main.py                 Entry point · loads artifacts once at startup
│   ├── utils.py                Shared feature engineering (DRY principle)
│   ├── schemas/
│   │   ├── request.py          LoanApplication · what API expects
│   │   └── response.py         AssessResponse, SegmentResponse, HealthResponse
│   └── routes/
│       ├── health.py           GET /health
│       ├── assess.py           POST /assess · credit risk + propensity
│       └── segment.py          POST /segment · RFM values
├── Dockerfile
├── docker-compose.yml          API + PostgreSQL + Grafana
└── requirements.txt
```

**The Golden Rule:** Write once. Import everywhere. Never copy-paste logic between files.

---

## Pipeline Architecture

```
configs/model.yaml
    ↓ load_config()
src/data/preprocessing.py       → cleaned df
    ↓
src/features/feature_engineering.py  → df with engineered features + feature list
    ↓
src/models/train.py             → ADASYN → XGBoost → calibration → threshold
    ↓ saves:
    credit_risk_model.pkl
    threshold.pkl
    imputer.pkl
    feature_names.pkl
    ↓
api/main.py                     → loads all 4 artifacts once at startup
    ↓
POST /assess request arrives
    ↓
api/routes/assess.py            → orchestrator
    ↓
api/utils.py                    → engineer_features() → 40-feature numpy array
    ↓
imputer.transform() → model.predict_proba()
    ↓
src/data/database.py            → save_prediction() to PostgreSQL
    ↓
return AssessResponse to caller
```

**Debugging rule:** If it breaks, the pipeline map tells you exactly which file to look in.

| Symptom | File to check |
|---------|--------------|
| Model prediction wrong | `src/models/train.py` |
| Feature mismatch error | `src/features/feature_engineering.py` |
| API endpoint 422 error | `api/routes/assess.py` |
| Database not saving | `src/data/database.py` |
| Config change not working | `configs/model.yaml` |

---

## Key Concepts

### Supervised vs Unsupervised

| Type | When to Use |
|------|-------------|
| **Supervised** | You have labels. Predicting a known outcome. XGBoost, LightGBM, Logistic Regression |
| **Unsupervised** | No labels. Finding hidden structure. KMeans, DBSCAN, PCA, Isolation Forest |

This pipeline uses both: XGBoost (supervised) for default prediction + KMeans (unsupervised) for customer segmentation.

---

### Why Threshold 0.183 Not 0.5

```python
# Default threshold 0.5 on imbalanced data:
# → Model predicts "no default" for everyone
# → 82.1% accuracy — looks good
# → 0% recall on defaulters — completely useless for a bank

# Business-calibrated threshold 0.183:
# → 65% recall on defaulters — catches 2 in 3 bad loans
# → Balances cost of missed defaulter vs false rejection
# → 0.183 is a business decision, not a technical one

# How we found it:
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
# Plot and find threshold where recall >= 0.65
```

**Interview answer:** "The default 0.5 threshold maximises accuracy but fails on imbalanced datasets. I calibrated to 0.183 to achieve 65% recall on defaulters — balancing the cost of a missed defaulter against the cost of a false rejection. That is a business decision, not a technical one."

---

### ADASYN vs SMOTE

| Method | When to Use |
|--------|-------------|
| **ADASYN** | Adaptive — generates more samples near decision boundary. Better for complex class imbalance |
| **SMOTE** | Simpler — generates synthetic samples uniformly. Good starting point |

```python
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
# 27,149 minority samples → 135,353 after ADASYN
```

---

### Model Calibration — Why It Matters

```python
from sklearn.calibration import CalibratedClassifierCV

# Without calibration:
# Model says probability = 0.9 but true probability = 0.6
# predict_proba() returns ranks not true probabilities

# With calibration:
calibrated_model = CalibratedClassifierCV(
    xgb_model,
    method='isotonic',  # isotonic for large datasets
    cv='prefit'         # model already trained
)
calibrated_model.fit(X_val, y_val)

# Now probabilities are reliable for business thresholds
```

---

### SHAP Explainability — Why It Matters for Banking

```python
import shap

# CRITICAL: Use base XGBoost estimator, not calibrated wrapper
base_model = model.calibrated_classifiers_[0].estimator
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_test)

# Waterfall plot — why THIS specific customer was declined
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0],
    feature_names=feature_names
))
```

**Why SHAP matters:**
- Feature importance = what matters globally across all customers
- SHAP = why **this specific customer** was declined — mathematically
- Basel III requires banks to explain every credit decision
- EU AI Act classifies credit scoring as high-risk AI — explainability mandatory

---

### RFM — Proxy vs True

```python
# True RFM requires transaction history — multiple purchases over time
# LendingClub has one loan per row — no transaction history

# Proxy RFM — what we built:
# Recency    → credit_history_years (how long since first credit)
# Frequency  → total_acc (total number of credit accounts)
# Monetary   → loan_amnt (loan amount requested)

# Always document this limitation in README and code comments
```

---

### Lasso vs Ridge vs ElasticNet

| Method | When to Use |
|--------|-------------|
| **Lasso (L1)** | Many features, some irrelevant. Drives coefficients to zero — automatic feature selection |
| **Ridge (L2)** | All features likely relevant. Shrinks coefficients but keeps all. Reduces overfitting |
| **ElasticNet** | Unsure which to use. Combines both. Safe default |

---

## FastAPI Patterns

### Three-Layer Schema Pattern

```python
# Layer 1 — What comes IN
# api/schemas/request.py
class LoanApplication(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    dti: float
    # ... all 40 features

# Layer 2 — The logic
# api/routes/assess.py
@router.post("/assess", response_model=AssessResponse)
async def assess(application: LoanApplication):
    features = engineer_features(application, feature_names)
    features_imputed = imputer.transform(features)
    proba = model.predict_proba(features_imputed)[0][1]
    decision = "DECLINE" if proba >= threshold else "APPROVE"
    save_prediction(application, response, rfm)
    return AssessResponse(...)

# Layer 3 — What goes OUT
# api/schemas/response.py
class AssessResponse(BaseModel):
    decision: str           # APPROVE or DECLINE
    default_probability: float
    propensity_score: float
    risk_tier: str          # LOW / MEDIUM / HIGH / VERY HIGH
    shap_top_factors: list
```

### Load Artifacts Once at Startup

```python
# api/main.py — CORRECT
# Load once when server starts — all requests share the same loaded model
model = joblib.load("models/credit_risk_model.pkl")
threshold = joblib.load("models/threshold.pkl")
imputer = joblib.load("models/imputer.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# In routes — import from main
from api.main import model, threshold, imputer, feature_names

# WRONG — never load per request
# @router.post("/assess")
# async def assess(application: LoanApplication):
#     model = joblib.load("models/credit_risk_model.pkl")  # ← too slow
```

### Always Wrap Endpoints in try/except

```python
@router.post("/assess", response_model=AssessResponse)
async def assess(application: LoanApplication):
    try:
        features = engineer_features(application, feature_names)
        # ... prediction logic
        return AssessResponse(...)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Without try/except → ugly Python traceback
# With try/except → clean JSON error message callers can read
```

---

## Docker Patterns

### Image vs Container

| Concept | Explanation |
|---------|-------------|
| **Image** | Blueprint. Read-only template. Built from Dockerfile. Like a recipe |
| **Container** | Running instance of an image. Has its own state. Like a cooked meal |

### Why COPY requirements.txt Before COPY .

```dockerfile
# CORRECT — cache-optimised
COPY requirements.txt .
RUN pip install -r requirements.txt  # cached unless requirements.txt changes
COPY . .                              # only invalidates package cache when code changes

# WRONG — rebuilds packages every time
COPY . .
RUN pip install -r requirements.txt  # reinstalls packages on every code change
```

### Docker Compose Key Points

```yaml
services:
  api:
    depends_on:
      - postgres        # postgres starts before api
    environment:
      - DB_HOST=postgres  # service name not localhost — critical inside Docker
    volumes:
      - .:/app          # code changes reflect without rebuilding

  postgres:
    volumes:
      - postgres_data:/var/lib/postgresql/data  # named volume persists across restarts

volumes:
  postgres_data:        # declare named volume here
```

---

## CLI Reference

### Training Pipeline

```bash
# Run full training pipeline
PYTHONPATH=. python src/data/preprocessing.py
PYTHONPATH=. python src/features/feature_engineering.py
PYTHONPATH=. python src/models/train.py
PYTHONPATH=. python src/models/evaluate.py

# Check saved artifacts
ls -lh models/
# credit_risk_model.pkl
# threshold.pkl
# imputer.pkl
# feature_names.pkl
```

### API Testing

```bash
# Health check
curl -s http://localhost:8005/health | python3 -m json.tool

# Credit risk assessment
curl -s -X POST http://localhost:8005/assess \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000,
    "int_rate": 13.5,
    "annual_inc": 65000,
    "dti": 18.5,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 8,
    "pub_rec": 0,
    "revol_bal": 12000,
    "revol_util": 45.2,
    "total_acc": 15,
    "credit_history_years": 8
  }' | python3 -m json.tool

# Customer segment
curl -s -X POST http://localhost:8005/segment \
  -H "Content-Type: application/json" \
  -d '{"loan_amnt": 15000, "annual_inc": 65000, "credit_history_years": 8, "total_acc": 15}' \
  | python3 -m json.tool
```

### Docker Commands

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View API logs
docker compose logs api --tail=20

# Restart API only
docker compose restart api

# Stop everything
docker compose down

# Rebuild after code changes
docker compose up --build -d

# Check port binding on server
sudo ss -tlnp | grep 8005
```

### MLflow Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002

# Check runs from terminal
python3 -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
runs = client.search_runs('1', order_by=['start_time DESC'], max_results=5)
for run in runs:
    print(f'{run.info.run_name} | {run.info.status} | {run.data.metrics}')
"
```

### Database Inspection

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U creditrisk -d creditrisk

# Inside psql:
\dt                              -- list tables
SELECT COUNT(*) FROM predictions;
SELECT decision, COUNT(*) FROM predictions GROUP BY decision;
SELECT AVG(default_probability) FROM predictions;
\q                               -- quit
```

### Git Workflow

```bash
git status
git add src/models/train.py src/models/evaluate.py
git commit -m "feat: add ADASYN balancing and threshold calibration"
git push origin main
git log --oneline -5
```

---

## Debugging Reference

### Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Check `__init__.py` exists in folder. Run as `python -m src.models.train` not `python src/models/train.py` |
| `SimpleImputer expects N features` | Feature mismatch between training and API. Check `feature_names.pkl` is saved and loaded correctly |
| `XGBClassifier regressor error` | sklearn version conflict. Use `scikit-learn==1.4.2` with `xgboost==2.0.3` |
| `FileNotFoundError models/` | Model files not committed. Remove `*.pkl` from `.gitignore` |
| `Address already in use port 8005` | Kill existing process: `lsof -i :8005` then `kill -9 <PID>` |
| `column does not exist PostgreSQL` | Table schema changed. `DROP TABLE predictions;` then restart API |
| `SHAP TreeExplainer error` | Use base estimator: `model.calibrated_classifiers_[0].estimator` |

### Debugging Order

```
1. Read the LAST line of the traceback — that is the actual error
2. Read the file and line number — that is where it happened
3. Check if it is a typo — most errors are typos
4. Check imports — is the module name correct?
5. Check file exists — ls the folder
6. Google the exact error message — someone has seen it before
```

---

## AWS EC2 Deployment

```bash
# SSH to server
ssh -i ~/Documents/GitHub/mlops-key.pem ubuntu@18.184.3.203

# Start credit risk API
cd ~/credit-risk-scoring-pipeline
docker compose up -d api

# Verify running
docker ps | grep credit
curl -s http://localhost:8005/health

# Check logs
docker compose logs api --tail=20

# Fix postgres port conflict (if 5432 already taken)
# sed -i 's/"5432:5432"/"5435:5432"/' docker-compose.yml
```

---

## Key Decisions Explained

**Why XGBoost over Logistic Regression?**
252,971 rows with non-linear relationships between features and default. XGBoost handles feature interactions automatically. Logistic regression assumes linear relationships — incorrect for credit data.

**Why ADASYN over SMOTE?**
17.9% default rate — moderate imbalance. ADASYN generates more synthetic samples near the decision boundary — the hard cases the model struggles with most. SMOTE generates uniformly — less targeted.

**Why isotonic calibration over Platt scaling?**
252,971 training samples — large dataset. Isotonic regression is non-parametric and performs better on large datasets. Platt scaling (sigmoid) is better for small datasets.

**Why proxy RFM instead of true RFM?**
LendingClub has one loan per customer row — no transaction history. True RFM requires multiple purchases over time. Proxy RFM uses available loan application features as approximations. Always documented as a limitation.

---

## Interview Q&A

**Q: What is ROC-AUC and what does 0.703 mean?**
A: Area Under the Receiver Operating Curve. Probability that the model ranks a random defaulter higher than a random non-defaulter. 0.703 means the model correctly ranks 70.3% of paired comparisons. A random model scores 0.5. Perfect model scores 1.0. 0.703 is competitive for credit risk on an imbalanced dataset.

**Q: Why not maximise accuracy?**
A: With 17.9% default rate, predicting "no default" for everyone gives 82.1% accuracy but 0% recall on defaulters — completely useless for a bank. I optimised for recall on the minority class with a business-calibrated threshold of 0.183.

**Q: What is SHAP and why does it matter for banking?**
A: SHapley Additive exPlanations — mathematical attribution of each feature's contribution to every individual prediction. Basel III requires that every credit decision be auditable. SHAP provides per-applicant, per-feature, directional attribution — not a summary, not an approximation. Audit-ready from day one.

**Q: What is data leakage and how did you prevent it?**
A: Data leakage is when information from the test set influences model training, giving optimistically biased performance metrics. Prevention: fit imputer and scaler only on training set, apply to test set. Apply ADASYN only on training set, never on test. Split before any preprocessing.

---

*Week 5 of 15 · Credit Risk Scoring Pipeline · Built in Nairobi, Kenya 🇰🇪*
*Live API: http://18.184.3.203:8005/docs · Repository: https://github.com/M20Jay/credit-risk-scoring-pipeline*
