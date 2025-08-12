# Telemetry Failure Prediction & Drift Alarms

Predict service incidents (P95 latency spikes or high error rates) **15 minutes early** with high precision, plus drift monitoring and a live dashboard.

**Why it matters:** In real services, early, precise alerts prevent outages and reduce pager fatigue.

## Architecture
Ingest (synthetic) → Parquet (bronze/silver/gold) → Feature builder → Model (XGBoost) → Thresholds & Alerts → Drift Monitors → Dashboard

## How to run (top-level)
```bash
# 1) install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) generate data, validate, build features, train & evaluate
make all

# 3) run dashboard
make dash


## Results

**Model:** XGBoost on engineered time-series features  
**Data:** 30 days × 50 services @ 1-min frequency (synthetic telemetry)

**Validation**
- PR-AUC: **0.1070**
- Calibrated threshold for demo: **0.976**
- Precision @thr: **0.326**
- Recall @thr: **0.103**
- Alerts fired (VAL): **227**

**Test (policy view)**
- Cooldown: **15 minutes**
- See `reports/alert_policy.md` for precision/recall and alerts/day

**Artifacts**
- Gold predictions: `data/gold/...`
- Model: `models/xgb_model.json`
- Reports: `reports/xgb_metrics.md`, `reports/alert_policy.md`

