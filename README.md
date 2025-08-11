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
