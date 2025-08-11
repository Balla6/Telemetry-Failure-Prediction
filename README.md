# Telemetry Failure Prediction & Drift Alarms

Predict service incidents (P95 latency spikes or high error rates) **15 minutes early** with high precision, plus drift monitoring and a live dashboard.

**Why it matters:** In real services, early, precise alerts prevent outages and reduce pager fatigue.

## Architecture
Ingest (synthetic) → Parquet (bronze/silver/gold) → Feature builder → Model (XGBoost) → Thresholds & Alerts → Drift Monitors → Dashboard

