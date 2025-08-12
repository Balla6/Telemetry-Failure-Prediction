# app/dashboard.py
# Streamlit dashboard for Telemetry Failure Prediction

import glob
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve

TARGET = "y_label_15min"

# -------- config helpers --------
def load_defaults(conf_path: str = "conf/config.yaml"):
    try:
        with open(conf_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
    except FileNotFoundError:
        y = {}
    data = y.get("data", {})
    alert = y.get("alerting", {})
    gold_dir = str(data.get("gold_dir", "data/gold"))
    default_thr = float(alert.get("default_threshold", 0.976))
    default_cd = int(alert.get("cooldown_min", 15))
    default_prec_target = float(alert.get("precision_target", 0.80))
    return gold_dir, default_thr, default_cd, default_prec_target

GOLD_DIR, DEFAULT_THR, DEFAULT_COOLDOWN, DEFAULT_PRECISION_TARGET = load_defaults()

@st.cache_data(show_spinner=False)
def load_gold(gold_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(gold_dir) / "date=*/service_id=*/part-0000.parquet")))
    if not files:
        raise FileNotFoundError(f"No gold predictions under {gold_dir}")
    parts = []
    for p in files:
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["split"] = df["split"].astype(str)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]
    need = {"timestamp","service_id","split",TARGET,"score"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"gold predictions missing columns: {missing}")
    return df.sort_values(["service_id","timestamp"]).reset_index(drop=True)

def calibrate_threshold(y_true: np.ndarray, scores: np.ndarray, precision_target: float) -> float:
    p, r, th = precision_recall_curve(y_true, scores)
    thr = 1.0
    for prec, t in zip(p, np.append(th, 1.0)):
        if prec >= precision_target:
            thr = min(thr, float(t))
    return float(thr)

def event_metrics(df: pd.DataFrame, thr: float):
    fired = df["score"] >= thr
    tp = int(((df[TARGET] == 1) & fired).sum())
    fp = int(((df[TARGET] == 0) & fired).sum())
    fn = int(((df[TARGET] == 1) & (~fired)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec, int(fired.sum())

def apply_cooldown(df: pd.DataFrame, cooldown_min: int, thr: float) -> pd.DataFrame:
    df = df.sort_values(["service_id","timestamp"]).copy()
    df["alert"] = 0
    last_ts = {}
    for i, row in df.iterrows():
        if row["score"] >= thr:
            svc = row["service_id"]
            ts = row["timestamp"]
            lt = last_ts.get(svc)
            if lt is None or (ts - lt).total_seconds() >= cooldown_min * 60:
                df.at[i, "alert"] = 1
                last_ts[svc] = ts
    return df

# ---------------- UI ----------------
st.set_page_config(page_title="Telemetry Failure Prediction", layout="wide")
st.title("Telemetry Failure Prediction")

df = load_gold(GOLD_DIR)
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

tab_alerts, tab_drift = st.tabs(["Alerts & Thresholds", "Drift"])

with tab_alerts:
    st.header("Alerts & Thresholds")
    st.sidebar.header("Controls")
    precision_target = st.sidebar.slider(
        "Precision target (for auto-calibrate from VAL)",
        0.5, 0.95, float(DEFAULT_PRECISION_TARGET), 0.01
    )
    auto_thr = calibrate_threshold(val[TARGET].to_numpy(), val["score"].to_numpy(), precision_target)

    threshold = st.sidebar.slider(
        "Threshold (manual)",
        0.0, 1.0,
        value=float(DEFAULT_THR), step=0.0001, key="threshold"
    )
    cooldown = st.sidebar.slider(
        "Cooldown (minutes)",
        0, 60,
        value=int(DEFAULT_COOLDOWN), step=1, key="cooldown"
    )

    colb1, colb2 = st.sidebar.columns(2)
    if colb1.button("Use auto thr"):
        st.session_state.threshold = float(round(auto_thr, 4))
        threshold = st.session_state.threshold
    if colb2.button("Use config"):
        st.session_state.threshold = float(DEFAULT_THR)
        st.session_state.cooldown = int(DEFAULT_COOLDOWN)
        threshold = st.session_state.threshold
        cooldown = st.session_state.cooldown

    ap_val = average_precision_score(val[TARGET], val["score"])
    p_val, r_val, n_val = event_metrics(val, threshold)
    st.subheader("Validation (used to calibrate)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC (VAL)", f"{ap_val:.4f}")
    c2.metric("Precision @thr", f"{p_val:.3f}")
    c3.metric("Recall @thr", f"{r_val:.3f}")
    c4.metric("Alerts fired (VAL)", f"{n_val}")

    test_cd = apply_cooldown(test, cooldown, threshold)
    p_te, r_te, n_te = event_metrics(test_cd, threshold)
    st.subheader("Test (policy view)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision @thr", f"{p_te:.3f}")
    c2.metric("Recall @thr", f"{r_te:.3f}")
    c3.metric("Alerts fired (TEST)", f"{n_te}")

    per_day = (
        test_cd.assign(date=test_cd["timestamp"].dt.tz_convert(None).dt.strftime("%Y-%m-%d"))
        .groupby("date")["alert"].sum().reset_index().rename(columns={"alert":"alerts"})
        .sort_values("date")
    )
    st.markdown("#### Alerts per day (TEST)")
    st.line_chart(per_day.set_index("date"))

    st.markdown("#### Top 50 alerts (TEST)")
    top = test_cd.loc[test_cd["score"] >= threshold, ["timestamp","service_id","score",TARGET]] \
                .sort_values("score", ascending=False).head(50)
    st.dataframe(top.reset_index(drop=True))

    st.caption(
        f"Config defaults — thr: {DEFAULT_THR:.4f} • cooldown: {DEFAULT_COOLDOWN} min • "
        f"VAL auto-calibrated thr @ precision≥{precision_target:.2f}: {auto_thr:.4f}"
    )

with tab_drift:
    st.header("Drift (VAL → TEST)")
    path = Path("reports/drift_report.md")
    if path.exists():
        st.download_button("Download drift_report.md", data=path.read_bytes(),
                           file_name="drift_report.md", mime="text/markdown")
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.info("Run:  `export PYTHONPATH=.; python src/monitoring/drift.py --conf conf/config.yaml`")
