# src/models/baseline.py
"""
Baseline: Logistic Regression on silver features.
- Reads data/silver
- downsample negatives on train to speed up / balance
- Calibrates threshold on VAL to hit config.alerting.precision_target
- Evaluates on VAL and TEST
- Writes predictions to data/gold and metrics to reports/baseline_metrics.md

CLI:
  python -m src.models.baseline --conf conf/config.yaml --neg_downsample 0.2
  # or
  python src/models/baseline.py --conf conf/config.yaml --neg_downsample 0.2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
)

# -------------------------
# Config
# -------------------------
@dataclass
class Cfg:
    silver_dir: Path
    gold_dir: Path
    precision_target: float

def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    data = y["data"]
    alert = y.get("alerting", {})
    return Cfg(
        silver_dir=Path(data["silver_dir"]),
        gold_dir=Path(data["gold_dir"]),
        precision_target=float(alert.get("precision_target", 0.8)),
    )

# -------------------------
# IO
# -------------------------
def read_silver(silver_dir: Path) -> pd.DataFrame:
    files = sorted(Path(silver_dir).rglob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No silver files under {silver_dir}")
    parts = []
    for p in tqdm(files, desc="read_silver", unit="file"):
        df = pd.read_parquet(p)
        # normalize dtypes
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    # de-dup columns defensively
    df = df.loc[:, ~df.columns.duplicated()]
    # some builds saved split as category / string — normalize to string
    df["split"] = df["split"].astype(str)
    return df

def write_preds(df: pd.DataFrame, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = df.copy()
    x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")
    for (date_str, svc), part in tqdm(x.groupby(["date","service_id"], sort=False), desc="write_gold"):
        d = out_dir / f"date={date_str}" / f"service_id={svc}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "part-0000.parquet"
        if p.exists():
            p.unlink()
        part.drop(columns=["date"], inplace=False).to_parquet(p, index=False)

def write_report(md: str, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(md, encoding="utf-8")

# -------------------------
# Features/targets
# -------------------------
KEYS = ["timestamp","service_id","env","split","day_idx"]
TARGET = "y_label_15min"

def make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    # numeric feature set: all numeric except the target
    num_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if TARGET in num_cols:
        num_cols.remove(TARGET)
    X = df[num_cols].astype("float32")
    y = df[TARGET].astype("int8").to_numpy()
    return X, y, num_cols

def downsample_negatives(X: pd.DataFrame, y: np.ndarray, frac: float, rng: np.random.RandomState):
    """Keep all positives, sample a fraction of negatives."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep_neg = rng.choice(neg_idx, size=int(len(neg_idx) * frac), replace=False) if len(neg_idx) else neg_idx
    keep = np.concatenate([pos_idx, keep_neg])
    keep.sort()
    return X.iloc[keep], y[keep]

# -------------------------
# Threshold calibration
# -------------------------
def calibrate_threshold(y_true: np.ndarray, scores: np.ndarray, precision_target: float) -> float:
    """Return the smallest threshold achieving >= precision_target on VAL.
    If never achieved, return 1.0 (no alerts)."""
    p, r, th = precision_recall_curve(y_true, scores)
    # p/r are length n+1, th length n
    best = 1.0
    for prec, t in zip(p, np.append(th, 1.0)):
        if prec >= precision_target:
            best = float(min(best, t))
    return float(best)

def metrics_block(y_true: np.ndarray, scores: np.ndarray, split_name: str, threshold: float) -> str:
    ap = average_precision_score(y_true, scores)
    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = float("nan")
    y_pred = (scores >= threshold).astype(int)
    rep = classification_report(y_true, y_pred, digits=3, zero_division=0)
    return (
        f"### {split_name}\n"
        f"- Average Precision (PR-AUC): **{ap:.4f}**\n"
        f"- ROC-AUC: **{roc:.4f}**\n"
        f"- Threshold used: **{threshold:.4f}**\n\n"
        "```\n" + rep + "\n```\n"
    )

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--neg_downsample", type=float, default=0.2,
                    help="fraction of negative class to keep in TRAIN (default 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    cfg = load_config(args.conf)

    print("[baseline] reading silver…")
    df = read_silver(cfg.silver_dir)

    # split views
    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    print(f"[baseline] sizes train={len(train):,} val={len(val):,} test={len(test):,}")

    # features / targets
    Xtr, ytr, feat_names = make_xy(train)
    Xva, yva, _ = make_xy(val)
    Xte, yte, _ = make_xy(test)

    # optional negative downsample on train
    if 0 < args.neg_downsample < 1.0:
        Xtr, ytr = downsample_negatives(Xtr, ytr, args.neg_downsample, rng)
        print(f"[baseline] after downsample: train={len(ytr):,} (pos={ytr.sum():,})")

    # scale
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # model
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=None,
        class_weight="balanced",
    )
    print("[baseline] training logistic regression…")
    clf.fit(Xtr_s, ytr)

    # scores
    va_scores = clf.predict_proba(Xva_s)[:, 1]
    te_scores = clf.predict_proba(Xte_s)[:, 1]

    # threshold by precision target on VAL
    thr = calibrate_threshold(yva, va_scores, cfg.precision_target)
    print(f"[baseline] calibrated threshold (val precision>={cfg.precision_target:.2f}) -> {thr:.4f}")

    # metrics report
    md = [
        "# Baseline (Logistic Regression) Metrics",
        "",
        f"- Features used: **{len(feat_names)}**",
        f"- Precision target (config): **{cfg.precision_target:.2f}**",
        "",
        metrics_block(yva, va_scores, "Validation", thr),
        metrics_block(yte, te_scores, "Test", thr),
    ]
    write_report("\n".join(md), Path("reports/baseline_metrics.md"))
    print("[baseline] wrote reports/baseline_metrics.md")

    # write predictions per partition to data/gold
    keep_cols = ["timestamp", "service_id", "env", "split", TARGET]
    val_out = val[keep_cols].copy()
    val_out["score"] = va_scores
    test_out = test[keep_cols].copy()
    test_out["score"] = te_scores
    preds = pd.concat([val_out, test_out], ignore_index=True)
    write_preds(preds, cfg.gold_dir)
    print(f"[baseline] wrote predictions to {cfg.gold_dir}")

    print("[baseline] done.")

if __name__ == "__main__":
    main()

