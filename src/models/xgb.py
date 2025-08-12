# src/models/xgb.py
"""
Primary model: XGBoost on silver features.

- Reads data/silver
- Handles class imbalance via scale_pos_weight
- Early-stopping on VAL with AUPRC (when available)
- Calibrates threshold on VAL to hit alerting.precision_target
- Evaluates on TEST
- Writes predictions to data/gold (overwrites baseline preds)
- Writes metrics to reports/xgb_metrics.md and saves model
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    raise SystemExit("xgboost not installed. Run:  pip install xgboost")

TARGET = "y_label_15min"
KEYS = ["timestamp", "service_id", "env", "split", "day_idx"]


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
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df["split"] = df["split"].astype(str)
    return df

def write_preds(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = df.copy()
    x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")
    for (date_str, svc), part in tqdm(x.groupby(["date", "service_id"], sort=False), desc="write_gold"):
        d = out_dir / f"date={date_str}" / f"service_id={svc}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "part-0000.parquet"
        if p.exists():
            p.unlink()
        part.drop(columns=["date"], inplace=False).to_parquet(p, index=False)

def write_report(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# -------------------------
# Features / target
# -------------------------
def make_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    num_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if TARGET in num_cols:
        num_cols.remove(TARGET)
    X = df[num_cols].astype("float32").to_numpy()
    y = df[TARGET].astype("int8").to_numpy()
    return X, y, num_cols

def scale_pos_weight(y: np.ndarray) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / max(1, pos))


# -------------------------
# Threshold calibration
# -------------------------
def calibrate_threshold(y_true: np.ndarray, scores: np.ndarray, precision_target: float) -> float:
    p, r, th = precision_recall_curve(y_true, scores)
    thr = 1.0
    for prec, t in zip(p, np.append(th, 1.0)):
        if prec >= precision_target:
            thr = min(thr, float(t))
    return float(thr)

def metrics_block(y_true: np.ndarray, scores: np.ndarray, name: str, thr: float) -> str:
    ap = average_precision_score(y_true, scores)
    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = float("nan")
    pred = (scores >= thr).astype(int)
    rep = classification_report(y_true, pred, digits=3, zero_division=0)
    return (
        f"### {name}\n"
        f"- PR-AUC: **{ap:.4f}**  |  ROC-AUC: **{roc:.4f}**  |  thr: **{thr:.4f}**\n\n"
        "```\n" + rep + "\n```\n"
    )


# -------------------------
# Helpers for version-proof prediction
# -------------------------
def booster_predict(booster: "xgb.Booster", dmat: "xgb.DMatrix") -> np.ndarray:
    """
    Try prediction using best iteration if available:
    1) iteration_range (newer)
    2) ntree_limit (older)
    3) plain predict (fallback)
    """
    best_it = getattr(booster, "best_iteration", None)
    best_nt = getattr(booster, "best_ntree_limit", None)

    # Newer API
    if best_it is not None:
        try:
            return booster.predict(dmat, iteration_range=(0, int(best_it) + 1))
        except TypeError:
            pass  # fall through

    # Older API
    if best_nt is not None:
        try:
            return booster.predict(dmat, ntree_limit=int(best_nt))
        except TypeError:
            pass  # fall through

    # Fallback: use all trees
    return booster.predict(dmat)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    cfg = load_config(args.conf)

    print("[xgb] reading silver…")
    df = read_silver(cfg.silver_dir)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    print(f"[xgb] sizes train={len(train):,} val={len(val):,} test={len(test):,}")

    Xtr, ytr, feat_names = make_xy(train)
    Xva, yva, _ = make_xy(val)
    Xte, yte, _ = make_xy(test)

    spw = scale_pos_weight(ytr)
    print(f"[xgb] scale_pos_weight={spw:.2f}")

    # Attempt sklearn wrapper first; if early stopping args fail, switch to DMatrix API
    model = XGBClassifier(
        tree_method="hist",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        eval_metric="aucpr",
        n_jobs=0,
        scale_pos_weight=spw,
    )

    print("[xgb] training…")
    try:
        # Some versions accept early_stopping_rounds directly
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False,
            early_stopping_rounds=50,
        )
        va_scores = model.predict_proba(Xva)[:, 1]
        te_scores = model.predict_proba(Xte)[:, 1]
        booster = model.get_booster()
    except TypeError:
        # Fall back to native Booster API (works across old versions)
        dtr = xgb.DMatrix(Xtr, label=ytr)
        dva = xgb.DMatrix(Xva, label=yva)
        dte = xgb.DMatrix(Xte, label=yte)
        params = {
            "tree_method": "hist",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "seed": args.seed,
            "scale_pos_weight": spw,
        }
        booster = xgb.train(
            params,
            dtr,
            num_boost_round=1000,
            evals=[(dva, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        va_scores = booster_predict(booster, dva)
        te_scores = booster_predict(booster, dte)

    # ---- Calibrate threshold on VAL and evaluate ----
    thr = calibrate_threshold(yva, va_scores, cfg.precision_target)
    print(f"[xgb] calibrated threshold (val precision>={cfg.precision_target:.2f}) -> {thr:.4f}")

    md = [
        "# XGBoost Metrics",
        "",
        f"- Features used: **{len(feat_names)}**",
        f"- Precision target (config): **{cfg.precision_target:.2f}**",
        "",
        metrics_block(yva, va_scores, "Validation", thr),
        metrics_block(yte, te_scores, "Test", thr),
    ]
    write_report(Path("reports/xgb_metrics.md"), "\n".join(md))
    print("[xgb] wrote reports/xgb_metrics.md")

    # write predictions (overwrites baseline gold)
    keep = ["timestamp", "service_id", "env", "split", TARGET]
    val_out = val[keep].copy();  val_out["score"] = va_scores
    test_out = test[keep].copy(); test_out["score"] = te_scores
    preds = pd.concat([val_out, test_out], ignore_index=True)
    write_preds(preds, cfg.gold_dir)
    print(f"[xgb] wrote predictions to {cfg.gold_dir}")

    # save model
    Path("models").mkdir(exist_ok=True)
    if hasattr(booster, "save_model"):
        booster.save_model("models/xgb_model.json")
    else:
        model.save_model("models/xgb_model.json")
    print("[xgb] saved model to models/xgb_model.json")
    print("[xgb] done.")

if __name__ == "__main__":
    main()
