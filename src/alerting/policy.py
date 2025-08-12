# src/alerting/policy.py
"""
Alert policy from gold predictions.
- Reads data/gold predictions (VAL + TEST)
- Calibrates threshold on VAL for precision >= target (unless --threshold is given)
- Applies cooldown on TEST and computes precision/recall/volume based on fired alerts
- Writes alerts preview and policy report

CLI:
  # auto-calibrate from VAL
  python -m src.alerting.policy --conf conf/config.yaml --cooldown_min 10 --precision_target 0.80

  # or lock a manual threshold
  python -m src.alerting.policy --conf conf/config.yaml --cooldown_min 15 --threshold 0.976
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score

TARGET = "y_label_15min"


@dataclass
class Cfg:
    gold_dir: Path
    report_path: Path
    precision_target: float


def load_config(path: str, precision_target: float | None) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    data = y["data"]
    alert = y.get("alerting", {})
    pt = precision_target if precision_target is not None else float(alert.get("precision_target", 0.8))
    return Cfg(
        gold_dir=Path(data["gold_dir"]),
        report_path=Path("reports/alert_policy.md"),
        precision_target=pt,
    )


def read_gold(gold_dir: Path) -> pd.DataFrame:
    files = sorted(Path(gold_dir).rglob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No gold predictions under {gold_dir}")
    parts = []
    for p in tqdm(files, desc="read_gold", unit="file"):
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["split"] = df["split"].astype(str)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    # drop any duplicated columns defensively
    df = df.loc[:, ~df.columns.duplicated()]

    need = {"timestamp", "service_id", "split", TARGET, "score"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"gold predictions missing columns: {missing}")
    return df


def calibrate_threshold(y_true: np.ndarray, scores: np.ndarray, precision_target: float) -> float:
    """Smallest threshold achieving >= precision_target on VAL; 1.0 if never achieved."""
    p, r, th = precision_recall_curve(y_true, scores)
    thr = 1.0
    for prec, t in zip(p, np.append(th, 1.0)):
        if prec >= precision_target:
            thr = min(thr, float(t))
    return float(thr)


def apply_cooldown(df: pd.DataFrame, cooldown_min: int, thr: float) -> pd.DataFrame:
    """Return rows where we fire an alert given threshold & cooldown, per service."""
    df = df.sort_values(["service_id", "timestamp"]).copy()
    df["alert"] = 0
    last_ts: dict[str, pd.Timestamp] = {}
    for i, row in df.iterrows():
        if row["score"] >= thr:
            svc = row["service_id"]
            ts = row["timestamp"]
            lt = last_ts.get(svc)
            if lt is None or (ts - lt).total_seconds() >= cooldown_min * 60:
                df.at[i, "alert"] = 1
                last_ts[svc] = ts
    return df


def split_sets(df: pd.DataFrame):
    return (
        df[df["split"] == "val"].reset_index(drop=True),
        df[df["split"] == "test"].reset_index(drop=True),
    )


def event_metrics(df: pd.DataFrame, thr: float, use_alert_column: bool = False) -> tuple[float, float, int]:
    """
    Precision/recall either at the row level (score >= thr) or using 'alert' column if use_alert_column=True.
    """
    if use_alert_column and "alert" in df.columns:
        fired = df["alert"] == 1
    else:
        fired = df["score"] >= thr

    tp = int(((df[TARGET] == 1) & fired).sum())
    fp = int(((df[TARGET] == 0) & fired).sum())
    fn = int(((df[TARGET] == 1) & (~fired)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec, int(fired.sum())


def df_to_md(df: pd.DataFrame, index=False) -> str:
    """Robust markdown table: try to_markdown; if tabulate missing, fallback to CSV fenced."""
    try:
        return df.to_markdown(index=index)
    except Exception:
        return "```\n" + df.to_csv(index=index) + "\n```"


def write_report(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--cooldown_min", type=int, default=10)
    ap.add_argument("--precision_target", type=float, default=None,
                    help="override config alerting.precision_target")
    ap.add_argument("--threshold", type=float, default=None,
                    help="manual threshold override; if omitted, auto-calibrate on VAL")
    args = ap.parse_args()

    cfg = load_config(args.conf, args.precision_target)
    print(f"[policy] reading gold predictions from {cfg.gold_dir} …")
    gold = read_gold(cfg.gold_dir)

    val, test = split_sets(gold)
    print(f"[policy] sizes val={len(val):,} test={len(test):,}")

    # Choose threshold: manual override OR auto-calibrate on VAL
    if args.threshold is None:
        thr = calibrate_threshold(val[TARGET].to_numpy(), val["score"].to_numpy(), cfg.precision_target)
        source = f"auto-calibrated from VAL @ precision>={cfg.precision_target:.2f}"
    else:
        thr = float(args.threshold)
        source = "manual override"

    ap_val = average_precision_score(val[TARGET], val["score"])
    p_val, r_val, n_alerts_val = event_metrics(val, thr, use_alert_column=False)
    print(f"[policy] threshold = {thr:.4f} ({source})")
    print(f"[policy] VAL: PR-AUC={ap_val:.4f}  P={p_val:.3f}  R={r_val:.3f}  alerts={n_alerts_val:,}")

    # Apply cooldown on TEST and compute metrics on the fired alerts
    test_cd = apply_cooldown(test, args.cooldown_min, thr)
    p_te, r_te, n_alerts_te = event_metrics(test_cd, thr, use_alert_column=True)

    # Per-day volume (number of fired alerts)
    per_day = (
        test_cd.assign(date=test_cd["timestamp"].dt.strftime("%Y-%m-%d"))
        .groupby("date")["alert"].sum().reset_index().rename(columns={"alert": "alerts"})
        .sort_values("date")
    )

    # Preview top alerts (by score) that actually fired
    preview = (
        test_cd.loc[test_cd["alert"] == 1, ["timestamp", "service_id", "score", TARGET]]
        .sort_values("score", ascending=False)
        .head(50)
    )

    md = []
    md += ["# Alert Policy Report", ""]
    md += [
        f"- Threshold: **{thr:.4f}**  ({source})",
        f"- Cooldown: **{args.cooldown_min} min**",
        f"- Precision target (config): **{cfg.precision_target:.2f}**",
        "",
    ]
    md += [
        "## Validation",
        f"- PR-AUC: **{ap_val:.4f}**",
        f"- Precision: **{p_val:.3f}**  |  Recall: **{r_val:.3f}**  |  Alerts: **{n_alerts_val:,}**",
        "",
    ]
    md += [
        "## Test (policy view)",
        f"- Precision: **{p_te:.3f}**  |  Recall: **{r_te:.3f}**  |  Alerts fired: **{n_alerts_te:,}**",
        "",
    ]
    md += ["## Alerts per day (TEST)", "", df_to_md(per_day, index=False), ""]
    md += ["## Top 50 alerts (TEST)", "", df_to_md(preview, index=False), ""]
    write_report(cfg.report_path, "\n".join(md))
    print(f"[policy] wrote {cfg.report_path}")
    print("[policy] done.")


if __name__ == "__main__":
    main()
