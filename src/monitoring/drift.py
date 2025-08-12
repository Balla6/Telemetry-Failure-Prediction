# src/monitoring/drift.py
"""
Compute drift between reference (VAL) and current (TEST):
- Feature drift (PSI) on silver features
- Score drift (PSI) on gold predictions
- Writes a concise markdown report: reports/drift_report.md

Run:
  export PYTHONPATH=.
  python src/monitoring/drift.py --conf conf/config.yaml
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

TARGET = "y_label_15min"
KEYS = ["timestamp","service_id","env","split","day_idx"]

# ---------------- Config ----------------
@dataclass
class Cfg:
    silver_dir: Path
    gold_dir: Path
    psi_warn: float
    psi_alert: float

def load_cfg(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return Cfg(
        silver_dir=Path(y["data"]["silver_dir"]),
        gold_dir=Path(y["data"]["gold_dir"]),
        psi_warn=float(y["drift"].get("psi_warn", 0.1)),
        psi_alert=float(y["drift"].get("psi_alert", 0.25)),
    )

# ---------------- IO ----------------
def _read_parquets(root: Path, pattern="part-*.parquet") -> pd.DataFrame:
    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files under {root}")
    parts = []
    for p in tqdm(files, desc=f"read {root}", unit="file"):
        df = pd.read_parquet(p)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.loc[:, ~out.columns.duplicated()]

def read_silver(silver_dir: Path) -> pd.DataFrame:
    return _read_parquets(silver_dir)

def read_gold(gold_dir: Path) -> pd.DataFrame:
    return _read_parquets(gold_dir)

# ---------------- PSI ----------------
def _clip_probs(a: np.ndarray) -> np.ndarray:
    # avoid div-by-zero/inf in PSI
    eps = 1e-6
    a = a.astype(float)
    a[a < eps] = eps
    return a

def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins from ref."""
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return np.nan
    # quantile bin edges from reference
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, qs))
    # fallback if not enough unique edges
    if len(edges) < 3:
        edges = np.linspace(ref.min(), ref.max() + 1e-9, bins + 1)
    r_hist, _ = np.histogram(ref, bins=edges)
    c_hist, _ = np.histogram(cur, bins=edges)
    r_prob = _clip_probs(r_hist / max(1, r_hist.sum()))
    c_prob = _clip_probs(c_hist / max(1, c_hist.sum()))
    return float(np.sum((c_prob - r_prob) * np.log(c_prob / r_prob)))

# ---------------- Report helpers ----------------
def to_md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_csv(index=False) + "\n```"

def write_report(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.conf)

    # 1) Feature drift on SILVER (VAL vs TEST)
    silver = read_silver(cfg.silver_dir)
    # Ensure splits
    silver["split"] = silver["split"].astype(str)
    val_s = silver[silver["split"] == "val"].copy()
    test_s = silver[silver["split"] == "test"].copy()

    # Choose numeric feature columns (exclude keys/labels)
    num_cols = silver.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    drop_cols = set([TARGET, "incident_start", "status_2xx","status_4xx","status_5xx"])  # optional
    feat_cols = [c for c in num_cols if c not in drop_cols]

    rows = []
    for col in feat_cols:
        try:
            score = psi(val_s[col].to_numpy(dtype=float), test_s[col].to_numpy(dtype=float), bins=10)
        except Exception:
            score = np.nan
        level = "ok"
        if pd.notna(score):
            if score >= cfg.psi_alert:
                level = "ALERT"
            elif score >= cfg.psi_warn:
                level = "WARN"
        rows.append({"feature": col, "psi": round(score if pd.notna(score) else np.nan, 4), "level": level})
    feat_df = pd.DataFrame(rows).sort_values(["level","psi"], ascending=[False, False])

    # 2) Score drift on GOLD (VAL vs TEST)
    gold = read_gold(cfg.gold_dir)
    gold["split"] = gold["split"].astype(str)
    val_g = gold[gold["split"] == "val"]["score"].to_numpy(dtype=float)
    test_g = gold[gold["split"] == "test"]["score"].to_numpy(dtype=float)
    score_psi = psi(val_g, test_g, bins=10)

    # Summaries
    n_warn = int((feat_df["level"] == "WARN").sum())
    n_alert = int((feat_df["level"] == "ALERT").sum())
    top_drift = feat_df.head(25)

    md = []
    md += ["# Drift Report", ""]
    md += [f"- PSI thresholds: WARN ≥ **{cfg.psi_warn}**, ALERT ≥ **{cfg.psi_alert}**", ""]
    md += [f"**Score PSI (VAL→TEST)**: **{0.0 if pd.isna(score_psi) else round(score_psi,4)}**", ""]
    md += [f"**Feature drift summary**: ALERT={n_alert}  |  WARN={n_warn}  |  Total checked={len(feat_df)}", ""]
    md += ["## Top drifting features (VAL→TEST)", "", to_md_table(top_drift), ""]
    md += ["## All features (sorted by severity)", "", to_md_table(feat_df), ""]
    write_report(Path("reports/drift_report.md"), "\n".join(md))

    print("[drift] wrote reports/drift_report.md")
    print(f"[drift] score PSI: {score_psi:.4f}" if pd.notna(score_psi) else "[drift] score PSI: nan")
    print(f"[drift] features: ALERT={n_alert} WARN={n_warn} / {len(feat_df)}")

if __name__ == "__main__":
    main()
