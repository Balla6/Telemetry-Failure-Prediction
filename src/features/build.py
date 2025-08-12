# src/features/build.py
"""
Build time-series features from bronze telemetry and write to data/silver
(partitioned by day/service). Adds a 'split' column (train/val/test) per config.

CLI:
  python -m src.features.build --conf conf/config.yaml
  # or
  python src/features/build.py --conf conf/config.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    bronze_dir: Path
    silver_dir: Path
    windows: List[int]
    stats: List[str]
    split_train: List[int]  # [start_day, end_day]
    split_val: List[int]
    split_test: List[int]

def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    data = y["data"]
    feats = y["features"]
    splits = y["splits"]
    return Cfg(
        bronze_dir=Path(data["bronze_dir"]),
        silver_dir=Path(data["silver_dir"]),
        windows=[int(w) for w in feats["windows_min"]],
        stats=[str(s) for s in feats["stats"]],
        split_train=[int(splits["train_days"][0]), int(splits["train_days"][1])],
        split_val=[int(splits["val_days"][0]), int(splits["val_days"][1])],
        split_test=[int(splits["test_days"][0]), int(splits["test_days"][1])],
    )


# -----------------------------
# IO helpers
# -----------------------------
def read_bronze(bronze_dir: Path) -> pd.DataFrame:
    files = sorted(Path(bronze_dir).rglob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No bronze files found under {bronze_dir}")
    parts = []
    for p in tqdm(files, desc="read_bronze", unit="file"):
        df = pd.read_parquet(p)
        # Always coerce to tz-aware datetime (safe if already correct)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    keep = [
        "timestamp","service_id","env",
        "request_count","error_count","error_rate",
        "latency_p50_ms","latency_p95_ms",
        "cpu_pct","mem_pct","disk_io_mb_s","net_in_mb_s","net_out_mb_s",
        "status_2xx","status_4xx","status_5xx",
        "incident_start","y_label_15min",
    ]
    df = df[keep].sort_values(["service_id","timestamp"]).reset_index(drop=True)
    return df


def write_silver(df: pd.DataFrame, silver_dir: Path):
    silver_dir = Path(silver_dir)
    silver_dir.mkdir(parents=True, exist_ok=True)
    x = df.copy()
    # ensure 'split' is plain string and drop duplicate columns defensively
    x["split"] = x["split"].astype(str)
    x = x.loc[:, ~x.columns.duplicated()]
    x["date"] = x["timestamp"].dt.strftime("%Y-%m-%d")
    for (date_str, svc), part in tqdm(x.groupby(["date","service_id"], sort=False), desc="write_silver"):
        out_dir = silver_dir / f"date={date_str}" / f"service_id={svc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part-0000.parquet"
        if out_path.exists():
            out_path.unlink()
        part = part.loc[:, ~part.columns.duplicated()]
        part.drop(columns=["date"], inplace=False).to_parquet(out_path, index=False)


# -----------------------------
# Feature engineering
# -----------------------------
BASE_METRICS = ["latency_p95_ms", "latency_p50_ms", "error_rate", "cpu_pct", "mem_pct", "request_count"]

def _rolling_features(g: pd.DataFrame, windows: List[int], stats: List[str]) -> pd.DataFrame:
    """
    Compute rolling features per service. Uses only data up to time t (no future leakage).
    """
    g = g.sort_values("timestamp").copy()
    g["minute_idx"] = np.arange(len(g), dtype="int32")

    for w in windows:
        # pick safe min_periods (must be <= window)
        mp_mean_std = 1 if w == 1 else min(3, w)
        mp_q = 1 if w == 1 else min(5, w)
        mp_slope = 1 if w == 1 else min(5, w)

        # mean/std for core metrics
        if "mean" in stats:
            for col in BASE_METRICS:
                g[f"{col}_mean_{w}m"] = (
                    g[col].rolling(window=w, min_periods=mp_mean_std).mean()
                ).astype("float32")

        if "std" in stats:
            for col in BASE_METRICS:
                g[f"{col}_std_{w}m"] = (
                    g[col].rolling(window=w, min_periods=mp_mean_std).std()
                ).astype("float32")

        # p95 for a few key metrics
        if "p95" in stats:
            for col in ["latency_p95_ms", "error_rate"]:
                g[f"{col}_p95_{w}m"] = (
                    g[col].rolling(window=w, min_periods=mp_q).quantile(0.95, interpolation="linear")
                ).astype("float32")

        # slope (trend) for latency_p95 and error_rate
        if "slope" in stats:
            for col in ["latency_p95_ms", "error_rate"]:
                def slope_func(x: np.ndarray) -> float:
                    n = len(x)
                    if n < 3:
                        return 0.0
                    X = np.arange(n, dtype=np.float32)
                    vx = np.var(X)
                    if vx == 0:
                        return 0.0
                    return float(np.cov(X, x, bias=True)[0, 1] / vx)
                g[f"{col}_slope_{w}m"] = (
                    g[col].rolling(window=w, min_periods=mp_slope).apply(slope_func, raw=True)
                ).astype("float32")

        # delta (current - previous) for select metrics (compute once; same name each loop is fine)
        if "delta" in stats:
            for col in ["latency_p95_ms", "error_rate", "request_count", "cpu_pct"]:
                g[f"{col}_delta_1m"] = (g[col] - g[col].shift(1)).astype("float32")

    # calendar/time features
    dt = g["timestamp"].dt
    g["hour"] = dt.hour.astype("int16")
    g["dow"] = dt.dayofweek.astype("int16")
    g["is_weekend"] = (g["dow"] >= 5).astype("int8")

    # clean NA from rolling warmup
    feat_cols = [c for c in g.columns if any(s in c for s in ["_mean_", "_std_", "_p95_", "_slope_", "_delta_1m"])]
    g[feat_cols] = g[feat_cols].fillna(0)

    return g.drop(columns=["minute_idx"])


def add_splits(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    # day index starting at 1 from the earliest date
    day0 = df["timestamp"].dt.floor("D").min()
    day_idx = (df["timestamp"].dt.floor("D") - day0).dt.days + 1

    split = np.full(len(df), "train", dtype=object)
    split[(day_idx >= cfg.split_val[0]) & (day_idx <= cfg.split_val[1])] = "val"
    split[(day_idx >= cfg.split_test[0]) & (day_idx <= cfg.split_test[1])] = "test"
    df = df.copy()
    df["day_idx"] = day_idx.astype("int32")
    df["split"] = pd.Categorical(split, categories=["train", "val", "test"])
    return df


def build_features(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    out_parts = []
    for _, g in tqdm(df.groupby("service_id", sort=False), desc="features_by_service"):
        out_parts.append(_rolling_features(g, cfg.windows, cfg.stats))
    out = pd.concat(out_parts, ignore_index=True)

    # Select columns to keep in silver
    label_cols = ["y_label_15min", "incident_start"]
    keep_cols = [
        "timestamp","service_id","env","split","day_idx",
        *BASE_METRICS,
        "status_5xx","status_4xx","status_2xx",
        "disk_io_mb_s","net_in_mb_s","net_out_mb_s",
        *[c for c in out.columns if any(s in c for s in ["_mean_", "_std_", "_p95_", "_slope_", "_delta_1m"])],
        *label_cols,
    ]
    out = out[keep_cols]
    # drop any accidental duplicate columns
    out = out.loc[:, ~out.columns.duplicated()]
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    args = ap.parse_args()

    cfg = load_config(args.conf)

    print("[features] reading bronze…")
    bronze = read_bronze(cfg.bronze_dir)

    print("[features] adding split columns…")
    bronze = add_splits(bronze, cfg)

    print("[features] building rolling features…")
    feat = build_features(bronze, cfg)

    print("[features] writing silver partitions…")
    write_silver(feat, cfg.silver_dir)

    print(f"[features] done. rows={len(feat):,} services={feat['service_id'].nunique()} "
          f"cols={feat.shape[1]} -> {cfg.silver_dir}")


if __name__ == "__main__":
    main()
