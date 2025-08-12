# src/ingest/generate.py
"""
Generate synthetic telemetry data with incidents, optional drift, and a few intentional
data-quality glitches. Writes minute-level Parquet files partitioned by day/service.

CLI:
  python -m src.ingest.generate --conf conf/config.yaml --inject_drift true --glitches true
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from dateutil import tz
from tqdm import tqdm


# -----------------------------
# Config helpers
# -----------------------------
@dataclass
class Cfg:
    base_dir: Path
    raw_dir: Path
    services: int
    days: int
    freq: str
    env: str
    seed: int
    horizon_min: int
    incident_p95_ms: float
    incident_err_rate: float
    incident_min_dur: int


def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    data = y["data"]
    labels = y["labels"]
    project = y.get("project", {})

    return Cfg(
        base_dir=Path(data["base_dir"]),
        raw_dir=Path(data["raw_dir"]),
        services=int(data["services"]),
        days=int(data["days"]),
        freq=str(data["freq"]),
        env=str(data.get("env", "prod")),
        seed=int(project.get("seed", 42)),
        horizon_min=int(labels["prediction_horizon_min"]),
        incident_p95_ms=float(labels["incident_latency_p95_ms"]),
        incident_err_rate=float(labels["incident_error_rate"]),
        incident_min_dur=int(labels["incident_min_duration_min"]),
    )


# -----------------------------
# Time index & base frame
# -----------------------------
def make_time_index(cfg: Cfg) -> pd.DataFrame:
    # 30 days ending yesterday, minute-aligned UTC
    end = pd.Timestamp.utcnow().floor("D").tz_localize("UTC")
    start = end - pd.Timedelta(days=cfg.days)
    idx = pd.date_range(start=start, end=end - pd.Timedelta(minutes=1), freq=cfg.freq, tz="UTC")

    service_ids = [f"svc_{i:03d}" for i in range(1, cfg.services + 1)]
    df = (
        pd.DataFrame({"timestamp": idx})
        .assign(key=1)
        .merge(pd.DataFrame({"service_id": service_ids, "key": 1}), on="key")
        .drop(columns="key")
    )
    df["env"] = cfg.env
    return df


# -----------------------------
# Normal metrics simulation
# -----------------------------
def simulate_normal_metrics(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # Diurnal + weekly factors
    ts = df["timestamp"].dt
    hour = ts.hour + ts.minute / 60.0
    # smooth day curve in [0.4, 1.8]
    diurnal = 1.1 + 0.7 * np.sin(2 * np.pi * (hour - 6) / 24.0)
    diurnal = np.clip(diurnal, 0.4, 1.8)
    weekday = np.where(ts.dayofweek < 5, 1.0, 0.85)

    # Per-service base load (requests/min)
    svc_hash = df["service_id"].str[-3:].astype(int).to_numpy()
    base_lambda = 140 + (svc_hash % 50) * 3  # 140..289 baseline spread
    base_lambda = base_lambda.astype(float)

    lam = base_lambda * diurnal * weekday
    lam = np.clip(lam, 5.0, None)
    request_count = rng.poisson(lam)

    # Baseline probabilities for status codes
    p5xx_base = 0.002 + 0.001 * (svc_hash % 3)  # 0.002..0.004
    p4xx_base = 0.010 + 0.002 * ((svc_hash // 3) % 3)  # 0.010..0.014
    p5xx = np.clip(p5xx_base + rng.normal(0, 0.0005, size=len(df)), 0, 0.05)
    p4xx = np.clip(p4xx_base + rng.normal(0, 0.0010, size=len(df)), 0, 0.07)

    # CPU & latency scale with load; add noise
    load_ratio = request_count / (lam + 1e-9)
    cpu_pct = 20 + 60 * load_ratio + rng.normal(0, 5, size=len(df))
    cpu_pct = np.clip(cpu_pct, 1, 99)
    mem_pct = 40 + 10 * (svc_hash % 5) + rng.normal(0, 3, size=len(df))
    mem_pct = np.clip(mem_pct, 10, 95)

    latency_p50 = 300 + 1.5 * cpu_pct + rng.normal(0, 20, size=len(df))
    surge = np.maximum(0, 0.2 * (request_count - lam))  # extra when traffic > expected
    latency_p95 = latency_p50 + 200 + 0.8 * cpu_pct + 0.5 * surge + rng.normal(0, 30, size=len(df))

    # IO & network roughly proportional to traffic
    disk_io = np.maximum(0, rng.normal(2.0, 0.5, size=len(df)) + 0.002 * request_count)
    net_in = np.maximum(0, rng.normal(3.0, 0.8, size=len(df)) + 0.003 * request_count)
    net_out = np.maximum(0, rng.normal(3.0, 0.8, size=len(df)) + 0.003 * request_count)

    out = df.copy()
    out["request_count"] = request_count.astype("int32")
    out["cpu_pct"] = cpu_pct.astype("float32")
    out["mem_pct"] = mem_pct.astype("float32")
    out["latency_p50_ms"] = latency_p50.astype("float32")
    out["latency_p95_ms"] = latency_p95.astype("float32")
    out["disk_io_mb_s"] = disk_io.astype("float32")
    out["net_in_mb_s"] = net_in.astype("float32")
    out["net_out_mb_s"] = net_out.astype("float32")
    out["p5xx"] = p5xx.astype("float32")
    out["p4xx"] = p4xx.astype("float32")
    return out


# -----------------------------
# Failure injection (5 modes)
# -----------------------------
def _pick_incidents_per_service(days: int, rng: np.random.Generator) -> int:
    # ~0.5 incidents/service/day on average
    return int(rng.poisson(0.5 * days))


def inject_failures(df: pd.DataFrame, cfg: Cfg, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    minutes_per_day = 24 * 60
    svc_groups = out.groupby("service_id", sort=False)

    for svc, gidx in tqdm(svc_groups.indices.items(), desc="inject_failures", leave=False):
        n = _pick_incidents_per_service(cfg.days, rng)
        if n == 0:
            continue
        idx = out.index[gidx]

        for _ in range(n):
            # pick a random start minute leaving room for duration
            dur = int(rng.integers(cfg.incident_min_dur, cfg.incident_min_dur + 26))  # 5..30
            start_pos = int(rng.integers(0, len(idx) - dur - 1))
            span = idx[start_pos : start_pos + dur]

            mode = int(rng.integers(1, 6))  # 1..5
            # Apply effects
            if mode == 1:  # Traffic surge → queueing
                out.loc[span, "request_count"] = (out.loc[span, "request_count"] * 1.8).astype("int32")
                out.loc[span, "latency_p95_ms"] += 450
                out.loc[span, "latency_p50_ms"] += 180
                out.loc[span, "cpu_pct"] = np.clip(out.loc[span, "cpu_pct"] + 20, 0, 100)
                out.loc[span, "p5xx"] += 0.010
            elif mode == 2:  # CPU saturation (hot path)
                out.loc[span, "cpu_pct"] = np.clip(out.loc[span, "cpu_pct"] + 30, 0, 100)
                out.loc[span, "latency_p95_ms"] += 520
                out.loc[span, "latency_p50_ms"] += 220
                out.loc[span, "p5xx"] += 0.006
            elif mode == 3:  # Memory leak
                leak = np.linspace(0, 30, len(span))
                out.loc[span, "mem_pct"] = np.clip(out.loc[span, "mem_pct"] + leak, 0, 100)
                out.loc[span, "latency_p95_ms"] += 480
                out.loc[span, "latency_p50_ms"] += 180
                out.loc[span, "p5xx"] += 0.015
            elif mode == 4:  # Dependency flakiness (5xx burst)
                out.loc[span, "p5xx"] += 0.050
                out.loc[span, "latency_p95_ms"] += 260
            elif mode == 5:  # Network degradation
                out.loc[span, "latency_p95_ms"] += 380
                out.loc[span, "latency_p50_ms"] += 140
                out.loc[span, "p5xx"] += 0.010
                # jitter network
                out.loc[span, "net_in_mb_s"] *= 0.8 + 0.4 * rng.random(len(span))
                out.loc[span, "net_out_mb_s"] *= 0.8 + 0.4 * rng.random(len(span))

    # keep probabilities sane
    out["p5xx"] = np.clip(out["p5xx"], 0, 0.40)
    out["p4xx"] = np.clip(out["p4xx"], 0, 0.20)

    # Positive numeric constraints
    out["latency_p50_ms"] = np.maximum(out["latency_p50_ms"], 1)
    out["latency_p95_ms"] = np.maximum(out["latency_p95_ms"], out["latency_p50_ms"] + 1)
    return out


# -----------------------------
# Derive status, errors, rates
# -----------------------------
def derive_status_and_errors(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    rc = out["request_count"].to_numpy()

    # Draw 5xx then 4xx, rest are 2xx
    p5 = out["p5xx"].to_numpy()
    s5 = rng.binomial(rc, np.clip(p5, 0, 1))
    rem = rc - s5

    p4 = out["p4xx"].to_numpy()
    s4 = rng.binomial(rem, np.clip(p4, 0, 1))
    s2 = rem - s4

    out["status_5xx"] = s5.astype("int32")
    out["status_4xx"] = s4.astype("int32")
    out["status_2xx"] = s2.astype("int32")

    error_count = s4 + s5
    out["error_count"] = error_count.astype("int32")
    out["error_rate"] = np.where(rc > 0, error_count / rc, 0.0).astype("float32")
    return out


# -----------------------------
# Label incidents (start + horizon)
# -----------------------------
def _label_one_service(svc_df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    x = svc_df.copy()
    bad = (x["latency_p95_ms"] > cfg.incident_p95_ms) | (x["error_rate"] >= cfg.incident_err_rate)

    # Find runs of consecutive bad minutes length >= min_dur, mark the first as incident_start
    incident_start = np.zeros(len(x), dtype=bool)
    i = 0
    while i < len(x):
        if not bad.iloc[i]:
            i += 1
            continue
        # start of a bad run
        j = i
        while j < len(x) and bad.iloc[j]:
            j += 1
        run_len = j - i
        if run_len >= cfg.incident_min_dur:
            incident_start[i] = True
        i = j  # continue after the run

    x["incident_start"] = incident_start

    # y_label_15min: 1 if an incident starts within [t, t+15]
    # Compute next-incident start index using a rolling window
    horizon = cfg.horizon_min
    starts_idx = np.where(incident_start)[0]
    y = np.zeros(len(x), dtype=bool)
    if len(starts_idx) > 0:
        # For each t, check if any start index in (t .. t+horizon]
        # Efficient approach: create an array of next start positions
        next_start = np.full(len(x), np.inf)
        prev = np.inf
        for k in reversed(starts_idx):
            next_start[: k + 1] = np.minimum(next_start[: k + 1], k)
        # Now for each t, y=1 if next_start[t] - t <= horizon
        idxs = np.arange(len(x))
        dist = next_start - idxs
        y = dist <= horizon
        y[np.isinf(dist)] = False
    x["y_label_15min"] = y
    return x


def label_incidents(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("service_id", sort=False):
        parts.append(_label_one_service(g, cfg))
    return pd.concat(parts, ignore_index=True)


# -----------------------------
# Drift injectors
# -----------------------------
def apply_drift(df: pd.DataFrame, cfg: Cfg, rng: np.random.Generator, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df
    out = df.copy()
    start_day = out["timestamp"].dt.floor("D").min()
    day_idx = (out["timestamp"].dt.floor("D") - start_day).dt.days + 1

    # Concept drift Day >= 20: latency spikes uncorrelated with CPU
    mask_cd = day_idx >= 20
    # random 10% of minutes after Day 20 get extra p95 +250 (no cpu change)
    cd_sel = mask_cd & (rng.random(len(out)) < 0.10)
    out.loc[cd_sel, "latency_p95_ms"] += 250

    # Data drift Day >= 25: evening traffic ↑, latency baseline +100
    mask_dd = (day_idx >= 25) & (out["timestamp"].dt.hour >= 18)
    out.loc[mask_dd, "request_count"] = (out.loc[mask_dd, "request_count"] * 1.2).astype("int32")
    out.loc[mask_dd, "latency_p50_ms"] += 60
    out.loc[mask_dd, "latency_p95_ms"] += 100

    # keep monotonic latency constraint
    out["latency_p95_ms"] = np.maximum(out["latency_p95_ms"], out["latency_p50_ms"] + 1)
    return out


# -----------------------------
# Data-quality glitches
# -----------------------------
def inject_glitches(df: pd.DataFrame, cfg: Cfg, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    start_day = out["timestamp"].dt.floor("D").min()
    day_idx = (out["timestamp"].dt.floor("D") - start_day).dt.days + 1

    # Day 10: negative disk IO for 3 mins on one random service around midday
    if (day_idx.max() >= 10) and (out["service_id"].nunique() > 0):
        svc = rng.choice(out["service_id"].unique())
        t_day10 = start_day + pd.Timedelta(days=9) + pd.Timedelta(hours=12)
        mask = (
            (out["service_id"] == svc)
            & (out["timestamp"] >= t_day10)
            & (out["timestamp"] < t_day10 + pd.Timedelta(minutes=3))
        )
        out.loc[mask, "disk_io_mb_s"] = -1.0  # invalid

    # Day 18: remove the minute HH:17 at 12:17 for 5 services
    if day_idx.max() >= 18:
        services = rng.choice(out["service_id"].unique(), size=min(5, out["service_id"].nunique()), replace=False)
        t_gap = start_day + pd.Timedelta(days=17) + pd.Timedelta(hours=12, minutes=17)
        drop_mask = (out["timestamp"] == t_gap) & (out["service_id"].isin(services))
        out = out.loc[~drop_mask].copy()

    # Day 24: counts mismatch on a few rows (status sum ≠ request_count)
    if day_idx.max() >= 24:
        day24 = start_day + pd.Timedelta(days=23)
        sel = (out["timestamp"].dt.floor("D") == day24)
        idx = out.index[sel]
        if len(idx) > 0:
            bad_rows = rng.choice(idx, size=min(20, len(idx)), replace=False)
            out.loc[bad_rows, "status_2xx"] += 1  # break equality
            # keep error_count unchanged so error_rate will no longer equal formula
    return out


# -----------------------------
# Write Parquet partitioned
# -----------------------------
def write_parquet(df: pd.DataFrame, raw_dir: Path) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    # Ensure dtypes
    dtype_map = {
        "service_id": "string",
        "env": "string",
        "request_count": "int32",
        "error_count": "int32",
        "error_rate": "float32",
        "latency_p50_ms": "float32",
        "latency_p95_ms": "float32",
        "cpu_pct": "float32",
        "mem_pct": "float32",
        "disk_io_mb_s": "float32",
        "net_in_mb_s": "float32",
        "net_out_mb_s": "float32",
        "status_2xx": "int32",
        "status_4xx": "int32",
        "status_5xx": "int32",
        "incident_start": "bool",
        "y_label_15min": "bool",
    }
    for c, t in dtype_map.items():
        df[c] = df[c].astype(t)

    # Partition by day and service
    for (date_str, svc), part in tqdm(df.groupby(["date", "service_id"], sort=False), desc="write_parquet"):
        out_dir = raw_dir / f"date={date_str}" / f"service_id={svc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "part-0000.parquet").unlink(missing_ok=True)
        part.drop(columns=["date"], inplace=False).to_parquet(out_dir / "part-0000.parquet", index=False)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True, help="Path to YAML config")
    ap.add_argument("--inject_drift", type=str, default="true")
    ap.add_argument("--glitches", type=str, default="true")
    args = ap.parse_args()

    cfg = load_config(args.conf)
    rng = np.random.default_rng(cfg.seed)

    print("[generate] building time index…")
    df = make_time_index(cfg)

    print("[generate] simulating normal metrics…")
    df = simulate_normal_metrics(df, rng)

    print("[generate] injecting failures…")
    df = inject_failures(df, cfg, rng)

    print("[generate] deriving status & error rates…")
    df = derive_status_and_errors(df, rng)

    use_drift = str(args.inject_drift).lower() in {"1", "true", "yes", "y"}
    if use_drift:
        print("[generate] applying drift scenarios…")
        df = apply_drift(df, cfg, rng, enabled=True)

        # Re-derive status & error rate after drift (latency drift only keeps counts; that's OK)

    print("[generate] labeling incidents & horizon…")
    df = label_incidents(df, cfg)

    use_glitches = str(args.glitches).lower() in {"1", "true", "yes", "y"}
    if use_glitches:
        print("[generate] injecting data-quality glitches…")
        df = inject_glitches(df, cfg, rng)

    # Keep only public columns (drop p4xx/p5xx internals)
    keep_cols = [
        "timestamp",
        "service_id",
        "env",
        "request_count",
        "error_count",
        "error_rate",
        "latency_p50_ms",
        "latency_p95_ms",
        "cpu_pct",
        "mem_pct",
        "disk_io_mb_s",
        "net_in_mb_s",
        "net_out_mb_s",
        "status_2xx",
        "status_4xx",
        "status_5xx",
        "incident_start",
        "y_label_15min",
    ]
    df = df[keep_cols].sort_values(["timestamp", "service_id"]).reset_index(drop=True)

    print("[generate] writing Parquet partitions…")
    write_parquet(df, cfg.raw_dir)

    total_rows = len(df)
    services = df["service_id"].nunique()
    positives = int(df["y_label_15min"].sum())
    pos_rate = positives / total_rows if total_rows else 0.0
    print(f"[generate] done. rows={total_rows:,} services={services} pos_rate={pos_rate:.4f}")


if __name__ == "__main__":
    main()
