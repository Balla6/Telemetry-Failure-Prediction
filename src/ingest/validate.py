# src/ingest/validate.py
"""
Validate raw telemetry parquet files against data contracts.
Good rows -> data/bronze/ (same partitions)
Bad rows  -> data/bronze/_quarantine/ (with a 'reason' column)
Also writes reports/data_quality_report.md

CLI:
  python -m src.ingest.validate --conf conf/config.yaml --quarantine true
(or)
  python src/ingest/validate.py --conf conf/config.yaml --quarantine true
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    raw_dir: Path
    bronze_dir: Path

def load_config(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    data = y["data"]
    return Cfg(
        raw_dir=Path(data["raw_dir"]),
        bronze_dir=Path(data["bronze_dir"]),
    )


# -----------------------------
# Read raw partitions
# -----------------------------
REQUIRED_COLS = [
    "timestamp","service_id","env",
    "request_count","error_count","error_rate",
    "latency_p50_ms","latency_p95_ms",
    "cpu_pct","mem_pct","disk_io_mb_s","net_in_mb_s","net_out_mb_s",
    "status_2xx","status_4xx","status_5xx",
    "incident_start","y_label_15min",
]

def iter_raw_parts(raw_dir: Path) -> Iterable[Tuple[str,str,pd.DataFrame,Path]]:
    """
    Yield (date_str, service_id, df, path) for each raw partition file.
    Layout: data/raw/date=YYYY-MM-DD/service_id=svc_###/part-0000.parquet
    """
    for p in sorted(Path(raw_dir).rglob("part-*.parquet")):
        # parse date and service from parent directories
        date_str = p.parent.parent.name.split("date=")[-1]
        service_id = p.parent.name.split("service_id=")[-1]
        df = pd.read_parquet(p)
        yield date_str, service_id, df, p


# -----------------------------
# Row-level checks
# -----------------------------
def check_and_tag_bad_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str,int]]:
    """
    Returns (good_df, bad_df, reason_counts)
    Adds per-row 'reason' for bad rows (semicolon-joined if multiple).
    """
    x = df.copy()
    reasons = []

    # Start every row as good
    bad_mask = np.zeros(len(x), dtype=bool)

    def add_reason(mask: np.ndarray, label: str):
        nonlocal bad_mask, reasons
        if mask.any():
            bad_mask |= mask
            reasons.append((label, int(mask.sum())))
            if "reason" not in x:
                x["reason"] = ""
            # append reason strings (preserve existing)
            x.loc[mask, "reason"] = np.where(
                x.loc[mask, "reason"].astype(str).str.len() > 0,
                x.loc[mask, "reason"] + ";" + label,
                label,
            )

    # Required columns present?
    missing = [c for c in REQUIRED_COLS if c not in x.columns]
    if missing:
        # If schema is broken, tag all rows with 'missing_columns'
        add_reason(np.ones(len(x), dtype=bool), f"missing_columns:{','.join(missing)}")
        # continue but remaining checks may fail gracefully

    # Equality/integrity checks
    counts_eq = x.get("request_count", 0) == (x.get("status_2xx", 0) + x.get("status_4xx", 0) + x.get("status_5xx", 0))
    add_reason(~counts_eq.to_numpy(), "counts_mismatch")

    err_approx = np.abs(x.get("error_count", 0) - (x.get("status_4xx", 0) + x.get("status_5xx", 0))) <= 1
    add_reason(~err_approx.to_numpy(), "error_count_mismatch")

    # error_rate equality (within tiny tolerance)
    denom = np.maximum(x.get("request_count", 0).astype(float), 1.0)
    calc_rate = (x.get("error_count", 0).astype(float) / denom).astype(float)
    rate_ok = np.abs(x.get("error_rate", 0).astype(float) - calc_rate) <= 1e-6
    add_reason(~rate_ok.to_numpy(), "error_rate_mismatch")

    # latency monotonic
    lat_ok = x.get("latency_p95_ms", 0).astype(float) >= x.get("latency_p50_ms", 0).astype(float)
    add_reason(~lat_ok.to_numpy(), "latency_p95_lt_p50")

    # Ranges
    def in_range(series, lo, hi):
        return (series.astype(float) >= lo) & (series.astype(float) <= hi)

    add_reason(~in_range(x.get("cpu_pct", 0), 0, 100).to_numpy(), "cpu_out_of_range")
    add_reason(~in_range(x.get("mem_pct", 0), 0, 100).to_numpy(), "mem_out_of_range")

    nonneg_cols = ["request_count","error_count","disk_io_mb_s","net_in_mb_s","net_out_mb_s",
                   "status_2xx","status_4xx","status_5xx","latency_p50_ms","latency_p95_ms"]
    for c in nonneg_cols:
        add_reason((x.get(c, 0).astype(float) < 0).to_numpy(), f"{c}_negative")

    # Build outputs
    bad_df = x.loc[bad_mask].copy()
    good_df = x.loc[~bad_mask].copy()

    # Count reasons
    reason_counts: Dict[str,int] = Counter()
    if "reason" in bad_df:
        for r in bad_df["reason"].astype(str):
            for tag in r.split(";"):
                reason_counts[tag] += 1

    return good_df, bad_df, reason_counts


# -----------------------------
# Continuity summary (non-row)
# -----------------------------
def continuity_summary(df: pd.DataFrame) -> Dict[str,int]:
    """
    Reports number of gaps/dupes in timestamp sequence (per partition).
    We don't quarantine for gaps (missing rows); we just report them.
    """
    if df.empty:
        return {"gaps": 0, "duplicates": 0}
    s = df.sort_values("timestamp")
    diffs = s["timestamp"].diff().dropna()
    gaps = int((diffs != pd.Timedelta(minutes=1)).sum())
    # duplicates would show as zero timedelta
    duplicates = int((diffs == pd.Timedelta(0)).sum())
    return {"gaps": gaps, "duplicates": duplicates}


# -----------------------------
# Write partitions
# -----------------------------
def write_partitions(df: pd.DataFrame, out_base: Path, quarantine: bool=False):
    """
    Writes df to partition folders by date/service_id under out_base.
    If quarantine=True, write under out_base/_quarantine/...
    """
    if df.empty:
        return
    base = Path(out_base)
    if quarantine:
        base = base / "_quarantine"
    base.mkdir(parents=True, exist_ok=True)

    dfx = df.copy()
    dfx["date"] = dfx["timestamp"].dt.strftime("%Y-%m-%d")
    for (date_str, svc), part in dfx.groupby(["date", "service_id"], sort=False):
        out_dir = base / f"date={date_str}" / f"service_id={svc}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part-0000.parquet"
        if out_path.exists():
            out_path.unlink()
        part.drop(columns=["date"], inplace=False).to_parquet(out_path, index=False)


# -----------------------------
# Report
# -----------------------------
def write_report(summary: Dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Data Quality Report\n")
    lines.append(f"**Total rows scanned:** {summary['rows_total']:,}")
    lines.append(f"**Good rows to bronze:** {summary['rows_good']:,}")
    lines.append(f"**Quarantined rows:** {summary['rows_bad']:,}\n")

    # Reasons table
    lines.append("## Quarantine reasons\n")
    if summary["reason_counts"]:
        lines.append("| reason | count |")
        lines.append("|---|---:|")
        for reason, cnt in sorted(summary["reason_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {cnt:,} |")
        lines.append("")
    else:
        lines.append("_No quarantined rows._\n")

    # Continuity
    lines.append("## Continuity summary (gaps/duplicates)\n")
    if summary["continuity"]:
        lines.append("| partition (date/service) | gaps | duplicates |")
        lines.append("|---|---:|---:|")
        for key, val in summary["continuity"].items():
            lines.append(f"| {key} | {val['gaps']} | {val['duplicates']} |")
        lines.append("")
    else:
        lines.append("_No continuity issues detected._\n")

    # Label prevalence & incidents
    lines.append("## Labels & incidents\n")
    prev = summary.get("label_prevalence", 0.0)
    lines.append(f"- Overall positive label rate (`y_label_15min`): **{prev:.4%}**")
    lines.append("- Incident starts per service (top 10):")
    svc_counts = summary.get("incidents_per_service", {})
    for svc, cnt in sorted(svc_counts.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"  - {svc}: {cnt}")
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--quarantine", type=str, default="true")
    args = ap.parse_args()

    cfg = load_config(args.conf)
    do_quarantine = str(args.quarantine).lower() in {"1", "true", "yes", "y"}

    total_rows = 0
    good_rows = 0
    bad_rows = 0
    reason_counts = Counter()
    continuity = {}

    good_parts = []
    bad_parts = []

    # Read every raw partition
    for date_str, svc, df, p in tqdm(iter_raw_parts(cfg.raw_dir), desc="validate", unit="part"):
        # Ensure tz-aware datetimes (safe even if already correct)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        total_rows += len(df)

        # Per-partition continuity summary (gaps show up here)
        cont = continuity_summary(df)
        if cont["gaps"] or cont["duplicates"]:
            continuity[f"{date_str}/{svc}"] = cont

        # Row validations
        g, b, rc = check_and_tag_bad_rows(df)
        if len(g) > 0:
            good_parts.append(g)
        if len(b) > 0:
            bad_parts.append(b)
        good_rows += len(g)
        bad_rows += len(b)
        reason_counts.update(rc)

    # Concatenate and write out
    good_all = pd.concat(good_parts, ignore_index=True) if good_parts else pd.DataFrame(columns=REQUIRED_COLS)
    bad_all = pd.concat(bad_parts, ignore_index=True) if bad_parts else pd.DataFrame(columns=REQUIRED_COLS + ["reason"])

    write_partitions(good_all, cfg.bronze_dir, quarantine=False)
    if do_quarantine:
        write_partitions(bad_all, cfg.bronze_dir, quarantine=True)

    # Prevalence & incidents summary
    label_prev = float(good_all["y_label_15min"].mean()) if len(good_all) else 0.0
    incidents_per_svc = good_all.groupby("service_id")["incident_start"].sum().astype(int).to_dict()

    summary = {
        "rows_total": total_rows,
        "rows_good": good_rows,
        "rows_bad": bad_rows,
        "reason_counts": dict(reason_counts),
        "continuity": continuity,
        "label_prevalence": label_prev,
        "incidents_per_service": incidents_per_svc,
    }
    write_report(summary, Path("reports/data_quality_report.md"))

    print(f"[validate] done. total={total_rows:,} good={good_rows:,} bad={bad_rows:,} "
          f"prevalence={label_prev:.4%}. Report: reports/data_quality_report.md")


if __name__ == "__main__":
    main()
