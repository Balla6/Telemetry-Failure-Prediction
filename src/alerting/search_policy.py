# src/alerting/search_policy.py
# Grid-search thresholds & cooldowns to find a good alerting policy.

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score

TARGET = "y_label_15min"

@dataclass
class Cfg:
    gold_dir: Path

def load_cfg(conf_path: str) -> Cfg:
    with open(conf_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return Cfg(gold_dir=Path(y["data"]["gold_dir"]))

def read_gold(gold_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(gold_dir / "date=*/service_id=*/part-0000.parquet")))
    if not files:
        raise FileNotFoundError(f"No gold files under {gold_dir}")
    parts = []
    for p in tqdm(files, desc="read_gold", unit="file"):
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["split"] = df["split"].astype(str)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]
    need = {"timestamp","service_id","split",TARGET,"score"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in gold: {miss}")
    return df

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

def event_metrics(df: pd.DataFrame, thr: float):
    fired = df["score"] >= thr
    tp = int(((df[TARGET] == 1) & fired).sum())
    fp = int(((df[TARGET] == 0) & fired).sum())
    fn = int(((df[TARGET] == 1) & (~fired)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec, int(fired.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--thr_min", type=float, default=0.965)
    ap.add_argument("--thr_max", type=float, default=0.995)
    ap.add_argument("--thr_step", type=float, default=0.001)
    ap.add_argument("--cooldowns", type=int, nargs="+", default=[5,10,15,20,30])
    ap.add_argument("--min_precision", type=float, default=0.55)
    ap.add_argument("--max_alerts_per_day", type=float, default=20.0)
    ap.add_argument("--beta", type=float, default=0.5, help="F-beta to rank (beta<1 favours precision)")
    args = ap.parse_args()

    cfg = load_cfg(args.conf)
    df = read_gold(cfg.gold_dir)
    val = df[df["split"]=="val"].copy()
    test = df[df["split"]=="test"].copy()

    # prep per-day denominator for TEST
    test["date"] = test["timestamp"].dt.tz_convert(None).dt.strftime("%Y-%m-%d")
    days = test["date"].nunique()

    rows = []
    thresholds = np.arange(args.thr_min, args.thr_max + 1e-12, args.thr_step)
    for thr in thresholds:
        # val metrics (no cooldown)
        p_val, r_val, n_val = event_metrics(val, thr)
        for cd in args.cooldowns:
            test_cd = apply_cooldown(test, cd, thr)
            p_te, r_te, n_te = event_metrics(test_cd, thr)

            per_day = n_te / days if days else 0.0
            fbeta = (1 + args.beta**2) * (p_te*r_te) / (args.beta**2 * p_te + r_te + 1e-12)

            rows.append({
                "threshold": round(float(thr), 4),
                "cooldown_min": int(cd),
                "val_precision": round(p_val, 3),
                "val_recall": round(r_val, 3),
                "test_precision": round(p_te, 3),
                "test_recall": round(r_te, 3),
                "alerts_test_total": int(n_te),
                "alerts_test_per_day": round(per_day, 2),
                "fbeta_test": round(float(fbeta), 4),
            })

    res = pd.DataFrame(rows)

    # Apply constraints, then rank
    ok = res[
        (res["test_precision"] >= args.min_precision) &
        (res["alerts_test_per_day"] <= args.max_alerts_per_day)
    ].copy()

    if len(ok) == 0:
        print("\nNo combo met the constraints. Showing top by F-beta anyway:\n")
        top = res.sort_values(["fbeta_test","test_precision","test_recall"], ascending=False).head(20)
    else:
        top = ok.sort_values(["fbeta_test","test_precision","test_recall"], ascending=False).head(20)

    out_path = Path("reports/policy_grid.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    print(f"\nWrote full grid to {out_path} (rows={len(res)})")
    print("\nTop candidates:\n")
    print(top.to_string(index=False))

if __name__ == "__main__":
    main()
