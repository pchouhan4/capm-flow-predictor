"""
Reproducible check: does NSE F&O participant OI net positioning serve as a
valid proxy for equity cash market FII/DII flows?

Revision 2 — fixes two problems found by review of revision 1:

1. INSUFFICIENT OVERLAP. Revision 1 only checked against the 167-day
   holdout (of which only 100 days actually had OI-archive coverage). The
   training data (data/raw/fii_dii.csv, 2015-2022) overlaps the OI archive
   for 1,650 days. Using only equity ground-truth here, never refitting
   anything, so this isn't a leakage concern — it's purely a correlation
   check between two raw data series. This revision uses the full overlap
   and reports results broken out by sub-period, since a single-window
   result can't speak to whether the relationship (or lack of one) is
   stable across regimes.

2. STOCK VS FLOW MISMATCH. The F&O "fii_net" column in nse_archives_oi.csv
   is a LEVEL — net outstanding long-short contracts on that day (a stock).
   Equity "fii_net" is a FLOW — that day's net buy/sell transaction. These
   are not the same kind of quantity; comparing them directly conflates
   "is F&O positioning growing/shrinking in the same direction as equity
   buying" with "is the day's F&O trading activity in the same direction as
   equity trading." This revision tests BOTH the raw level (as rev. 1 did)
   and the day-over-day CHANGE in the level (the closer flow-equivalent),
   and reports them separately so the construction choice is visible
   instead of buried in one number.

KNOWN LIMITATION NOT FIXED HERE: no network access in this environment, so
the NSE archive file itself (data/raw/nse_archives_oi.csv) cannot be
independently re-fetched and diffed against the live source in this run.
Source-quality verification is limited to what's already recorded in
data/MANIFEST.md (hash + "declared, not independently re-verified").
"""

import os
import pandas as pd
from scipy import stats

from config import RAW_DIR, DATA_DIR, RESULTS_DIR

OI_PATH = os.path.join(RAW_DIR, "nse_archives_oi.csv")
TRAIN_PATH = os.path.join(RAW_DIR, "fii_dii.csv")
HOLDOUT_PATH = os.path.join(DATA_DIR, "holdout", "fii_dii_holdout.csv")

PERIODS = [
    ("2015-01-01", "2016-12-31"),
    ("2017-01-01", "2018-12-31"),
    ("2019-01-01", "2020-12-31"),
    ("2021-01-01", "2022-12-08"),
    ("2025-08-01", "2025-12-31"),  # holdout segment, OI archive ends 2025-12-31
]


def compute_metrics(merged, fo_col, eq_col):
    n = len(merged)
    if n < 10:
        return None
    rho, p = stats.spearmanr(merged[fo_col], merged[eq_col])
    sign_agree = ((merged[fo_col] > 0) == (merged[eq_col] > 0)).mean()
    return {"n": n, "spearman_r": rho, "spearman_p": p, "sign_agreement": sign_agree}


def run():
    oi = pd.read_csv(OI_PATH, parse_dates=["date"]).sort_values("date")
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    holdout = pd.read_csv(HOLDOUT_PATH, parse_dates=["date"])
    equity = pd.concat([train, holdout], ignore_index=True).sort_values("date")

    # Flow-equivalent: day-over-day change in F&O net OI level
    oi["fii_net_change"] = oi["fii_net"].diff()

    merged = pd.merge(oi[["date", "fii_net", "fii_net_change"]],
                       equity[["date", "fii_net"]],
                       on="date", suffixes=("_fo", "_equity"))

    print(f"Full overlap: {merged.date.min().date()} to {merged.date.max().date()} (n={len(merged)})")
    print("(Training + holdout equity data combined as ground truth only — nothing is refit here.)\n")

    results = []

    for label, col in [("LEVEL (F&O net OI, as in rev.1)", "fii_net_fo"),
                        ("CHANGE (day-over-day diff, flow-equivalent)", "fii_net_change")]:
        m = compute_metrics(merged.dropna(subset=[col]), col, "fii_net_equity")
        print(f"--- {label} ---")
        if m:
            print(f"  n={m['n']}  Spearman r={m['spearman_r']:.3f} (p={m['spearman_p']:.4f})  "
                  f"sign agreement={m['sign_agreement']*100:.1f}%")
            results.append({"construction": label, "period": "full overlap", **m})
        print()

    print("--- Breakdown by period (CHANGE construction, the more defensible one) ---")
    for start, end in PERIODS:
        sub = merged[(merged.date >= start) & (merged.date <= end)].dropna(subset=["fii_net_change"])
        m = compute_metrics(sub, "fii_net_change", "fii_net_equity")
        if m:
            print(f"  {start} to {end}: n={m['n']:4d}  r={m['spearman_r']:+.3f}  "
                  f"sign_agree={m['sign_agreement']*100:.1f}%")
            results.append({"construction": "CHANGE (day-over-day diff)", "period": f"{start} to {end}", **m})
        else:
            print(f"  {start} to {end}: insufficient overlap")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "oi_proxy_validation.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print("\nReminder: data/raw/nse_archives_oi.csv was not re-fetched from NSE in this run "
          "(no network access here) — see data/MANIFEST.md for what's actually verified vs declared.")


if __name__ == "__main__":
    run()
