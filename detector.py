"""
Layer 3 — Anomaly Detector

Flags days where 2+ of 3 signals are in extreme percentiles
(top/bottom 10th percentile by default).

No behavioral labels. No herding claims.
Just: "signals are simultaneously extreme."
"""

import pandas as pd
import numpy as np

from config import EXTREME_PERCENTILE, MIN_CONCORDANCE


def compute_anomaly_flags(df, extreme=EXTREME_PERCENTILE, min_concordance=MIN_CONCORDANCE):
    """
    Flag days where multiple signals are simultaneously extreme.

    A signal is "extreme" if its percentile rank is >= extreme or <= (1 - extreme).
    Default: top/bottom 10th percentile.

    Args:
        df: DataFrame with signal percentile columns
        extreme: percentile threshold (0.90 = top/bottom 10%)
        min_concordance: minimum number of extreme signals to flag (default 2)

    Returns:
        DataFrame with anomaly_flag and n_extreme columns.
    """
    print(f"\nComputing anomaly flags (extreme={extreme}, min={min_concordance})...")

    signal_cols = ["flow_div_pct", "pcr_dev_pct", "vol_spread_pct"]
    available = [c for c in signal_cols if c in df.columns]

    if len(available) < min_concordance:
        print(f"  WARNING: Only {len(available)} signals available, "
              f"need {min_concordance} for concordance. Lowering threshold.")
        min_concordance = max(1, len(available))

    # Count extreme signals per day
    extreme_counts = pd.Series(0, index=df.index, dtype=int)
    extreme_details = {}

    for col in available:
        is_high = df[col] >= extreme
        is_low = df[col] <= (1 - extreme)
        is_extreme = is_high | is_low
        extreme_counts += is_extreme.astype(int)
        extreme_details[col] = is_extreme

        n_high = is_high.sum()
        n_low = is_low.sum()
        print(f"  {col}: {n_high} high-extreme, {n_low} low-extreme days")

    df["n_extreme"] = extreme_counts
    df["anomaly_flag"] = (extreme_counts >= min_concordance).astype(int)

    # Statistics
    total_days = len(df.dropna(subset=available))
    n_flagged = df["anomaly_flag"].sum()
    pct_flagged = n_flagged / total_days * 100 if total_days > 0 else 0

    print(f"\n  Total trading days (post warm-up): {total_days}")
    print(f"  Days flagged: {n_flagged} ({pct_flagged:.1f}%)")

    if pct_flagged > 30:
        print("  WARNING: >30% flagged — signals may be too correlated (echo problem)")
    elif pct_flagged < 2:
        print("  WARNING: <2% flagged — too restrictive for statistical testing")
    elif n_flagged < 20:
        print(f"  WARNING: Only {n_flagged} flags — statistical tests will be weak")

    # Flag clustering: how long do anomaly periods last?
    if n_flagged > 0:
        flag_runs = []
        current_run = 0
        for flag in df["anomaly_flag"]:
            if flag == 1:
                current_run += 1
            else:
                if current_run > 0:
                    flag_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            flag_runs.append(current_run)

        if flag_runs:
            print(f"  Flag clusters: {len(flag_runs)} episodes, "
                  f"avg duration={np.mean(flag_runs):.1f} days, "
                  f"max={max(flag_runs)} days")

    return df


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from config import CLEAN_DIR

    csv_path = os.path.join(CLEAN_DIR, "capm_errors.csv")
    if not os.path.exists(csv_path):
        print(f"Run capm.py first. Missing: {csv_path}")
        raise SystemExit(1)

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df = compute_anomaly_flags(df)

    out_path = os.path.join(CLEAN_DIR, "flagged.csv")
    df.to_csv(out_path)
    print(f"\nSaved flagged data to {out_path}")
