"""
Layer 1 — Signal Construction

Computes 3 signals from aligned daily data, each mapped to
a rolling 252-day percentile rank:

  1. flow_div  — FII net - DII net (institutional flow divergence)
  2. pcr_dev   — PCR deviation from 30-day baseline (z-score)
  3. vol_spread — India VIX implied vol - 20-day realized vol

Percentile rank maps each signal to [0, 1] where values near 0 or 1
indicate historically extreme readings relative to the trailing year.
"""

import pandas as pd
import numpy as np

from config import (
    PERCENTILE_WINDOW,
    PCR_BASELINE_WINDOW,
    REALIZED_VOL_WINDOW,
)


def compute_flow_divergence(df, window=PERCENTILE_WINDOW):
    """
    Signal 1: FII/DII flow divergence.

    Raw: FII_net - DII_net (INR Crores)
    Percentile: rolling 252-day rank of raw divergence.

    High percentile = FII buying much more than DII (or DII selling more)
    Low percentile = DII buying much more than FII (or FII selling more)
    """
    if "fii_net" not in df.columns or "dii_net" not in df.columns:
        print("  SKIP: flow divergence (no FII/DII data)")
        return df

    df["flow_div_raw"] = df["fii_net"] - df["dii_net"]
    df["flow_div_pct"] = df["flow_div_raw"].rolling(window).rank(pct=True)

    n_valid = df["flow_div_pct"].notna().sum()
    print(f"  Flow divergence: {n_valid} valid days (warm-up: {window})")
    return df


def compute_pcr_deviation(df, baseline=PCR_BASELINE_WINDOW, window=PERCENTILE_WINDOW):
    """
    Signal 2: PCR baseline deviation.

    Raw: (PCR - 30d mean) / 30d std (z-score)
    Percentile: rolling 252-day rank of z-score.

    High percentile = PCR unusually elevated vs recent history (fear/hedging)
    Low percentile = PCR unusually low vs recent history (complacency)
    """
    if "pcr" not in df.columns:
        print("  SKIP: PCR deviation (no PCR data)")
        return df

    pcr_mean = df["pcr"].rolling(baseline).mean()
    pcr_std = df["pcr"].rolling(baseline).std()

    # Guard against zero std (constant PCR over baseline period)
    pcr_std = pcr_std.replace(0, np.nan)

    df["pcr_dev_raw"] = (df["pcr"] - pcr_mean) / pcr_std
    df["pcr_dev_raw"] = df["pcr_dev_raw"].fillna(0)  # zero if constant
    df["pcr_dev_pct"] = df["pcr_dev_raw"].rolling(window).rank(pct=True)

    n_valid = df["pcr_dev_pct"].notna().sum()
    warmup = baseline + window
    print(f"  PCR deviation: {n_valid} valid days (warm-up: {warmup})")
    return df


def compute_vol_spread(df, rv_window=REALIZED_VOL_WINDOW, window=PERCENTILE_WINDOW):
    """
    Signal 3: Implied/realized volatility spread.

    Raw: India VIX (annualized implied vol) - 20-day realized vol (annualized)
    Percentile: rolling 252-day rank of spread.

    High percentile = market pricing more risk than recent history shows
    Low percentile = market pricing less risk than recent history (complacency)

    India VIX is quoted as percentage (e.g., 15 = 15% annualized).
    Realized vol computed from log returns, annualized by sqrt(252).
    """
    if "india_vix" not in df.columns:
        print("  SKIP: vol spread (no India VIX data)")
        return df

    # Log returns
    log_ret = np.log(df["nifty_close"] / df["nifty_close"].shift(1))

    # Realized vol: annualized
    realized_vol = log_ret.rolling(rv_window).std() * np.sqrt(252)

    # India VIX: convert from percentage to decimal
    implied_vol = df["india_vix"] / 100.0

    df["realized_vol"] = realized_vol
    df["vol_spread_raw"] = implied_vol - realized_vol
    df["vol_spread_pct"] = df["vol_spread_raw"].rolling(window).rank(pct=True)

    n_valid = df["vol_spread_pct"].notna().sum()
    warmup = rv_window + window
    print(f"  Vol spread: {n_valid} valid days (warm-up: {warmup})")
    return df


def compute_all_signals(df, window=PERCENTILE_WINDOW):
    """Compute all 3 signals. Returns DataFrame with new columns."""
    print("\nComputing signals...")

    df = compute_flow_divergence(df, window)
    df = compute_pcr_deviation(df, PCR_BASELINE_WINDOW, window)
    df = compute_vol_spread(df, REALIZED_VOL_WINDOW, window)

    # Count available signals
    signal_cols = ["flow_div_pct", "pcr_dev_pct", "vol_spread_pct"]
    available = [c for c in signal_cols if c in df.columns]
    print(f"\n  Signals available: {len(available)}/3 — {available}")

    # Summary stats for available signals (post warm-up)
    for col in available:
        valid = df[col].dropna()
        print(f"  {col}: mean={valid.mean():.3f}, std={valid.std():.3f}, "
              f"min={valid.min():.3f}, max={valid.max():.3f}")

    return df


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from config import CLEAN_DIR

    csv_path = os.path.join(CLEAN_DIR, "aligned_daily.csv")
    if not os.path.exists(csv_path):
        print(f"Run data_pipeline.py first. Missing: {csv_path}")
        raise SystemExit(1)

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df = compute_all_signals(df)

    out_path = os.path.join(CLEAN_DIR, "signals.csv")
    df.to_csv(out_path)
    print(f"\nSaved signals to {out_path}")
