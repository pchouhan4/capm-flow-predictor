"""
Layer 2 — CAPM Pricing Error

Computes rolling CAPM for each sector index vs Nifty 50:
  - Rolling 60-day beta (OLS regression)
  - Expected return = Rf + beta * (Rm - Rf)
  - Pricing error = actual return - expected return

Output: capm_error and capm_error_sign columns per sector.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from config import CAPM_BETA_WINDOW, RFR_ANNUAL, SECTORS


def compute_capm_errors(df, beta_window=CAPM_BETA_WINDOW):
    """
    For each sector index, compute rolling CAPM pricing error vs Nifty 50.

    Returns DataFrame with columns per sector:
      {sector}_return, {sector}_beta, {sector}_expected, {sector}_capm_error,
      {sector}_error_sign
    """
    print("\nComputing CAPM pricing errors...")

    # Daily risk-free rate
    rf_daily = (1 + RFR_ANNUAL) ** (1 / 252) - 1

    # Market (Nifty 50) daily return
    df["nifty_return"] = df["nifty_close"].pct_change()
    df["market_excess"] = df["nifty_return"] - rf_daily

    sectors_computed = []

    for name, symbol in SECTORS.items():
        col = f"{name.lower()}_close"
        if col not in df.columns:
            print(f"  SKIP: {name} (no data)")
            continue

        prefix = name.lower()

        # Sector daily return
        df[f"{prefix}_return"] = df[col].pct_change()
        df[f"{prefix}_excess"] = df[f"{prefix}_return"] - rf_daily

        # Rolling OLS: sector_excess ~ market_excess
        # Note: this is O(n * beta_window) — a Python loop fitting one OLS per row.
        # For the dataset sizes here (~1,900 days) this is fast enough.
        # If extending to larger datasets, statsmodels.regression.rolling.RollingOLS
        # runs the same computation with a recursive QR update and is significantly faster.
        betas = []
        alphas = []

        for i in range(len(df)):
            if i < beta_window:
                betas.append(np.nan)
                alphas.append(np.nan)
                continue

            y = df[f"{prefix}_excess"].iloc[i - beta_window:i].values
            x = df["market_excess"].iloc[i - beta_window:i].values

            # Skip if insufficient valid data
            valid = ~(np.isnan(y) | np.isnan(x))
            if valid.sum() < beta_window * 0.8:  # need 80% valid
                betas.append(np.nan)
                alphas.append(np.nan)
                continue

            y_clean = y[valid]
            x_clean = sm.add_constant(x[valid])

            try:
                model = sm.OLS(y_clean, x_clean).fit()
                alphas.append(model.params[0])
                betas.append(model.params[1])
            except Exception:
                betas.append(np.nan)
                alphas.append(np.nan)

        df[f"{prefix}_beta"] = betas
        df[f"{prefix}_alpha"] = alphas

        # Expected return (CAPM)
        df[f"{prefix}_expected"] = rf_daily + df[f"{prefix}_beta"] * df["market_excess"]

        # Pricing error
        df[f"{prefix}_capm_error"] = df[f"{prefix}_return"] - df[f"{prefix}_expected"]

        # Error sign (+1 = CAPM underestimated, -1 = CAPM overestimated)
        df[f"{prefix}_error_sign"] = np.sign(df[f"{prefix}_capm_error"])

        # Stats
        valid_errors = df[f"{prefix}_capm_error"].dropna()
        mean_beta = df[f"{prefix}_beta"].dropna().mean()
        mean_error = valid_errors.mean()
        std_error = valid_errors.std()

        print(f"  {name}: avg_beta={mean_beta:.3f}, "
              f"mean_error={mean_error:.6f}, std_error={std_error:.4f}, "
              f"n={len(valid_errors)}")

        sectors_computed.append(name)

    print(f"\n  Sectors computed: {len(sectors_computed)}/{len(SECTORS)}")
    return df


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from config import CLEAN_DIR

    csv_path = os.path.join(CLEAN_DIR, "signals.csv")
    if not os.path.exists(csv_path):
        print(f"Run signals.py first. Missing: {csv_path}")
        raise SystemExit(1)

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df = compute_capm_errors(df)

    out_path = os.path.join(CLEAN_DIR, "capm_errors.csv")
    df.to_csv(out_path)
    print(f"\nSaved CAPM errors to {out_path}")
