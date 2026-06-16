"""
Holdout check — Aug 2025 to Apr 2026 (data/holdout/fii_dii_holdout.csv).

IMPORTANT CAVEAT, read before trusting this output:

The main pipeline's signals use a 252-day rolling percentile window. The
holdout segment is only 167 trading days and is separated from the training
data by a ~2.5 year gap (2023-2024 has zero FII/DII coverage in this
project's data). That means there is no way to compute a "trailing 252
trading days" feature for the holdout period that means the same thing it
meant during training — it would either reach back across the gap into 2022
(a ~3.5 calendar-year lookback masquerading as a 1-year one) or rely on
holdout-only data that doesn't exist yet.

So this is NOT a like-for-like out-of-sample replication of the main test.
It is a secondary, lower-power, methodologically distinct exploratory check:
a 60-day rolling window confined entirely to the holdout's own data. Treat
its result as directional color, not as confirmation or refutation of the
main finding.
"""

import os
import pandas as pd

from config import RAW_DIR, RESULTS_DIR, SECTORS
from capm import compute_capm_errors
from detector import compute_anomaly_flags
from test_prediction import test_forward_prediction_episodes, sign_test

HOLDOUT_PATH = os.path.join(os.path.dirname(RAW_DIR), "holdout", "fii_dii_holdout.csv")
SHORT_WINDOW = 60  # confined to holdout-only data; not comparable to main's 252-day window


def run():
    print("=" * 60)
    print("HOLDOUT CHECK (Aug 2025 - Apr 2026) — exploratory, see module docstring")
    print("=" * 60)

    import yfinance as yf
    from config import MARKET_SYMBOL, VIX_SYMBOL

    fii_dii = pd.read_csv(HOLDOUT_PATH, parse_dates=["date"]).set_index("date")
    start, end = fii_dii.index.min(), fii_dii.index.max() + pd.Timedelta(days=1)

    market = {}
    for label, symbol in {"nifty": MARKET_SYMBOL, "india_vix": VIX_SYMBOL, **{k.lower(): v for k, v in SECTORS.items()}}.items():
        hist = yf.download(symbol, start=start - pd.Timedelta(days=10), end=end, progress=False)
        if hist.empty:
            print(f"  WARNING: no data for {symbol}")
            continue
        close = hist["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        col = "india_vix" if label == "india_vix" else f"{label}_close"
        market[col] = close

    df = pd.DataFrame(market)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(fii_dii[["fii_net", "dii_net"]], how="inner").sort_index()

    if len(df) < SHORT_WINDOW + 20:
        print(f"  Not enough holdout rows ({len(df)}) for even a {SHORT_WINDOW}-day window. Aborting.")
        return

    df["flow_div_raw"] = df["fii_net"] - df["dii_net"]
    df["flow_div_pct"] = df["flow_div_raw"].rolling(SHORT_WINDOW).rank(pct=True)

    import numpy as np
    log_ret = np.log(df["nifty_close"] / df["nifty_close"].shift(1))
    realized_vol = log_ret.rolling(20).std() * np.sqrt(252)
    df["vol_spread_raw"] = (df["india_vix"] / 100.0) - realized_vol
    df["vol_spread_pct"] = df["vol_spread_raw"].rolling(SHORT_WINDOW).rank(pct=True)

    df = compute_capm_errors(df, beta_window=20)  # shortened to fit holdout length
    df = compute_anomaly_flags(df, extreme=0.90, min_concordance=2)

    results = []
    for name in SECTORS:
        prefix = name.lower()
        error_col = f"{prefix}_capm_error"
        if error_col not in df.columns:
            continue
        for fwd in [1, 3, 5]:
            signs = test_forward_prediction_episodes(df, prefix, fwd)
            n_pos, n_total, hit_rate, p_val = sign_test(signs)
            results.append({"sector": name, "forward_days": fwd, "n_flags": n_total,
                             "hit_rate": hit_rate, "p_value": p_val})
            print(f"  {name:15s} | fwd={fwd}d | n={n_total} | hit={hit_rate if pd.notna(hit_rate) else 'n/a'} | p={p_val}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "holdout_check.csv"), index=False)
    print(f"\nSaved to {RESULTS_DIR}/holdout_check.csv")
    print("\nReminder: this used a 60-day window confined to holdout data, not the "
          "main methodology's 252-day window. Sample sizes here are small (n<15 typical). "
          "Do not present this as a confirmatory replication.")


if __name__ == "__main__":
    run()
