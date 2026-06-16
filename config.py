"""
Configuration — all tunable parameters in one place.

Research question: Does daily FII/DII flow divergence predict the
direction of CAPM pricing error in Indian equities?
"""

import os

# --- Date range ---
# Bounded to the clean, continuous FII/DII training window. The holdout
# segment (data/holdout/fii_dii_holdout.csv, Aug 2025-Apr 2026) is kept out
# of this range deliberately so it never gets pulled into training/testing.
START_DATE = "2015-01-01"
END_DATE = "2022-12-08"

# --- Rolling windows ---
PERCENTILE_WINDOW = 252       # 1 trading year for percentile rank
PCR_BASELINE_WINDOW = 30      # 30-day PCR mean/std for z-score
REALIZED_VOL_WINDOW = 20      # 20-day realized vol (annualized)
CAPM_BETA_WINDOW = 60         # 60-day rolling beta estimation

# --- Signal thresholds ---
EXTREME_PERCENTILE = 0.90     # top/bottom 10th percentile = "extreme"
MIN_CONCORDANCE = 2           # minimum signals in extreme zone to flag

# --- Forward test horizons ---
FORWARD_WINDOWS = [1, 3, 5, 10]  # days ahead to test CAPM error direction

# --- Risk-free rate ---
RFR_ANNUAL = 0.065            # ~6.5% India 10Y govt bond yield

# --- Sector indices (yfinance symbols) ---
SECTORS = {
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_AUTO": "^CNXAUTO",
}

# --- Market index ---
MARKET_SYMBOL = "^NSEI"       # Nifty 50
VIX_SYMBOL = "^INDIAVIX"      # India VIX

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- Robustness sweep ranges ---
WINDOW_SWEEP = [126, 189, 252, 378]
PERCENTILE_SWEEP = [0.85, 0.87, 0.90, 0.93, 0.95]
