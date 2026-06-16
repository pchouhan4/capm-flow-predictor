"""
Layer 0 — Data Pipeline

Downloads, cleans, and aligns:
  1. Nifty 50 OHLCV (yfinance)
  2. India VIX (yfinance)
  3. Sector indices — Bank, IT, Pharma, Auto (yfinance)
  4. FII/DII daily flows (NSE API / CSV fallback)
  5. PCR — Put-Call Ratio (NSE derivatives / manual CSV fallback)

Output: data/clean/aligned_daily.csv
"""

import os
import json
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

from config import (
    START_DATE, END_DATE, MARKET_SYMBOL, VIX_SYMBOL,
    SECTORS, RAW_DIR, CLEAN_DIR
)


# ---------------------------------------------------------------------------
# 1. Nifty 50 + India VIX + Sector indices (yfinance — reliable)
# ---------------------------------------------------------------------------

def download_yfinance(symbol, name, start=START_DATE, end=END_DATE):
    """Download OHLCV from yfinance, save raw CSV, return DataFrame."""
    print(f"  Downloading {name} ({symbol})...")
    df = yf.download(symbol, start=start, end=end, progress=False)

    raw_path = os.path.join(RAW_DIR, f"{name}.csv")

    if df.empty:
        print(f"  WARNING: No data for {symbol}")
        if os.path.exists(raw_path):
            print(f"  Falling back to cached {raw_path}")
            cached = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            cached.index.name = "date"
            return cached
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "date"
    df.to_csv(raw_path)
    print(f"  Saved {len(df)} rows to {raw_path}")
    return df


def download_market_data():
    """Download Nifty 50, India VIX, and sector indices."""
    data = {}

    # Nifty 50
    nifty = download_yfinance(MARKET_SYMBOL, "nifty50")
    if not nifty.empty:
        data["nifty"] = nifty[["Close"]].rename(columns={"Close": "nifty_close"})

    # India VIX
    vix = download_yfinance(VIX_SYMBOL, "india_vix")
    if not vix.empty:
        data["vix"] = vix[["Close"]].rename(columns={"Close": "india_vix"})

    # Sector indices
    for name, symbol in SECTORS.items():
        sector_df = download_yfinance(symbol, name.lower())
        if not sector_df.empty:
            col = f"{name.lower()}_close"
            data[name.lower()] = sector_df[["Close"]].rename(columns={"Close": col})

    return data


# ---------------------------------------------------------------------------
# 2. FII/DII daily flows
# ---------------------------------------------------------------------------

def download_fii_dii_nse():
    """
    Attempt to fetch FII/DII data from NSE API.
    This often requires browser-like session cookies.
    Returns DataFrame or None on failure.
    """
    print("  Attempting NSE API for FII/DII...")
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/reports/fii-dii",
    }

    session = requests.Session()
    try:
        # First hit the main page to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data)
            if not df.empty:
                print(f"  Got {len(df)} rows from NSE API")
                return df
    except Exception as e:
        print(f"  NSE API failed: {e}")

    return None


def load_fii_dii_csv(filepath):
    """
    Load FII/DII data from a manually downloaded CSV.

    Expected columns (flexible matching):
      date, FII/FPI-Buy, FII/FPI-Sell, FII/FPI-Net,
      DII-Buy, DII-Sell, DII-Net

    Values in INR Crores.
    """
    print(f"  Loading FII/DII from CSV: {filepath}")
    df = pd.read_csv(filepath)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "date" in cl:
            col_map[col] = "date"
        elif ("fii" in cl or "fpi" in cl) and "buy" in cl and "net" not in cl:
            col_map[col] = "fii_buy"
        elif ("fii" in cl or "fpi" in cl) and "sell" in cl:
            col_map[col] = "fii_sell"
        elif ("fii" in cl or "fpi" in cl) and "net" in cl:
            col_map[col] = "fii_net"
        elif "dii" in cl and "buy" in cl and "net" not in cl:
            col_map[col] = "dii_buy"
        elif "dii" in cl and "sell" in cl:
            col_map[col] = "dii_sell"
        elif "dii" in cl and "net" in cl:
            col_map[col] = "dii_net"

    df = df.rename(columns=col_map)

    required = ["date", "fii_net", "dii_net"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns {missing} in FII/DII CSV")
        print(f"  Found columns: {list(df.columns)}")
        return pd.DataFrame()

    # Parse dates (try ISO first, fall back to dayfirst for DD-MM-YYYY)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().sum() > len(df) * 0.5:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()

    # Clean numeric columns — remove commas, convert to float
    for col in ["fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("(", "-", regex=False)
                .str.replace(")", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Loaded {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
    return df


def get_fii_dii():
    """
    Get FII/DII data. Strategy:
      1. Try NSE API (recent data only, ~30 days)
      2. Look for manual CSV in data/raw/fii_dii.csv
      3. If neither, create placeholder and print instructions
    """
    # Check for manual CSV first (most reliable for historical data)
    csv_path = os.path.join(RAW_DIR, "fii_dii.csv")
    if os.path.exists(csv_path):
        df = load_fii_dii_csv(csv_path)
        if not df.empty:
            return df[["fii_net", "dii_net"]]

    # Try NSE API (returns today only — category rows for FII/FPI and DII)
    api_df = download_fii_dii_nse()
    if api_df is not None and not api_df.empty:
        raw_path = os.path.join(RAW_DIR, "fii_dii_api.json")
        api_df.to_json(raw_path, orient="records", indent=2)

        # Parse category-row format: {"category": "FII/FPI", "date": ..., "netValue": ...}
        try:
            row_map = {}
            for _, row in api_df.iterrows():
                cat = str(row.get("category", "")).upper()
                date = pd.to_datetime(row.get("date", ""),
                                       dayfirst=True, errors="coerce")
                net = float(str(row.get("netValue", 0)).replace(",", ""))
                if pd.notna(date):
                    row_map[cat] = {"date": date, "net": net}

            fii_row = row_map.get("FII/FPI") or row_map.get("FII")
            dii_row = row_map.get("DII")
            if fii_row and dii_row and fii_row["date"] == dii_row["date"]:
                record = {
                    "date": fii_row["date"],
                    "fii_net": fii_row["net"],
                    "dii_net": dii_row["net"],
                }
                df = pd.DataFrame([record]).set_index("date")
                print(f"  NSE API: got 1 row for {record['date'].date()}")
                # Only 1 row — not useful for historical; fall through to archives
        except Exception as e:
            print(f"  Failed to parse API data: {e}")

    # Neither worked — provide instructions
    print("\n" + "=" * 70)
    print("FII/DII DATA NOT FOUND")
    print("=" * 70)
    print("""
To get historical FII/DII data (2015-2025):

Option 1 — NSE bulk download:
  1. Go to https://www.nseindia.com/reports/fii-dii
  2. Download the CSV for each year
  3. Combine into one CSV with columns:
     date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net
  4. Save as: data/raw/fii_dii.csv

Option 2 — MoneyControl:
  1. Search "FII DII data historical download moneycontrol"
  2. Download and format as above

Option 3 — Use the fii-dii-data repo:
  git clone https://github.com/MrChartist/fii-dii-data.git
  (contains historical append scripts with pre-2020 data)

Values should be in INR Crores. Dates in DD-MM-YYYY or YYYY-MM-DD.
""")
    print("=" * 70)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. PCR (Put-Call Ratio)
# ---------------------------------------------------------------------------

def load_pcr_csv(filepath):
    """Load PCR from a manually downloaded CSV."""
    print(f"  Loading PCR from CSV: {filepath}")
    df = pd.read_csv(filepath)

    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "date" in cl:
            col_map[col] = "date"
        elif "pcr" in cl or "put" in cl and "call" in cl:
            col_map[col] = "pcr"

    df = df.rename(columns=col_map)

    if "date" not in df.columns or "pcr" not in df.columns:
        print(f"  WARNING: Cannot find date/pcr columns in {filepath}")
        print(f"  Found: {list(df.columns)}")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().sum() > len(df) * 0.5:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["pcr"] = pd.to_numeric(df["pcr"], errors="coerce")
    df = df.dropna(subset=["date", "pcr"])
    df = df.set_index("date").sort_index()

    print(f"  Loaded {len(df)} rows, range: {df.index.min()} to {df.index.max()}")
    return df


def get_pcr():
    """
    Get PCR data. Strategy:
      1. Look for manual CSV in data/raw/pcr.csv
      2. If not found, print instructions
    """
    csv_path = os.path.join(RAW_DIR, "pcr.csv")
    if os.path.exists(csv_path):
        df = load_pcr_csv(csv_path)
        if not df.empty:
            return df[["pcr"]]

    print("\n" + "=" * 70)
    print("PCR DATA NOT FOUND")
    print("=" * 70)
    print("""
To get historical PCR (Put-Call Ratio) data:

Option 1 — NSE:
  1. Go to https://www.nseindia.com/option-chain
  2. Historical PCR data may be available in derivatives reports
  3. Save as CSV with columns: date, pcr
  4. Save as: data/raw/pcr.csv

Option 2 — Trading platforms:
  - Sensibull, Opstra, or similar platforms publish historical PCR

Option 3 — Compute from options data:
  PCR = total_put_OI / total_call_OI (for Nifty 50 options)

If PCR data is unavailable, the system will run with 2 signals
(flow divergence + vol spread) instead of 3.
""")
    print("=" * 70)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Align and merge all data
# ---------------------------------------------------------------------------

def align_data(data_dict, fii_dii_df, pcr_df):
    """
    Align all data sources on trading dates.
    Inner join — only keep days where core sources have data.
    """
    print("\nAligning data sources...")

    # Start with Nifty (most reliable date index)
    if "nifty" not in data_dict or data_dict["nifty"].empty:
        raise ValueError("Nifty 50 data is required. Check yfinance connection.")

    merged = data_dict["nifty"].copy()

    # Add India VIX
    if "vix" in data_dict and not data_dict["vix"].empty:
        merged = merged.join(data_dict["vix"], how="inner")
    else:
        print("  WARNING: No India VIX data — vol spread signal unavailable")

    # Add sector indices
    for name in SECTORS:
        key = name.lower()
        if key in data_dict and not data_dict[key].empty:
            merged = merged.join(data_dict[key], how="inner")

    # Add FII/DII
    if not fii_dii_df.empty:
        # Ensure index tz-naive for join
        fii_dii_df.index = pd.to_datetime(fii_dii_df.index).normalize()
        merged.index = pd.to_datetime(merged.index).normalize()
        merged = merged.join(fii_dii_df, how="inner")
    else:
        print("  WARNING: No FII/DII data — flow divergence signal unavailable")

    # Add PCR
    if not pcr_df.empty:
        pcr_df.index = pd.to_datetime(pcr_df.index).normalize()
        merged = merged.join(pcr_df, how="inner")
    else:
        print("  WARNING: No PCR data — PCR deviation signal unavailable")

    # Forward-fill 1-day gaps (holiday misalignment)
    merged = merged.ffill(limit=1)

    # Drop any remaining NaN rows in core columns
    core_cols = ["nifty_close"]
    if "india_vix" in merged.columns:
        core_cols.append("india_vix")
    merged = merged.dropna(subset=core_cols)

    merged = merged.sort_index()

    print(f"\n  Aligned dataset: {len(merged)} trading days")
    print(f"  Date range: {merged.index.min().date()} to {merged.index.max().date()}")
    print(f"  Columns: {list(merged.columns)}")

    # Count available signals
    n_signals = 0
    if "fii_net" in merged.columns:
        n_signals += 1
        print(f"  FII/DII: {merged['fii_net'].notna().sum()} days")
    if "pcr" in merged.columns:
        n_signals += 1
        print(f"  PCR: {merged['pcr'].notna().sum()} days")
    if "india_vix" in merged.columns:
        n_signals += 1
        print(f"  India VIX: {merged['india_vix'].notna().sum()} days")

    print(f"\n  Available signals: {n_signals}/3")
    if n_signals < 2:
        print("  WARNING: Need at least 2 signals for concordance detection.")

    return merged


# ---------------------------------------------------------------------------
# 5. Validation checks
# ---------------------------------------------------------------------------

def validate_data(df):
    """Sanity checks on the aligned dataset."""
    print("\nValidation checks:")
    issues = []

    # Date monotonic
    if not df.index.is_monotonic_increasing:
        issues.append("Date index is not monotonically increasing")

    # Nifty close in reasonable range
    if "nifty_close" in df.columns:
        nmin, nmax = df["nifty_close"].min(), df["nifty_close"].max()
        if nmin < 1000 or nmax > 50000:
            issues.append(f"Nifty close out of range: [{nmin:.0f}, {nmax:.0f}]")
        else:
            print(f"  Nifty close range: [{nmin:.0f}, {nmax:.0f}] OK")

    # India VIX in reasonable range
    if "india_vix" in df.columns:
        vmin, vmax = df["india_vix"].min(), df["india_vix"].max()
        if vmin < 5 or vmax > 100:
            issues.append(f"India VIX out of range: [{vmin:.1f}, {vmax:.1f}]")
        else:
            print(f"  India VIX range: [{vmin:.1f}, {vmax:.1f}] OK")

    # FII/DII: either INR Crores (equity flows) or F&O contracts (positioning)
    # Just check non-zero and finite
    if "fii_net" in df.columns:
        fmin, fmax = df["fii_net"].min(), df["fii_net"].max()
        if not (np.isfinite(fmin) and np.isfinite(fmax)):
            issues.append(f"FII net has non-finite values")
        else:
            print(f"  FII net range: [{fmin:.0f}, {fmax:.0f}] OK")

    # PCR in reasonable range
    if "pcr" in df.columns:
        pmin, pmax = df["pcr"].min(), df["pcr"].max()
        if pmin < 0.1 or pmax > 5.0:
            issues.append(f"PCR out of range: [{pmin:.2f}, {pmax:.2f}]")
        else:
            print(f"  PCR range: [{pmin:.2f}, {pmax:.2f}] OK")

    # NaN check
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"  NaN counts: {dict(nan_cols)}")

    if issues:
        print(f"\n  ISSUES FOUND:")
        for i in issues:
            print(f"    - {i}")
    else:
        print("  All checks passed.")

    return len(issues) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline():
    """Run the full data pipeline."""
    print("=" * 60)
    print("CAPM FLOW PREDICTOR — DATA PIPELINE")
    print("=" * 60)

    # Ensure directories exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    # 1. Download market data from yfinance
    print("\n[1/4] Downloading market data (yfinance)...")
    market_data = download_market_data()

    # 2. Get FII/DII data
    print("\n[2/4] Getting FII/DII flow data...")
    fii_dii = get_fii_dii()

    # 3. Get PCR data
    print("\n[3/4] Getting PCR data...")
    pcr = get_pcr()

    # 4. Align everything
    print("\n[4/4] Aligning data sources...")
    aligned = align_data(market_data, fii_dii, pcr)

    # Validate
    validate_data(aligned)

    # Save
    out_path = os.path.join(CLEAN_DIR, "aligned_daily.csv")
    aligned.to_csv(out_path)
    print(f"\nSaved aligned dataset to {out_path}")
    print(f"Shape: {aligned.shape}")

    return aligned


if __name__ == "__main__":
    run_pipeline()
