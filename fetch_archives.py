"""
NSE Archives Fetcher — downloads F&O participant OI data from NSE archives.

Source: nsearchives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv
Available: 2015-present

Extracts per trading day:
  - fii_net: FII total long - total short contracts (F&O positioning proxy)
  - dii_net: DII total long - total short contracts
  - pcr:     (index put OI + stock put OI) / (index call OI + stock call OI)

NOTE: fii_net/dii_net here measure F&O market positioning (contracts), NOT
equity cash market flows (INR Crores). They capture directional conviction
of FII vs DII and serve as a valid proxy for institutional divergence.

Saves:
  data/raw/fii_dii.csv   (fii_net, dii_net per date)
  data/raw/pcr.csv        (pcr per date)

Run this ONCE before data_pipeline.py. Incremental — resumes if interrupted.
"""

import os
import time
import datetime
import requests
import pandas as pd
from io import StringIO

from config import START_DATE, END_DATE, RAW_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NSE_ARCHIVES_BASE = "https://nsearchives.nseindia.com/content/nsccl"
NSE_HOME = "https://www.nseindia.com"
CACHE_PATH = os.path.join(RAW_DIR, "nse_archives_oi.csv")
DELAY_SECONDS = 0.6   # rate limit — NSE archives is tolerant but be polite
MAX_ERRORS_IN_ROW = 10  # abort if too many consecutive failures


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

_session = None


def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.get(NSE_HOME, headers=HEADERS, timeout=15)
        time.sleep(1.5)
    return _session


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_participant_oi(text, date):
    """
    Parse NSE participant OI CSV for one trading day.

    Returns dict with: date, fii_net, dii_net, pcr
    Returns None if parsing fails.
    """
    lines = [l for l in text.strip().split('\n') if l.strip()]
    if len(lines) < 4:
        return None

    # Line 0: title, Line 1: column headers, Lines 2+: data rows
    result = {"date": date.isoformat()}

    for line in lines[2:]:
        # Clean: remove quotes, tabs, trailing whitespace
        raw = line.replace('"', '').replace('\t', ',')
        parts = [p.strip() for p in raw.split(',')]
        if not parts or len(parts) < 13:
            continue

        row_name = parts[0].upper().strip()

        if row_name in ("FII", "DII"):
            try:
                long_ = float(parts[13] or 0)
                short_ = float(parts[14] or 0)
                key = row_name.lower()
                result[f"{key}_long"] = long_
                result[f"{key}_short"] = short_
                result[f"{key}_net"] = long_ - short_
            except (ValueError, IndexError):
                pass

        elif row_name == "TOTAL":
            try:
                # Columns (0-indexed after client_type):
                # 1=FutIdxLong  2=FutIdxShort  3=FutStkLong  4=FutStkShort
                # 5=OptIdxCallLong  6=OptIdxPutLong
                # 7=OptIdxCallShort 8=OptIdxPutShort
                # 9=OptStkCallLong  10=OptStkPutLong
                # 11=OptStkCallShort 12=OptStkPutShort
                # 13=TotalLong  14=TotalShort
                #
                # In OI, long = short (market clears). Use long-side for OI.
                idx_call = float(parts[5] or 0)
                idx_put  = float(parts[6] or 0)
                stk_call = float(parts[9] or 0)
                stk_put  = float(parts[10] or 0)

                call_oi = idx_call + stk_call
                put_oi  = idx_put  + stk_put

                if call_oi > 0:
                    result["pcr"] = round(put_oi / call_oi, 4)
            except (ValueError, IndexError):
                pass

    # Require at minimum fii_net, dii_net, pcr
    if "fii_net" in result and "dii_net" in result and "pcr" in result:
        return result

    return None


# ---------------------------------------------------------------------------
# Download one day
# ---------------------------------------------------------------------------

def download_one_day(date, session):
    """Download and parse participant OI for one date. Returns dict or None."""
    ds = date.strftime("%d%m%Y")
    url = f"{NSE_ARCHIVES_BASE}/fao_participant_oi_{ds}.csv"

    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            return None   # holiday / weekend — normal
        if resp.status_code != 200:
            return None

        return parse_participant_oi(resp.text, date)

    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_all(start=START_DATE, end=END_DATE):
    """
    Download participant OI for all NSE trading days in [start, end].

    Uses a local cache (nse_archives_oi.csv) to resume if interrupted.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    start_dt = pd.Timestamp(start).date()
    end_dt   = pd.Timestamp(end).date()

    # Load cache
    if os.path.exists(CACHE_PATH):
        cached = pd.read_csv(CACHE_PATH, parse_dates=["date"])
        cached_dates = set(pd.to_datetime(cached["date"]).dt.date)
        print(f"Cache: {len(cached)} rows, "
              f"{min(cached_dates)} → {max(cached_dates)}")
    else:
        cached = pd.DataFrame()
        cached_dates = set()

    # Build list of weekdays in range (weekends have no data)
    all_days = pd.bdate_range(start=start_dt, end=end_dt)
    missing = [d.date() for d in all_days if d.date() not in cached_dates]

    if not missing:
        print("Cache is complete — nothing to download.")
        return cached

    print(f"Downloading {len(missing)} trading days "
          f"({missing[0]} → {missing[-1]})...")
    print(f"Estimated time: {len(missing) * DELAY_SECONDS / 60:.0f}–"
          f"{len(missing) * DELAY_SECONDS * 2 / 60:.0f} minutes")

    session = get_session()
    records = []
    errors_in_row = 0
    n_ok = 0
    n_skip = 0
    save_every = 100  # save to CSV every N successful downloads

    for i, d in enumerate(missing):
        rec = download_one_day(d, session)

        if rec:
            records.append(rec)
            n_ok += 1
            errors_in_row = 0

            if n_ok % save_every == 0:
                _save(cached, records)
                pct = (i + 1) / len(missing) * 100
                print(f"  [{pct:.0f}%] {d}: saved {n_ok} records "
                      f"({n_skip} skipped)")
        else:
            n_skip += 1
            errors_in_row += 1

        if errors_in_row >= MAX_ERRORS_IN_ROW:
            print(f"\n  ABORT: {MAX_ERRORS_IN_ROW} consecutive failures. "
                  f"NSE may be blocking. Saving progress.")
            break

        time.sleep(DELAY_SECONDS)

    # Final save
    result = _save(cached, records)
    print(f"\nDone: {n_ok} downloaded, {n_skip} skipped (holidays/weekends)")
    print(f"Total cached rows: {len(result)}")
    print(f"Date range: {result['date'].min()} → {result['date'].max()}")
    return result


def _save(existing, new_records):
    """Merge new records with cache and save."""
    if not new_records:
        return existing

    new_df = pd.DataFrame(new_records)
    new_df["date"] = pd.to_datetime(new_df["date"])

    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date")
    else:
        combined = new_df.sort_values("date")

    combined.to_csv(CACHE_PATH, index=False)
    return combined


# ---------------------------------------------------------------------------
# Export to pipeline-compatible files
# ---------------------------------------------------------------------------

def export_to_pipeline_files(df=None):
    """
    Convert the archived data into the files that data_pipeline.py expects:
      data/raw/fii_dii.csv  — columns: date, fii_net, dii_net
      data/raw/pcr.csv      — columns: date, pcr
    """
    if df is None:
        if not os.path.exists(CACHE_PATH):
            print("No cache found. Run fetch_all() first.")
            return
        df = pd.read_csv(CACHE_PATH, parse_dates=["date"])

    df = df.sort_values("date")

    # FII/DII file — SKIP: F&O net positioning is a structurally inverted proxy
    # for equity cash flows. Do not overwrite the real equity FII/DII CSV.
    print("  SKIP: fii_dii.csv (F&O positioning != equity flows)")

    # PCR file
    pcr_path = os.path.join(RAW_DIR, "pcr.csv")
    pcr_df = df[["date", "pcr"]].dropna()
    pcr_df.to_csv(pcr_path, index=False)
    print(f"  Wrote {len(pcr_df)} rows to {pcr_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NSE ARCHIVES FETCHER")
    print("F&O Participant OI → FII/DII positioning + PCR")
    print("=" * 60)
    print()

    df = fetch_all()

    print("\nExporting to pipeline files...")
    export_to_pipeline_files(df)

    print("\nNext step: python3 run.py")
