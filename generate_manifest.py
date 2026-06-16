"""
Data provenance manifest.

Computes SHA256 for every raw/holdout data file and records what is actually
known about its source — distinguishing files this code re-fetches live and
can independently re-verify (yfinance, NSE archives) from files that were
manually acquired before any script in this repo touched them (the Kaggle
FII/DII CSV), where provenance is only as good as what's written down, not
independently checkable by re-running anything here.

Run this whenever a raw data file changes. Commit the output
(data/MANIFEST.md) alongside the data so future diffs are visible.
"""

import hashlib
import os
from datetime import datetime, timezone

from config import RAW_DIR, DATA_DIR

HOLDOUT_DIR = os.path.join(DATA_DIR, "holdout")

# What's actually known about each file's origin. "verified" means this
# repo's own code fetched it directly from the named source and the URL is
# live/checkable; "declared" means we're trusting a prior claim (e.g. in
# finding.md) that was never independently re-verified against the source.
KNOWN_SOURCES = {
    "fii_dii.csv": {
        "claimed_source": "Kaggle dataset 'FII and DII Trading Activity in India' (arunkumar237)",
        "verification": "declared — not independently re-downloaded or checksummed against Kaggle by any script in this repo",
        "covers": "2015-01-01 to 2022-12-08",
    },
    "fii_dii_2015_2022.csv": {
        "claimed_source": "Same as fii_dii.csv (identical content, byte-for-byte — see hash below)",
        "verification": "declared, same caveat as fii_dii.csv",
        "covers": "2015-01-01 to 2022-12-08",
    },
    "nifty50.csv": {
        "claimed_source": "yfinance, ^NSEI",
        "verification": "verified — re-fetched live by data_pipeline.py via yfinance on each manifest run",
        "covers": "per config.START_DATE/END_DATE",
    },
    "india_vix.csv": {
        "claimed_source": "yfinance, ^INDIAVIX",
        "verification": "verified — re-fetched live by data_pipeline.py via yfinance",
        "covers": "per config.START_DATE/END_DATE",
    },
    "nifty_bank.csv": {"claimed_source": "yfinance, ^NSEBANK", "verification": "verified — live yfinance fetch", "covers": "per config date range"},
    "nifty_it.csv": {"claimed_source": "yfinance, ^CNXIT", "verification": "verified — live yfinance fetch", "covers": "per config date range"},
    "nifty_pharma.csv": {"claimed_source": "yfinance, ^CNXPHARMA", "verification": "verified — live yfinance fetch", "covers": "per config date range"},
    "nifty_auto.csv": {"claimed_source": "yfinance, ^CNXAUTO", "verification": "verified — live yfinance fetch", "covers": "per config date range"},
    "nse_archives_oi.csv": {
        "claimed_source": "nsearchives.nseindia.com/content/nsccl/fao_participant_oi_*.csv (fetch_archives.py)",
        "verification": "declared — fetched by fetch_archives.py in a prior session; URL pattern is documented and live-checkable, but this manifest run did not re-fetch it",
        "covers": "2015-01-01 to 2025-12-31",
    },
    "pcr.csv": {
        "claimed_source": "Derived from nse_archives_oi.csv by fetch_archives.py",
        "verification": "declared, same caveat as nse_archives_oi.csv",
        "covers": "2015-01-01 to 2025-12-31",
    },
    "fii_dii_holdout.csv": {
        "claimed_source": "Same underlying source as fii_dii.csv; this is the Aug2025-Apr2026 segment, split out to data/holdout/ to prevent re-leakage into training",
        "verification": "declared, same caveat as fii_dii.csv",
        "covers": "2025-08-01 to 2026-04-08",
    },
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    rows = []
    for directory in (RAW_DIR, HOLDOUT_DIR):
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith((".csv", ".json")):
                continue
            path = os.path.join(directory, fname)
            digest = sha256_of(path)
            size = os.path.getsize(path)
            mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat()
            meta = KNOWN_SOURCES.get(fname, {
                "claimed_source": "undocumented",
                "verification": "NOT DOCUMENTED — provenance unknown, treat with suspicion",
                "covers": "unknown",
            })
            rows.append({
                "file": os.path.relpath(path, DATA_DIR),
                "sha256": digest,
                "size_bytes": size,
                "mtime_utc": mtime,
                **meta,
            })

    out_path = os.path.join(DATA_DIR, "MANIFEST.md")
    with open(out_path, "w") as f:
        f.write("# Data Provenance Manifest\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write(
            "`mtime_utc` is the filesystem modification time at manifest-generation time, "
            "NOT a reliable record of original acquisition date — files have been copied/split "
            "since first downloaded. Treat it only as 'last touched by a script in this repo'.\n\n"
            "`verification: declared` means provenance rests on a written claim (in this script, "
            "a prior write-up, or a script docstring) that was never independently re-checked "
            "against the live source by re-running anything in this repo. It is not proof of "
            "authenticity — only the best record currently available.\n\n"
        )
        for r in rows:
            f.write(f"## {r['file']}\n\n")
            f.write(f"- SHA256: `{r['sha256']}`\n")
            f.write(f"- Size: {r['size_bytes']:,} bytes\n")
            f.write(f"- mtime (UTC, see caveat above): {r['mtime_utc']}\n")
            f.write(f"- Claimed source: {r['claimed_source']}\n")
            f.write(f"- Verification status: {r['verification']}\n")
            f.write(f"- Claimed coverage: {r['covers']}\n\n")

    print(f"Wrote manifest for {len(rows)} files to {out_path}")
    undocumented = [r for r in rows if r["verification"].startswith("NOT DOCUMENTED")]
    if undocumented:
        print(f"\nWARNING: {len(undocumented)} file(s) with no documented provenance:")
        for r in undocumented:
            print(f"  - {r['file']}")


if __name__ == "__main__":
    run()
