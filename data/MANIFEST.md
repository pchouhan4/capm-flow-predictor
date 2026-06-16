# Data Provenance Manifest

Generated: 2026-06-16T10:36:51.918783+00:00

`mtime_utc` is the filesystem modification time at manifest-generation time, NOT a reliable record of original acquisition date — files have been copied/split since first downloaded. Treat it only as 'last touched by a script in this repo'.

`verification: declared` means provenance rests on a written claim (in this script, a prior write-up, or a script docstring) that was never independently re-checked against the live source by re-running anything in this repo. It is not proof of authenticity — only the best record currently available.

## raw/fii_dii.csv

- SHA256: `9587738ff7927cfc54a60c2efd9544f5e5e087826cfe303c333380eca04cccbb`
- Size: 49,309 bytes
- mtime (UTC, see caveat above): 2026-06-16T08:20:47.285714+00:00
- Claimed source: Kaggle dataset 'FII and DII Trading Activity in India' (arunkumar237)
- Verification status: declared — not independently re-downloaded or checksummed against Kaggle by any script in this repo
- Claimed coverage: 2015-01-01 to 2022-12-08

## raw/fii_dii_2015_2022.csv

- SHA256: `9587738ff7927cfc54a60c2efd9544f5e5e087826cfe303c333380eca04cccbb`
- Size: 49,309 bytes
- mtime (UTC, see caveat above): 2026-04-09T13:43:33.405874+00:00
- Claimed source: Same as fii_dii.csv (identical content, byte-for-byte — see hash below)
- Verification status: declared, same caveat as fii_dii.csv
- Claimed coverage: 2015-01-01 to 2022-12-08

## raw/fii_dii_api.json

- SHA256: `4a5a0e6fcd2c87dab29a219668cc52e0e0fb881690599b87041cd877798c105d`
- Size: 280 bytes
- mtime (UTC, see caveat above): 2026-04-07T04:14:23.390038+00:00
- Claimed source: undocumented
- Verification status: NOT DOCUMENTED — provenance unknown, treat with suspicion
- Claimed coverage: unknown

## raw/india_vix.csv

- SHA256: `762ccdbfc893b2833db855cc3843c5ba7d447768b0f155a5763f1701620fa1ce`
- Size: 166,791 bytes
- mtime (UTC, see caveat above): 2026-06-16T08:23:23.660727+00:00
- Claimed source: yfinance, ^INDIAVIX
- Verification status: verified — re-fetched live by data_pipeline.py via yfinance
- Claimed coverage: per config.START_DATE/END_DATE

## raw/nifty50.csv

- SHA256: `291a653067a3499af226b4fafde3de558ce908f083dcdf86093398bc1d9fa419`
- Size: 148,794 bytes
- mtime (UTC, see caveat above): 2026-06-16T08:23:22.907082+00:00
- Claimed source: yfinance, ^NSEI
- Verification status: verified — re-fetched live by data_pipeline.py via yfinance on each manifest run
- Claimed coverage: per config.START_DATE/END_DATE

## raw/nifty_auto.csv

- SHA256: `fcdfad347873ad780627ba488a1e7ff22e2eb27f2d8851318e80a0c42cd0aabc`
- Size: 147,372 bytes
- mtime (UTC, see caveat above): 2026-06-16T08:23:27.065444+00:00
- Claimed source: yfinance, ^CNXAUTO
- Verification status: verified — live yfinance fetch
- Claimed coverage: per config date range

## raw/nifty_bank.csv

- SHA256: `d2938e24b8a011c543ba18e1ae1ddc2d27f8060ecbccc809b87eeedc01613b7c`
- Size: 167,106 bytes
- mtime (UTC, see caveat above): 2026-06-16T08:23:24.929441+00:00
- Claimed source: yfinance, ^NSEBANK
- Verification status: verified — live yfinance fetch
- Claimed coverage: per config date range

## raw/nifty_it.csv

- SHA256: `f856ec248eee0397a077e69c77b81f99c4f3bd09226077fe913cbf32b35bca32`
- Size: 142,785 bytes
- mtime (UTC, see caveat above): 2026-06-16T08:23:25.605959+00:00
- Claimed source: yfinance, ^CNXIT
- Verification status: verified — live yfinance fetch
- Claimed coverage: per config date range

## raw/nifty_pharma.csv

- SHA256: `78b69ae2fe20788456d41aba5d07597c210e34f3f188958652d147b4161fb20d`
- Size: 153,296 bytes
- mtime (UTC, see caveat above): 2026-06-16T08:23:26.553139+00:00
- Claimed source: yfinance, ^CNXPHARMA
- Verification status: verified — live yfinance fetch
- Claimed coverage: per config date range

## raw/nse_archives_oi.csv

- SHA256: `1b14343c66a0425055e9b41c9f9e2df1720fa1f14f5bfca43951ee13d6087f0b`
- Size: 205,234 bytes
- mtime (UTC, see caveat above): 2026-04-07T15:20:25.646234+00:00
- Claimed source: nsearchives.nseindia.com/content/nsccl/fao_participant_oi_*.csv (fetch_archives.py)
- Verification status: declared — fetched by fetch_archives.py in a prior session; URL pattern is documented and live-checkable, but this manifest run did not re-fetch it
- Claimed coverage: 2015-01-01 to 2025-12-31

## raw/pcr.csv

- SHA256: `5c638b603f10073f8b13ffeb63c1a2e298cfa0f3783e78343fdc0eeca750faa9`
- Size: 48,550 bytes
- mtime (UTC, see caveat above): 2026-04-07T15:20:25.664961+00:00
- Claimed source: Derived from nse_archives_oi.csv by fetch_archives.py
- Verification status: declared, same caveat as nse_archives_oi.csv
- Claimed coverage: 2015-01-01 to 2025-12-31

## holdout/fii_dii_holdout.csv

- SHA256: `afd2bacc8c5f5770574db890991c60f7c4dc335339dae3b363ed1b7840f05673`
- Size: 4,391 bytes
- mtime (UTC, see caveat above): 2026-06-16T08:20:47.287208+00:00
- Claimed source: Same underlying source as fii_dii.csv; this is the Aug2025-Apr2026 segment, split out to data/holdout/ to prevent re-leakage into training
- Verification status: declared, same caveat as fii_dii.csv
- Claimed coverage: 2025-08-01 to 2026-04-08

