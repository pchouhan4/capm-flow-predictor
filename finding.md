# FII/DII Flow Divergence as a Predictor of CAPM Pricing Error Direction: A Null Finding (Corrected)

**Dataset:** Indian equity markets, 2015-01-01 to 2022-12-08 (clean, continuous training window)
**Holdout:** Aug 2025 - Apr 2026 (167 days), isolated and never used in training — see Section 6
**Signals tested:** 2-of-3 concordance (flow divergence, PCR deviation, VIX implied-realized vol spread)
**Run date:** 2026-06-16
**Supersedes:** `finding.zip` (2026-04-07) — that version was generated from a dataset that, two days later, had the intended out-of-sample holdout merged into it, and its reported statistics no longer match anything reproducible in this repo. See Section 7 for the full list of corrections made.

---

## Abstract

We test whether daily FII/DII equity flow divergence predicts the direction of CAPM pricing errors in four Indian sector indices (Bank, IT, Pharma, Auto) over 1-, 3-, 5-, and 10-day forward horizons, using 1,883 days of real equity FII/DII flow data (2015-01-01 to 2022-12-08). We find no broad predictive edge: across 16 sector/horizon cells, none survive Bonferroni/BH correction for multiple comparisons, and the robustness sweep shows significance in 5.6% of 160 parameter combinations — statistically indistinguishable from the 5% expected by chance. We also test, with a reproducible and properly-constructed script (1,750-day overlap, comparing flow-equivalent change rather than raw level), whether F&O participant OI proxies equity cash flows: it does not — sign agreement is 48.9%, indistinguishable from chance and stable across five sub-periods, which is a materially different and weaker finding than a prior claim that F&O OI was "structurally inverted" (that claim used only 100 days and a stock/flow construction mismatch that manufactures spurious anti-correlation).

---

## 1. Hypothesis

**Research question:** Do days when FII and DII flows are simultaneously extreme and divergent coincide with predictable CAPM mispricing direction in Indian sector indices?

**Mechanism (prior):** When FII sells heavily while DII buys (or vice versa), one side is systematically wrong about fair value. If large enough, this should bias next-day(s) sector returns relative to a rolling-beta CAPM prediction.

---

## 2. Data and Methodology

### 2.1 Market data
yfinance, Nifty 50 (^NSEI), India VIX (^INDIAVIX), 4 sector indices, 2015-01-01 to 2022-12-08.

### 2.2 Institutional flow data
`data/raw/fii_dii.csv` — 1,883 rows, 2015-01-01 to 2022-12-08, INR Crores. This is the **only** FII/DII data used for training and the main test. The Aug 2025–Apr 2026 segment that was previously merged into this file has been moved to `data/holdout/fii_dii_holdout.csv` and is excluded from `config.py`'s date range, so it cannot be silently re-absorbed into a future run.

### 2.3 Signal construction
Unchanged from the original design: flow divergence (FII−DII, 252-day percentile rank), PCR deviation (z-score vs 30-day baseline, 252-day percentile rank), vol spread (VIX − 20-day realized vol, 252-day percentile rank). Verified: all three use trailing-only windows (`.rolling(window)`, no centering) — no look-ahead bias in signal construction.

### 2.4 CAPM pricing error
Rolling 60-day OLS per sector, beta estimated strictly on the prior 60 days (`iloc[i-60:i]`, excludes day i) — verified no leakage.

### 2.5 Anomaly detection
2-of-3 signals simultaneously in top/bottom 10th percentile (unchanged).

### 2.6 Statistical test — corrected for autocorrelation

**This is the main methodological fix in this revision.** The original sign test treated every flagged day as an independent Bernoulli trial. But flagged days cluster: in the training data, **65% of consecutive flagged-day pairs fall within 3 trading days of each other**, so their forward-error windows overlap and their signs are correlated, not i.i.d. — this inflates apparent significance.

Fix (`test_forward_prediction_episodes` in `test_prediction.py`): consecutive flagged days are collapsed into one episode, evaluated from the day after the episode ends. Episodes whose forward window would still overlap the next episode's trigger are dropped rather than counted, so retained trials are genuinely non-overlapping. This trades sample size (n drops from ~170 to ~55-105 depending on horizon) for validity. Both the naive and corrected numbers are reported in `results/main_results.csv` for transparency — **the corrected column is the one that should be trusted.**

---

## 3. Results

### 3.1 Main test (corrected, non-overlapping episodes)

| Sector | 1d (n,hit,p) | 3d | 5d | 10d |
|---|---|---|---|---|
| NIFTY_BANK | n=105, 42.9%, p=0.17 | n=86, **38.4%, p=0.040 (sig)** | n=77, 49.4%, p=1.0 | n=55, 47.3%, p=0.79 |
| NIFTY_IT | n=105, 49.5%, p=1.0 | n=86, 58.1%, p=0.16 | n=77, 51.9%, p=0.82 | n=55, 47.3%, p=0.79 |
| NIFTY_PHARMA | n=105, 51.4%, p=0.85 | n=86, 50.0%, p=1.0 | n=77, 44.2%, p=0.36 | n=55, 36.4%, p=0.058 |
| NIFTY_AUTO | n=105, 46.7%, p=0.56 | n=86, 51.2%, p=0.91 | n=77, 50.6%, p=1.0 | n=55, 45.5%, p=0.59 |

1 of 16 cells significant at α=0.05 — at-or-below the ~0.8 expected by chance with 16 comparisons, no Bonferroni applied. NIFTY_BANK's 3-day hit rate is *below* 50%, i.e. anti-predictive of the naive directional hypothesis.

### 3.2 Walk-forward (3-day forward, corrected test, 3 full 2-year windows + 1 partial year)

| Period | BANK | IT | PHARMA | AUTO |
|---|---|---|---|---|
| 2016–2017 (n=18) | 38.9% | 38.9% | 66.7% | 55.6% |
| 2018–2019 (n=31) | 41.9% | 67.7% | 38.7% | 45.2% |
| 2020–2021 (n=26) | **23.1%, p=0.009** | **73.1%, p=0.029** | 61.5% | 46.2% |
| 2022 partial (n=10) | 70.0% | 20.0% | 30.0% | 70.0% |

NIFTY_BANK's hit rate ranges from 23% to 70% across periods — not a stable relationship. The 2020-2021 significance for both BANK and IT looks COVID-regime-specific, not a generalizable edge. The 2022 row is a partial year (Jan–Dec 8 only, n=10) due to the data ending mid-December; treat it as low-power.

*Note: the previous "2024-2025" walk-forward row has been removed. It no longer exists as a meaningful period — the underlying data has zero rows for 2023-2024, and what little remained (2025-26) was the leaked holdout, now correctly excluded.*

### 3.3 Signal decomposition (3-day forward, corrected test)

| Signal | BANK | IT | PHARMA | AUTO |
|---|---|---|---|---|
| flow_div alone (n=165) | 50.9% | 47.9% | 48.5% | 52.1% |
| pcr_dev alone (n=106) | 44.3% | 60.4% (p=0.041) | 46.2% | 49.1% |
| vol_spread alone (n=61) | 39.3% | 50.8% | 44.3% | 47.5% |

No signal in isolation shows a reliable edge either (one marginal hit for PCR/IT, not replicated elsewhere).

### 3.4 Robustness sweep

**9 of 160 parameter combinations (5.6%) show p<0.05** — now genuinely consistent with the ~5% expected under the null, and (unlike the previous write-up) this number is reproducible directly from `results/robustness.csv`. 8 of the 9 significant combinations are NIFTY_BANK at 3-day forward; the recurring pattern (hit rate 29%-41%, consistently below 50%) suggests there may be *something* idiosyncratic to NIFTY_BANK at the 3-day horizon, but it does not survive walk-forward (Section 3.2: direction flips by period) and is not strong enough to call a real edge.

---

## 4. F&O OI Proxy Check — Now Reproducible, and the Original Claim Doesn't Survive It

This section has been revised twice. The original (`finding.zip`) cited a Spearman r=−0.019 / 28%-sign-agreement claim with no script or data backing it anywhere in the repo. A first fix (`validate_oi_proxy.py` rev.1) made it reproducible but only checked a 100-day window and compared the F&O OI *level* (a stock, cumulative outstanding position) against equity *flow* (a daily transaction) — and got r=−0.011, 36% sign agreement, which read as "confirms the original claim, F&O OI is anti-correlated/backwards."

**That confirmation does not survive a larger, correctly-constructed test.** Two fixes, both raised independently by review:

1. **More data.** The training equity data (2015-2022) overlaps the OI archive for 1,750 days, not just the 100-day holdout slice.
2. **Stock vs. flow mismatch.** F&O net OI is a level; equity net flow is a daily transaction amount. Comparing them directly conflates "is the *outstanding position* growing in the same direction as today's equity buying" with "is *today's* F&O activity in the same direction as today's equity activity." The economically meaningful comparison is the **day-over-day change** in F&O net OI (the flow-equivalent), not the level.

| Construction | n | Spearman r | Sign agreement |
|---|---|---|---|
| LEVEL (as originally tested) | 1,750 | −0.017 (p=0.47) | 40.5% |
| **CHANGE (flow-equivalent, the defensible construction)** | 1,749 | **−0.002 (p=0.95)** | **48.9%** |

Broken out by sub-period (CHANGE construction):

| Period | n | r | Sign agreement |
|---|---|---|---|
| 2015-2016 | 425 | +0.057 | 51.5% |
| 2017-2018 | 430 | −0.043 | 45.3% |
| 2019-2020 | 438 | −0.012 | 51.1% |
| 2021-2022 | 356 | +0.017 | 47.5% |
| Aug-Dec 2025 | 100 | +0.012 | 49.0% |

**Revised conclusion: F&O participant OI carries no detectable relationship to equity flow direction — it is uninformative noise (sign agreement ≈ chance, stable at ~45-52% across five different multi-year periods), not a "structurally inverted" or anti-correlated signal.** The original "FII hedges equity sells by going long futures, so F&O OI direction is backwards" mechanism is a plausible story but is **not supported** by this data once tested properly — if that hedging mechanism reliably drove a sign inversion, sign agreement should sit consistently below 50% across periods; instead it straddles 50% with no consistent direction.

The practical implication is unchanged in one sense (don't use F&O OI level as an equity-flow proxy — it doesn't work) but the *reason* is different and weaker than originally claimed: it's not "actively misleading," it's "uninformative." That distinction matters if anyone downstream were to consider, e.g., inverting the F&O signal to use it as a contrarian proxy — that would not work either, since there's no stable relationship in either direction.

---

## 5. Holdout (Aug 2025 – Apr 2026) — Could Not Be Validated, and We're Saying So

We attempted a genuine holdout check (`holdout_check.py`) rather than skip this honestly. Result: **it doesn't work, and shouldn't be presented as if it does.**

The main methodology needs a 252-day rolling window. The holdout is only 167 days, separated from training by a 2.5-year gap with zero FII/DII coverage (2023-2024). There's no way to compute a "trailing year" feature for the holdout that means the same thing it meant during training. A reduced 60-day-window variant, confined entirely to holdout data, was run as a fallback — it produced only **1 flagged day** across the whole holdout period (insufficient for any statistical test; see `results/holdout_check.csv`).

**Conclusion: with the data currently available, this project cannot produce a valid out-of-sample validation.** Closing this gap would require backfilling FII/DII data for 2023-2024, not a code fix.

---

## 5.5 Extended Validity Diagnostics (`validity_checks.py`)

Four additional checks, run after external review of the corrections above, to close remaining gaps: whether episode-correction alone was enough, whether 16 comparisons needed an explicit multiple-testing correction, and whether hit-rate significance corresponds to anything economically meaningful.

### Multiple testing correction

Bonferroni and Benjamini-Hochberg FDR applied to the 16 main sector/horizon p-values (now columns `p_bonferroni`, `p_bh_fdr` in `results/main_results.csv`).

- Bonferroni threshold: 0.05/16 = 0.0031
- NIFTY_BANK 3-day (p=0.0399) does not clear it.
- **0 of 16 cells significant after either correction.**

This is the strongest statement of the null finding in this document: even the one cell that survived episode-correction does not survive accounting for the fact that 16 hypotheses were tested.

### Permutation test (5000 shuffles, preserves flag run-length structure, no distributional assumptions)

| Sector | 3d hit rate | p (permutation) |
|---|---|---|
| NIFTY_BANK | 38.4% | 0.038 |
| All other 15 cells | — | p > 0.07, mostly p > 0.4 |

The permutation p for NIFTY_BANK (0.038) closely tracks the parametric binomial p (0.040) — confirming the binomial approximation itself wasn't the source of the apparent significance. (Full table: `results/permutation_test.csv`.)

### Block bootstrap (20-day blocks, 2000 resamples) — NIFTY_BANK 3-day only

| Metric | Value |
|---|---|
| Naive n | 170 |
| Variance inflation factor (block vs. naive binomial variance) | 1.41 |
| Effective n | ≈120 |
| Block-bootstrap p-value | 0.027 |

Regime clustering inflates the variance by ~41%, but this alone doesn't explain away the NIFTY_BANK 3-day result — it takes the multiple-testing correction above to do that. Full output: `results/block_bootstrap_nifty_bank_3d.csv`.

### Economic significance (mean forward return and Sharpe-like ratio, flagged vs. unconditional)

| Sector | 3d mean return, flagged | 3d mean return, unconditional | Sharpe-like, flagged |
|---|---|---|---|
| NIFTY_BANK | -0.032% | +0.066% | -0.024 |
| NIFTY_IT | +0.161% | +0.073% | +0.188 |
| NIFTY_PHARMA | +0.152% | +0.023% | +0.134 |
| NIFTY_AUTO | +0.067% | +0.042% | +0.054 |

All magnitudes are small (Sharpe-like ratios well under 0.3), and none of IT/PHARMA/AUTO's flagged-day deltas correspond to a statistically significant hit rate in Section 3.1 — a reminder that hit-rate significance and economic magnitude are different questions, and neither would plausibly survive transaction costs even where the sign looks favorable. Full table: `results/economic_significance.csv`.

**Net effect of this diagnostic pass: the null finding is now airtight, not just probable.**

---

## 6. Limitations

- **Non-independence correction trades power for validity** — the corrected test has roughly half the naive n. A real effect of moderate size could be missed. (Section 5.5 shows the residual autocorrelation effect is modest — VIF≈1.4 — so this isn't a large hidden cost.)
- **Sign test only captures direction, not magnitude** — addressed partially in Section 5.5's economic-significance check, which shows the magnitudes involved are small regardless.
- **2023-2024 data gap** — see Section 5. Blocks any holdout validation under the current design.
- **PCR data** is included in the 2-of-3 concordance filter in this revision (previous version ran 2-of-2 without it); results above already reflect 3-signal concordance.
- **Single equity flow source**, not cross-validated against a second provider.
- **Data provenance is now hashed, not fully verified.** `generate_manifest.py` produces `data/MANIFEST.md` with SHA256/size/mtime for every raw file, and explicitly labels each file's source claim as `verified` (yfinance/NSE data this repo's own code re-fetches live and can re-check on demand) or `declared` (the Kaggle FII/DII CSV and the NSE F&O OI archive — provenance rests on a written claim that nothing here independently re-validates against the original source). This closes the "could be silently swapped without anyone noticing" gap but does **not** prove the Kaggle data is what it claims to be — that would require re-downloading from Kaggle and diffing, which hasn't been done.
- **2023-2024 FII/DII gap remains unresolved.** It's now documented prominently (Section 5) rather than glossed over, but documenting a hole isn't the same as filling it. No holdout validation is possible until this is backfilled with real data.
- **CAPM may be a noisy benchmark for this question — not yet tested against alternatives.** This is the most substantive open item. A real flow effect could exist and still not register against rolling-beta CAPM pricing-error sign; it might instead show up in residual returns, sector-relative returns, factor-neutral (e.g. Fama-French-style) returns, or volatility forecasts. This project disproves one specific formulation of the hypothesis (CAPM error sign), not the underlying hypothesis that flows matter. Re-testing against a different target variable is a real next step, not implemented here — it would need either factor data (not currently in this repo) or a redesigned signal-construction step, not just a parameter tweak.

---

## 7. Corrections from the Previous Version (`finding.zip`, 2026-04-07)

For transparency, an explicit diff of what changed and why:

1. **Holdout leakage fixed.** `data/raw/fii_dii.csv` had the Aug2025-Apr2026 holdout merged into it by the time of the previous run. It has been split back into `data/raw/fii_dii.csv` (train, ≤2022-12-08) and `data/holdout/fii_dii_holdout.csv`, and `config.py`'s `END_DATE` now hard-bounds the training window.
2. **Robustness sweep number corrected.** Previous claim: "13/160 (8.1%)". That number didn't match `results/robustness.csv` even at the time (actual was 34/160 = 21.25% on the leaked dataset). Current, correct, reproducible number: 9/160 (5.6%) on clean data.
3. **Sign test independence fixed.** Previous version used every flagged day as an independent trial despite heavy autocorrelation. The corrected episode-based test is now the default; naive numbers are kept alongside for comparison only.
4. **F&O proxy claim made reproducible, then found not to replicate as originally stated.** First pass (`validate_oi_proxy.py` rev.1) made the number reproducible but only checked a 100-day window using a level-vs-flow construction mismatch, and got a result that read as confirming the original "structurally inverted" claim. A second pass (rev.2, prompted by review questioning the construction and overlap period) used the full 1,750-day overlap and the correct flow-equivalent (day-over-day OI change) construction — and found **no relationship at all** (48.9% sign agreement, chance level, stable across 5 sub-periods), not an inversion. See Section 4.
5. **Walk-forward periods fixed.** Removed the "2024-2025" period (no longer meaningful given the data gap and the holdout exclusion); replaced with the real partial-year 2022 window.
6. **Holdout validation attempted honestly.** Rather than silently dropping it or faking a result, `holdout_check.py` documents exactly why it can't currently produce a valid result.
7. **Extended validity diagnostics added** (`validity_checks.py`, Section 5.5), in response to external review: multiple-testing correction (Bonferroni/BH), a distribution-free permutation test, a block bootstrap to quantify how much regime-clustering inflates significance, and economic-significance metrics (mean return, Sharpe-like ratio) since hit-rate alone doesn't establish exploitability.
8. **Data provenance manifest added** (`generate_manifest.py` → `data/MANIFEST.md`), in response to further review: SHA256 + verification status (independently re-fetchable vs. merely declared) for every raw data file.

---

## 8. Conclusions

1. **Null finding holds, and is now airtight rather than merely probable.** FII/DII flow divergence (2-of-3 concordance with PCR and vol spread) does not predict CAPM error direction in Indian equities, 2015–2022. Holds under: the corrected non-overlapping-episode sign test, a distribution-free permutation test, and — decisively — after Bonferroni/BH correction for the 16 sector/horizon comparisons (0/16 significant).
2. **NIFTY_BANK at 3-day forward was the one cell that looked interesting**, surviving episode-correction (p=0.040) and the permutation test (p=0.038), with a block-bootstrap effective n of ~120 (down from 170, but not collapsed). It is killed specifically by the multiple-testing correction, and independently undermined by walk-forward instability (23%-70% hit rate range across regimes) and a negative, noise-level economic effect (Sharpe-like -0.024 at 3-day). Not exploitable by any measure applied here.
3. **F&O participant OI is uninformative about equity flow direction — but not in the way originally claimed.** The original (and first-revision) claim that it was "structurally inverted" / "anti-correlated" does not survive a properly-constructed, full-overlap test: sign agreement is 48.9%, chance-level, stable across five sub-periods spanning 2015-2025. The practical conclusion is the same (don't use it as a proxy) but the mechanism claimed in the original write-up (FII hedging causes a sign inversion) is not supported by the data — it's noise, not a backwards signal. This is the clearest example in this project of why "reproducible" and "correct" are different bars: rev.1 was fully reproducible and still wrong, because the construction (level vs. flow, 100 days vs. 1,750) was the actual problem.
4. **A real out-of-sample validation is not currently possible** without backfilling the 2023-2024 FII/DII gap. This is now stated directly rather than implied as a "next step" while the holdout was quietly already consumed.
5. **Remaining honest gaps, ranked by what it would take to close them**: (a) data provenance — partially addressed (`data/MANIFEST.md` hashes everything and labels what's independently re-verifiable vs. merely declared), but the Kaggle source itself is still unverified against its origin; (b) the 2023-2024 gap — needs new data, not more analysis; (c) the CAPM target-variable choice — the most interesting open question, requires re-testing against residual/sector-relative/factor-neutral returns, a genuine scope expansion rather than a bug fix.

---

## Appendix: Pipeline Summary

| Layer | File | Description |
|---|---|---|
| L0 | data_pipeline.py | Market data (yfinance) + FII/DII CSV ingestion + alignment |
| L1 | signals.py | Flow divergence, PCR deviation, VIX vol spread (252-day percentile) |
| L2 | capm.py | Rolling 60-day OLS CAPM per sector; pricing error computation |
| L3 | detector.py | 2-of-N concordance flag at configurable percentile threshold |
| L4 | test_prediction.py | Episode-corrected sign test + walk-forward + decomposition + robustness sweep |
| L5 | validity_checks.py | Bonferroni/BH correction + permutation test + block bootstrap + economic significance (new) |
| — | validate_oi_proxy.py | Reproducible F&O-OI-vs-equity-flow comparison (new) |
| — | holdout_check.py | Honest holdout attempt + documented failure mode (new) |
| — | generate_manifest.py | SHA256 + source-verification-status manifest for all raw data files (new) |
| — | run.py | End-to-end orchestrator (main test only; holdout/proxy/validity scripts run separately) |
| — | fetch_archives.py | NSE Archives PCR fetch (F&O OI → PCR only; FII/DII export disabled) |

**Reproducibility:** `python3 run.py` → `python3 validity_checks.py` for the main result and its diagnostics; `python3 validate_oi_proxy.py` and `python3 holdout_check.py` for the two ancillary checks. All numbers in this document were generated by these exact scripts on 2026-06-16.
