"""
Layer 5 — Validity diagnostics beyond the basic episode-correction.

Addresses gaps identified in external review of the previous correction pass:

1. Multiple-testing correction (Bonferroni + Benjamini-Hochberg FDR) applied
   to the 16 main sector/horizon p-values.
2. Permutation test — shuffles which trading days are "flagged" (preserving
   the number and run-length structure of flags) and recomputes the episode
   hit rate under the null, 5000x. This makes no asymptotic/independence
   assumption at all, unlike the binomial sign test.
3. Block bootstrap — resamples contiguous blocks of trading days (not
   individual days) to estimate an effective sample size for the one
   borderline result (NIFTY_BANK, 3-day). If regimes drive the result, the
   block-bootstrap p-value will be much weaker than the naive or
   episode-corrected one.
4. Economic significance — mean forward return and a Sharpe-like ratio on
   flagged vs unflagged days, since hit rate alone can't distinguish a
   worthless 53% edge from a valuable 51% one.

Run after run.py (needs data/clean/flagged.csv).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from config import SECTORS, CLEAN_DIR, RESULTS_DIR
from test_prediction import get_flag_episodes, test_forward_prediction_episodes, sign_test

RNG = np.random.default_rng(42)
N_PERMUTATIONS = 5000
N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# 1. Multiple testing correction
# ---------------------------------------------------------------------------

def apply_multiple_testing_correction(main_results_path):
    df = pd.read_csv(main_results_path)
    pvals = df["p_value_corrected"].values
    n = len(pvals)

    # Bonferroni
    df["p_bonferroni"] = np.minimum(pvals * n, 1.0)

    # Benjamini-Hochberg FDR
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]  # enforce monotonicity
    bh_full = np.empty(n)
    bh_full[order] = np.minimum(bh, 1.0)
    df["p_bh_fdr"] = bh_full

    df["significant_bonferroni"] = df["p_bonferroni"] < 0.05
    df["significant_bh_fdr"] = df["p_bh_fdr"] < 0.05

    print(f"\n  Bonferroni threshold: 0.05/{n} = {0.05/n:.5f}")
    print(f"  Significant after Bonferroni: {df['significant_bonferroni'].sum()}/{n}")
    print(f"  Significant after BH-FDR: {df['significant_bh_fdr'].sum()}/{n}")

    return df


# ---------------------------------------------------------------------------
# 2. Permutation test
# ---------------------------------------------------------------------------

def permutation_test(df, sector_prefix, forward_days, n_perm=N_PERMUTATIONS):
    """
    Null distribution: shuffle the flag run-lengths to random start positions
    (preserves how many flags exist and how clustered they are, destroys any
    real relationship between flag timing and forward CAPM error).
    """
    episodes = get_flag_episodes(df)
    run_lengths = [end - start + 1 for start, end in episodes]
    n_days = len(df)

    observed_signs = test_forward_prediction_episodes(df, sector_prefix, forward_days)
    if len(observed_signs) < 10:
        return np.nan, np.nan

    _, _, observed_hit_rate, _ = sign_test(observed_signs)
    observed_stat = abs(observed_hit_rate - 0.5)

    null_stats = []
    flag_col_backup = df["anomaly_flag"].copy()

    for _ in range(n_perm):
        new_flags = np.zeros(n_days, dtype=int)
        # place each run length at a random non-overlapping-ish start;
        # simplest valid approach: random start, allow overlap suppression naturally
        # via get_flag_episodes recomputation downstream.
        for rl in run_lengths:
            start = RNG.integers(0, max(1, n_days - rl))
            new_flags[start:start + rl] = 1
        df["anomaly_flag"] = new_flags

        signs = test_forward_prediction_episodes(df, sector_prefix, forward_days)
        if len(signs) < 5:
            continue
        _, _, hr, _ = sign_test(signs)
        if not np.isnan(hr):
            null_stats.append(abs(hr - 0.5))

    df["anomaly_flag"] = flag_col_backup  # restore

    if not null_stats:
        return observed_hit_rate, np.nan

    null_stats = np.array(null_stats)
    p_perm = (null_stats >= observed_stat).mean()
    return observed_hit_rate, p_perm


# ---------------------------------------------------------------------------
# 3. Block bootstrap — effective sample size for the borderline result
# ---------------------------------------------------------------------------

def block_bootstrap_test(df, sector_prefix, forward_days, block_size=20, n_boot=N_BOOTSTRAP):
    """
    Resample contiguous BLOCKS of trading days (with replacement) to build a
    bootstrap distribution of the hit rate. If a handful of regimes drive the
    result, block resampling will show much wider variance than the
    naive/episode-corrected tests assume, and the effective n
    (n_naive / variance_inflation_factor) will be small.
    """
    error_col = f"{sector_prefix}_capm_error"
    if error_col not in df.columns:
        return None

    n_days = len(df)
    n_blocks = n_days // block_size

    # Build the full per-day flagged-day sign series once (naive, for variance estimation)
    flagged_idx = df.index[df["anomaly_flag"] == 1]
    day_signs = {}
    for date in flagged_idx:
        loc = df.index.get_loc(date)
        end = min(loc + forward_days + 1, n_days)
        if loc + 1 >= n_days:
            continue
        fwd = df[error_col].iloc[loc + 1:end]
        if len(fwd) < max(1, forward_days * 0.5):
            continue
        day_signs[loc] = np.sign(fwd.mean())

    if len(day_signs) < 10:
        return None

    boot_hit_rates = []
    block_starts = np.arange(0, n_days - block_size, block_size)

    for _ in range(n_boot):
        chosen_blocks = RNG.choice(block_starts, size=n_blocks, replace=True)
        signs_in_sample = []
        for b in chosen_blocks:
            for loc in range(b, min(b + block_size, n_days)):
                if loc in day_signs:
                    signs_in_sample.append(day_signs[loc])
        if len(signs_in_sample) < 5:
            continue
        signs_in_sample = np.array(signs_in_sample)
        hr = (signs_in_sample > 0).mean()
        boot_hit_rates.append(hr)

    if not boot_hit_rates:
        return None

    boot_hit_rates = np.array(boot_hit_rates)
    naive_n = len(day_signs)
    naive_var = 0.25 / naive_n  # binomial variance under p=0.5, n=naive_n
    boot_var = boot_hit_rates.var()
    vif = boot_var / naive_var if naive_var > 0 else np.nan
    effective_n = naive_n / vif if vif and vif > 0 else np.nan

    # Two-sided bootstrap p-value: how often does |hit_rate - 0.5| from the
    # bootstrap distribution, centered at its own mean exceed the observed
    # deviation from 0.5 — using the boot distribution's spread around 0.5.
    observed_hr = (np.array(list(day_signs.values())) > 0).mean()
    centered = boot_hit_rates - boot_hit_rates.mean() + 0.5
    p_boot = (np.abs(centered - 0.5) >= abs(observed_hr - 0.5)).mean()

    return {
        "naive_n": naive_n,
        "observed_hit_rate": observed_hr,
        "block_bootstrap_var": boot_var,
        "naive_binomial_var": naive_var,
        "variance_inflation_factor": vif,
        "effective_n": effective_n,
        "p_block_bootstrap": p_boot,
    }


# ---------------------------------------------------------------------------
# 4. Economic significance
# ---------------------------------------------------------------------------

def economic_significance(df, sector_prefix, forward_days):
    error_col = f"{sector_prefix}_capm_error"
    return_col = f"{sector_prefix}_return"
    if error_col not in df.columns:
        return None

    flagged_idx = df.index[df["anomaly_flag"] == 1]
    flagged_fwd_returns = []
    for date in flagged_idx:
        loc = df.index.get_loc(date)
        end = min(loc + forward_days + 1, len(df))
        if loc + 1 >= len(df):
            continue
        fwd = df[return_col].iloc[loc + 1:end]
        if len(fwd) < max(1, forward_days * 0.5):
            continue
        flagged_fwd_returns.append(fwd.mean())

    all_fwd_returns = []
    for loc in range(len(df) - forward_days - 1):
        fwd = df[return_col].iloc[loc + 1:loc + forward_days + 1]
        if len(fwd) == forward_days:
            all_fwd_returns.append(fwd.mean())

    if not flagged_fwd_returns or not all_fwd_returns:
        return None

    flagged_fwd_returns = np.array(flagged_fwd_returns)
    all_fwd_returns = np.array(all_fwd_returns)

    def sharpe_like(x):
        return x.mean() / x.std() if x.std() > 0 else np.nan

    return {
        "mean_return_flagged": flagged_fwd_returns.mean(),
        "mean_return_unconditional": all_fwd_returns.mean(),
        "sharpe_like_flagged": sharpe_like(flagged_fwd_returns),
        "sharpe_like_unconditional": sharpe_like(all_fwd_returns),
        "n_flagged_days": len(flagged_fwd_returns),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("VALIDITY DIAGNOSTICS")
    print("=" * 60)

    # 1. Multiple testing correction
    main_path = os.path.join(RESULTS_DIR, "main_results.csv")
    corrected_df = apply_multiple_testing_correction(main_path)
    corrected_df.to_csv(main_path, index=False)
    print(f"\n  Updated {main_path} with Bonferroni/BH columns")

    # Load flagged data for permutation/bootstrap/economic checks
    flagged_path = os.path.join(CLEAN_DIR, "flagged.csv")
    df = pd.read_csv(flagged_path, index_col="date", parse_dates=True)

    print("\n" + "-" * 60)
    print("PERMUTATION TEST (5000 shuffles, preserves flag run-lengths)")
    print("-" * 60)
    perm_results = []
    for name in SECTORS:
        prefix = name.lower()
        for fwd in [1, 3, 5, 10]:
            hr, p_perm = permutation_test(df, prefix, fwd)
            perm_results.append({"sector": name, "forward_days": fwd,
                                  "hit_rate": hr, "p_permutation": p_perm})
            print(f"  {name:15s} | fwd={fwd:2d}d | hit={hr:.3f} | p_perm={p_perm:.4f}"
                  if not np.isnan(hr) else f"  {name:15s} | fwd={fwd:2d}d | insufficient data")
    pd.DataFrame(perm_results).to_csv(os.path.join(RESULTS_DIR, "permutation_test.csv"), index=False)

    print("\n" + "-" * 60)
    print("BLOCK BOOTSTRAP (effective sample size, 20-day blocks, 2000 resamples)")
    print("Focused on NIFTY_BANK 3-day — the one cell that survived episode correction")
    print("-" * 60)
    bb = block_bootstrap_test(df, "nifty_bank", 3)
    if bb:
        for k, v in bb.items():
            print(f"  {k}: {v}")
        pd.DataFrame([bb]).to_csv(os.path.join(RESULTS_DIR, "block_bootstrap_nifty_bank_3d.csv"), index=False)

    print("\n" + "-" * 60)
    print("ECONOMIC SIGNIFICANCE (mean forward return, Sharpe-like ratio)")
    print("-" * 60)
    econ_results = []
    for name in SECTORS:
        prefix = name.lower()
        for fwd in [1, 3, 5, 10]:
            econ = economic_significance(df, prefix, fwd)
            if econ:
                econ["sector"] = name
                econ["forward_days"] = fwd
                econ_results.append(econ)
                print(f"  {name:15s} | fwd={fwd:2d}d | mean_ret_flagged={econ['mean_return_flagged']*100:.3f}% "
                      f"| mean_ret_unconditional={econ['mean_return_unconditional']*100:.3f}% "
                      f"| sharpe_flagged={econ['sharpe_like_flagged']:.3f}")
    pd.DataFrame(econ_results).to_csv(os.path.join(RESULTS_DIR, "economic_significance.csv"), index=False)

    print(f"\nAll validity diagnostics saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
