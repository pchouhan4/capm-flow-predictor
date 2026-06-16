"""
Layer 4 — The Prediction Test

THE ENTIRE POINT OF THIS PROJECT.

For every day where the anomaly detector fires:
  - Look at CAPM pricing error over the next 1/3/5/10 sessions
  - Is the error consistently positive or negative? (sign test)
  - If yes → the signal has forward predictive content
  - If no → the signal is noise

Also runs:
  - Walk-forward validation (5 non-overlapping periods)
  - Robustness sweep (vary window, percentile, forward horizon)
  - Signal decomposition (which signal drives prediction?)
  - Sector comparison (FII-heavy vs domestic sectors)
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

from config import (
    FORWARD_WINDOWS, SECTORS, CLEAN_DIR, RESULTS_DIR,
    WINDOW_SWEEP, PERCENTILE_SWEEP,
)
from signals import compute_all_signals
from detector import compute_anomaly_flags


# ---------------------------------------------------------------------------
# Core prediction test
# ---------------------------------------------------------------------------

def sign_test(signs):
    """
    Test whether a sequence of +1/-1 signs deviates from 50/50.
    Returns (n_positive, n_total, hit_rate, p_value).
    """
    signs = signs.dropna()
    signs = signs[signs != 0]  # exclude exact zeros
    n_total = len(signs)

    if n_total < 5:
        return 0, n_total, np.nan, np.nan

    n_positive = (signs > 0).sum()
    hit_rate = n_positive / n_total

    # Two-sided binomial test
    p_value = stats.binomtest(n_positive, n_total, 0.5).pvalue

    return int(n_positive), n_total, hit_rate, p_value


def test_forward_prediction(df, sector_prefix, forward_days):
    """
    For each flagged day, compute the mean CAPM error over the next
    `forward_days` sessions and record its sign.

    NAIVE version: treats every flagged day as an independent trial. When
    flags cluster (consecutive flagged days), their forward windows overlap
    heavily and the resulting signs are autocorrelated, not i.i.d. — this
    inflates apparent significance. Use `test_forward_prediction_episodes`
    for a statistically valid test; this version is kept only for comparison.

    Returns Series of signs (+1 or -1) for each flagged day.
    """
    error_col = f"{sector_prefix}_capm_error"
    if error_col not in df.columns:
        return pd.Series(dtype=float)

    flagged_idx = df.index[df["anomaly_flag"] == 1]
    signs = []

    for date in flagged_idx:
        loc = df.index.get_loc(date)
        # Forward window: next forward_days sessions
        end = min(loc + forward_days + 1, len(df))
        if loc + 1 >= len(df):
            continue

        forward_errors = df[error_col].iloc[loc + 1:end]
        if len(forward_errors) < max(1, forward_days * 0.5):
            continue  # not enough forward data

        mean_error = forward_errors.mean()
        signs.append(np.sign(mean_error))

    return pd.Series(signs)


def get_flag_episodes(df):
    """
    Collapse consecutive flagged days into episodes. Returns a list of
    (start_loc, end_loc) integer positions (inclusive) for each episode.
    """
    flag = df["anomaly_flag"].values
    episodes = []
    start = None
    for i, v in enumerate(flag):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(flag) - 1))
    return episodes


def test_forward_prediction_episodes(df, sector_prefix, forward_days):
    """
    Statistically valid version of the sign test.

    Each *episode* (a run of consecutive flagged days) is treated as ONE
    trial, evaluated from the day after the episode ends — this removes the
    within-episode autocorrelation that the naive day-by-day version ignores.

    Episodes whose forward window would still overlap the next episode's
    trigger day are dropped (not merged) so that the retained trials are
    non-overlapping and therefore approximately independent. This trades
    sample size for statistical validity.

    Returns Series of signs (+1 or -1), one per retained non-overlapping episode.
    """
    error_col = f"{sector_prefix}_capm_error"
    if error_col not in df.columns:
        return pd.Series(dtype=float)

    episodes = get_flag_episodes(df)
    signs = []
    last_window_end = -1

    for start, end in episodes:
        trigger = end  # evaluate from the end of the episode, not the start
        if trigger <= last_window_end:
            continue  # would overlap previous trial's forward window — skip

        window_start = trigger + 1
        window_end = min(trigger + forward_days + 1, len(df))
        if window_start >= len(df):
            continue

        forward_errors = df[error_col].iloc[window_start:window_end]
        if len(forward_errors) < max(1, forward_days * 0.5):
            continue

        mean_error = forward_errors.mean()
        signs.append(np.sign(mean_error))
        last_window_end = window_end - 1

    return pd.Series(signs)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_main_test(df):
    """
    Run the core prediction test across all sectors and forward windows.
    Returns results DataFrame.
    """
    print("\n" + "=" * 60)
    print("PREDICTION TEST — MAIN RESULTS")
    print("=" * 60)

    results = []

    for name in SECTORS:
        prefix = name.lower()
        error_col = f"{prefix}_capm_error"
        if error_col not in df.columns:
            continue

        for fwd in FORWARD_WINDOWS:
            signs = test_forward_prediction(df, prefix, fwd)
            n_pos, n_total, hit_rate, p_val = sign_test(signs)

            ep_signs = test_forward_prediction_episodes(df, prefix, fwd)
            ep_n_pos, ep_n_total, ep_hit_rate, ep_p_val = sign_test(ep_signs)

            sig = "YES" if ep_p_val is not None and ep_p_val < 0.05 else "no"
            if ep_n_total < 10:
                sig = "n/a (insufficient data)"

            results.append({
                "sector": name,
                "forward_days": fwd,
                "n_flags_naive": n_total,
                "hit_rate_naive": hit_rate,
                "p_value_naive": p_val,
                "n_flags_corrected": ep_n_total,
                "hit_rate_corrected": ep_hit_rate,
                "p_value_corrected": ep_p_val,
                "significant": sig,
            })

            hr_str = f"{hit_rate:.3f}" if not np.isnan(hit_rate) else "n/a"
            pv_str = f"{p_val:.4f}" if p_val is not None and not np.isnan(p_val) else "n/a"
            ep_hr_str = f"{ep_hit_rate:.3f}" if not np.isnan(ep_hit_rate) else "n/a"
            ep_pv_str = f"{ep_p_val:.4f}" if ep_p_val is not None and not np.isnan(ep_p_val) else "n/a"
            print(f"  {name:15s} | fwd={fwd:2d}d | naive: n={n_total:3d} hit={hr_str} p={pv_str} "
                  f"| corrected (non-overlapping episodes): n={ep_n_total:3d} hit={ep_hr_str} p={ep_pv_str} | {sig}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def run_walk_forward(df):
    """
    Split into 5 non-overlapping ~2-year periods.
    Run sign test within each period.
    Check if hit_rate is consistent across periods.
    """
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)

    # Define periods. Bounded to the training data's actual coverage
    # (2015-01 to 2022-12-08) — the holdout (Aug 2025-Apr 2026) is
    # intentionally excluded here; it is evaluated separately, never mixed
    # into walk-forward periods.
    periods = [
        ("2016-01-01", "2017-12-31"),
        ("2018-01-01", "2019-12-31"),
        ("2020-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-08"),
    ]

    results = []

    for name in SECTORS:
        prefix = name.lower()
        error_col = f"{prefix}_capm_error"
        if error_col not in df.columns:
            continue

        print(f"\n  {name}:")
        for start, end in periods:
            mask = (df.index >= start) & (df.index <= end)
            period_df = df[mask]

            if period_df["anomaly_flag"].sum() < 3:
                print(f"    {start[:4]}-{end[:4]}: insufficient flags")
                continue

            # Use 3-day forward, corrected (non-overlapping episode) test
            signs = test_forward_prediction_episodes(period_df, prefix, 3)
            n_pos, n_total, hit_rate, p_val = sign_test(signs)

            hr_str = f"{hit_rate:.3f}" if not np.isnan(hit_rate) else "n/a"
            print(f"    {start[:4]}-{end[:4]}: n={n_total:3d}, hit={hr_str}")

            results.append({
                "sector": name,
                "period": f"{start[:4]}-{end[:4]}",
                "n_flags": n_total,
                "hit_rate": hit_rate,
                "p_value": p_val,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Signal decomposition
# ---------------------------------------------------------------------------

def run_signal_decomposition(df_original):
    """
    Test each signal ALONE vs concordance.
    Does flow alone predict? Does VIX alone? Does combination add value?
    """
    print("\n" + "=" * 60)
    print("SIGNAL DECOMPOSITION")
    print("=" * 60)

    signal_cols = ["flow_div_pct", "pcr_dev_pct", "vol_spread_pct"]
    available = [c for c in signal_cols if c in df_original.columns]

    results = []

    # Test each signal individually
    for sig_col in available:
        df_test = df_original.copy()

        # Flag based on single signal
        is_extreme = (df_test[sig_col] >= 0.90) | (df_test[sig_col] <= 0.10)
        df_test["anomaly_flag"] = is_extreme.astype(int)

        for name in SECTORS:
            prefix = name.lower()
            error_col = f"{prefix}_capm_error"
            if error_col not in df_test.columns:
                continue

            signs = test_forward_prediction_episodes(df_test, prefix, 3)
            n_pos, n_total, hit_rate, p_val = sign_test(signs)

            results.append({
                "signal": sig_col.replace("_pct", ""),
                "sector": name,
                "n_flags": n_total,
                "hit_rate": hit_rate,
                "p_value": p_val,
            })

            hr_str = f"{hit_rate:.3f}" if not np.isnan(hit_rate) else "n/a"
            print(f"  {sig_col:20s} | {name:15s} | n={n_total:3d} | hit={hr_str}")

    # Compare with concordance (already in df_original)
    print(f"\n  {'CONCORDANCE (2/3)':20s} — already computed in main test above")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Robustness sweep
# ---------------------------------------------------------------------------

def run_robustness_sweep(df_raw):
    """
    Sweep across percentile windows and extreme thresholds.
    Check if results are stable across a plateau of parameters.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS SWEEP")
    print("=" * 60)

    results = []

    for pct_window in WINDOW_SWEEP:
        # Recompute signals with different window
        df_test = df_raw.copy()
        df_test = compute_all_signals(df_test, window=pct_window)

        for extreme in PERCENTILE_SWEEP:
            df_test = compute_anomaly_flags(df_test, extreme=extreme, min_concordance=2)

            for fwd in [3, 5]:  # focus on 3 and 5 day forward
                for name in SECTORS:
                    prefix = name.lower()
                    error_col = f"{prefix}_capm_error"
                    if error_col not in df_test.columns:
                        continue

                    signs = test_forward_prediction_episodes(df_test, prefix, fwd)
                    n_pos, n_total, hit_rate, p_val = sign_test(signs)

                    results.append({
                        "pct_window": pct_window,
                        "extreme": extreme,
                        "forward_days": fwd,
                        "sector": name,
                        "n_flags": n_total,
                        "hit_rate": hit_rate,
                        "p_value": p_val,
                    })

    result_df = pd.DataFrame(results)

    # Summary: how many parameter combos show significance?
    if not result_df.empty and "p_value" in result_df.columns:
        sig_count = (result_df["p_value"] < 0.05).sum()
        total_count = result_df["p_value"].notna().sum()
        print(f"\n  Significant results: {sig_count}/{total_count} "
              f"({sig_count/total_count*100:.1f}% of parameter combos)")

        if sig_count / max(total_count, 1) > 0.5:
            print("  ROBUST: >50% of parameter combos show significance")
        elif sig_count / max(total_count, 1) > 0.2:
            print("  MODERATE: 20-50% of parameter combos show significance")
        else:
            print("  WEAK: <20% of parameter combos show significance")

    return result_df


# ---------------------------------------------------------------------------
# Generate result plots
# ---------------------------------------------------------------------------

def generate_plots(df, main_results, robustness_results):
    """Generate and save analysis plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot 1: Nifty price with anomaly flags
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["nifty_close"], color="black", linewidth=0.8, label="Nifty 50")

    flagged = df[df["anomaly_flag"] == 1]
    if not flagged.empty:
        ax.scatter(flagged.index, flagged["nifty_close"],
                   color="red", s=8, alpha=0.6, label="Anomaly flag", zorder=5)

    ax.set_title("Nifty 50 with Anomaly Flags")
    ax.set_ylabel("Nifty 50 Close")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "01_nifty_flags.png"), dpi=150)
    plt.close()
    print("  Saved 01_nifty_flags.png")

    # Plot 2: Signal percentiles over time
    signal_cols = ["flow_div_pct", "pcr_dev_pct", "vol_spread_pct"]
    available = [c for c in signal_cols if c in df.columns]

    if available:
        fig, axes = plt.subplots(len(available), 1, figsize=(14, 3 * len(available)),
                                 sharex=True)
        if len(available) == 1:
            axes = [axes]

        colors = ["#185FA5", "#0F6E56", "#534AB7"]
        for i, col in enumerate(available):
            axes[i].plot(df.index, df[col], color=colors[i], linewidth=0.6)
            axes[i].axhline(0.90, color="red", linestyle="--", alpha=0.5)
            axes[i].axhline(0.10, color="red", linestyle="--", alpha=0.5)
            axes[i].set_ylabel(col.replace("_pct", ""))
            axes[i].set_ylim(-0.05, 1.05)

        axes[-1].set_xlabel("Date")
        fig.suptitle("Signal Percentile Ranks (red = extreme threshold)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "02_signals.png"), dpi=150)
        plt.close()
        print("  Saved 02_signals.png")

    # Plot 3: Main results — hit rate by sector and forward window
    if not main_results.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sectors = main_results["sector"].unique()
        x = np.arange(len(FORWARD_WINDOWS))
        width = 0.8 / len(sectors)

        for i, sector in enumerate(sectors):
            mask = main_results["sector"] == sector
            hit_rates = main_results[mask]["hit_rate_corrected"].values
            bars = ax.bar(x + i * width, hit_rates, width, label=sector, alpha=0.8)

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Random (50%)")
        ax.set_xlabel("Forward Window (days)")
        ax.set_ylabel("Hit Rate (corrected, non-overlapping episodes)")
        ax.set_title("CAPM Error Direction Prediction — Hit Rate by Sector")
        ax.set_xticks(x + width * len(sectors) / 2)
        ax.set_xticklabels(FORWARD_WINDOWS)
        ax.legend(fontsize=8)
        ax.set_ylim(0.3, 0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "03_hit_rates.png"), dpi=150)
        plt.close()
        print("  Saved 03_hit_rates.png")

    # Plot 4: Robustness heatmap
    if not robustness_results.empty:
        for sector in robustness_results["sector"].unique():
            mask = (robustness_results["sector"] == sector) & \
                   (robustness_results["forward_days"] == 3)
            subset = robustness_results[mask]

            if subset.empty:
                continue

            pivot = subset.pivot_table(
                values="hit_rate", index="pct_window", columns="extreme"
            )

            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.3, vmax=0.7,
                           aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Extreme Percentile")
            ax.set_ylabel("Percentile Window")
            ax.set_title(f"Robustness: {sector} (3-day forward, hit rate)")
            plt.colorbar(im, label="Hit Rate")

            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=9, color="black")

            plt.tight_layout()
            fname = f"04_robustness_{sector.lower()}.png"
            plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
            plt.close()
            print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(df):
    """Run all tests and generate outputs."""

    # Main test
    main_results = run_main_test(df)

    # Walk-forward
    wf_results = run_walk_forward(df)

    # Signal decomposition
    decomp_results = run_signal_decomposition(df)

    # Robustness sweep (uses raw data to recompute signals with different windows)
    print("\n  Running robustness sweep (this may take a minute)...")
    robustness_results = run_robustness_sweep(df)

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(df, main_results, robustness_results)

    # Save result tables
    os.makedirs(RESULTS_DIR, exist_ok=True)

    main_results.to_csv(os.path.join(RESULTS_DIR, "main_results.csv"), index=False)
    wf_results.to_csv(os.path.join(RESULTS_DIR, "walk_forward.csv"), index=False)
    decomp_results.to_csv(os.path.join(RESULTS_DIR, "decomposition.csv"), index=False)
    robustness_results.to_csv(os.path.join(RESULTS_DIR, "robustness.csv"), index=False)

    print(f"\nAll results saved to {RESULTS_DIR}/")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not main_results.empty:
        sig = main_results[main_results["significant"] == "YES"]
        if len(sig) > 0:
            print(f"\n  SIGNIFICANT results found in {len(sig)} sector/window combos:")
            for _, row in sig.iterrows():
                print(f"    {row['sector']} | {row['forward_days']}d | "
                      f"hit={row['hit_rate_corrected']:.3f} | p={row['p_value_corrected']:.4f}")
        else:
            print("\n  No significant results. The signal does not predict "
                  "CAPM error direction.")
            print("  This is a valid negative finding.")

    return main_results, wf_results, decomp_results, robustness_results


if __name__ == "__main__":
    csv_path = os.path.join(CLEAN_DIR, "flagged.csv")
    if not os.path.exists(csv_path):
        print(f"Run detector.py first. Missing: {csv_path}")
        raise SystemExit(1)

    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    run_all(df)
