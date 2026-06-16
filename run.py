"""
Main runner — executes the full pipeline end-to-end.

Usage: python3 run.py
"""

import os
import sys

# Ensure we're in the project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import run_pipeline
from signals import compute_all_signals
from capm import compute_capm_errors
from detector import compute_anomaly_flags
from test_prediction import run_all
from config import CLEAN_DIR


def main():
    print("=" * 60)
    print("CAPM FLOW PREDICTOR")
    print("Does FII/DII flow divergence predict CAPM error direction?")
    print("=" * 60)

    # Step 1: Data pipeline
    print("\n\n>>> STEP 1: DATA PIPELINE\n")
    df = run_pipeline()

    # Step 2: Signals
    print("\n\n>>> STEP 2: SIGNAL CONSTRUCTION\n")
    df = compute_all_signals(df)
    df.to_csv(os.path.join(CLEAN_DIR, "signals.csv"))

    # Step 3: CAPM errors
    print("\n\n>>> STEP 3: CAPM PRICING ERRORS\n")
    df = compute_capm_errors(df)
    df.to_csv(os.path.join(CLEAN_DIR, "capm_errors.csv"))

    # Step 4: Anomaly detection
    print("\n\n>>> STEP 4: ANOMALY DETECTION\n")
    df = compute_anomaly_flags(df)
    df.to_csv(os.path.join(CLEAN_DIR, "flagged.csv"))

    # Step 5: THE TEST
    print("\n\n>>> STEP 5: PREDICTION TEST\n")
    results = run_all(df)

    print("\n\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nResults in: {os.path.join(os.getcwd(), 'results')}/")
    print(f"Data in: {os.path.join(os.getcwd(), 'data', 'clean')}/")

    return results


if __name__ == "__main__":
    main()
