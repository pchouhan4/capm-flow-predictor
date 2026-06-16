"""
Unit tests for each component.

Run: python3 -m pytest test_components.py -v
"""

import numpy as np
import pandas as pd
import pytest

from signals import compute_flow_divergence, compute_pcr_deviation, compute_vol_spread
from detector import compute_anomaly_flags
from test_prediction import sign_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Create synthetic daily data for testing."""
    np.random.seed(42)
    n = 500  # ~2 years of trading days
    dates = pd.bdate_range("2020-01-01", periods=n)

    df = pd.DataFrame(index=dates)
    df.index.name = "date"

    # Nifty close: random walk starting at 12000
    df["nifty_close"] = 12000 + np.cumsum(np.random.randn(n) * 100)

    # India VIX: mean-reverting around 15
    vix = [15.0]
    for i in range(1, n):
        vix.append(vix[-1] + 0.1 * (15 - vix[-1]) + np.random.randn() * 1.5)
    df["india_vix"] = np.clip(vix, 8, 80)

    # FII/DII flows
    df["fii_net"] = np.random.randn(n) * 2000
    df["dii_net"] = np.random.randn(n) * 1500

    # PCR
    df["pcr"] = 0.8 + np.random.randn(n) * 0.15
    df["pcr"] = df["pcr"].clip(0.3, 2.5)

    return df


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------

class TestSignals:

    def test_flow_divergence_range(self, synthetic_data):
        df = compute_flow_divergence(synthetic_data.copy(), window=100)
        valid = df["flow_div_pct"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 1

    def test_flow_divergence_warmup(self, synthetic_data):
        window = 100
        df = compute_flow_divergence(synthetic_data.copy(), window=window)
        # First (window - 1) rows should be NaN
        assert df["flow_div_pct"].iloc[:window - 1].isna().all()

    def test_pcr_deviation_range(self, synthetic_data):
        df = compute_pcr_deviation(synthetic_data.copy(), baseline=30, window=100)
        valid = df["pcr_dev_pct"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 1

    def test_pcr_zero_std_handling(self, synthetic_data):
        """PCR constant for 30 days should not produce NaN."""
        df = synthetic_data.copy()
        df["pcr"] = 1.0  # constant
        df = compute_pcr_deviation(df, baseline=30, window=100)
        # pcr_dev_raw should be 0 (not NaN) when std is 0
        valid_raw = df["pcr_dev_raw"].dropna()
        assert (valid_raw == 0).all()

    def test_vol_spread_range(self, synthetic_data):
        df = compute_vol_spread(synthetic_data.copy(), rv_window=20, window=100)
        valid = df["vol_spread_pct"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 1

    def test_missing_columns_skip(self, synthetic_data):
        """Signals should skip gracefully if source columns missing."""
        df = synthetic_data.drop(columns=["fii_net", "dii_net"]).copy()
        df = compute_flow_divergence(df, window=100)
        assert "flow_div_pct" not in df.columns


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------

class TestDetector:

    def test_extreme_data_flags(self):
        """Synthetic extreme data should trigger flags."""
        n = 300
        dates = pd.bdate_range("2020-01-01", periods=n)
        df = pd.DataFrame(index=dates)

        # Normal for first 250 days, then all signals go extreme
        normal = np.random.uniform(0.3, 0.7, 250)
        extreme = np.random.uniform(0.92, 0.99, 50)

        for col in ["flow_div_pct", "pcr_dev_pct", "vol_spread_pct"]:
            df[col] = np.concatenate([normal, extreme])

        df = compute_anomaly_flags(df, extreme=0.90, min_concordance=2)
        # Last 50 days should all be flagged
        assert df["anomaly_flag"].iloc[-50:].sum() == 50

    def test_normal_data_no_flags(self):
        """Data in normal range should not trigger flags."""
        n = 300
        dates = pd.bdate_range("2020-01-01", periods=n)
        df = pd.DataFrame(index=dates)

        for col in ["flow_div_pct", "pcr_dev_pct", "vol_spread_pct"]:
            df[col] = np.random.uniform(0.2, 0.8, n)

        df = compute_anomaly_flags(df, extreme=0.90, min_concordance=2)
        assert df["anomaly_flag"].sum() == 0


# ---------------------------------------------------------------------------
# Sign test
# ---------------------------------------------------------------------------

class TestSignTest:

    def test_random_signs_not_significant(self):
        """Random 50/50 signs should not be significant."""
        np.random.seed(42)
        signs = pd.Series(np.random.choice([-1, 1], size=100))
        _, _, hit_rate, p_val = sign_test(signs)
        assert abs(hit_rate - 0.5) < 0.15  # should be near 0.5
        assert p_val > 0.05  # should not be significant

    def test_biased_signs_significant(self):
        """Strongly biased signs should be significant."""
        signs = pd.Series([1] * 40 + [-1] * 10)
        _, _, hit_rate, p_val = sign_test(signs)
        assert hit_rate > 0.7
        assert p_val < 0.01

    def test_insufficient_data(self):
        """Fewer than 5 observations should return NaN."""
        signs = pd.Series([1, -1, 1])
        _, n, hit_rate, p_val = sign_test(signs)
        assert np.isnan(hit_rate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
