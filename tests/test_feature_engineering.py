import pytest
import pandas as pd
import numpy as np
from features.feature_engineering import (
    spread,
    candle_information,
    log_transform,
    mathematical_derivatives,
    parkinson_estimator,
    yang_zhang_estimator,
    market_regime_dc,
    gap_detection,
    displacement_detection,
    set_double_barrier_label
)

@pytest.fixture
def sample_df():
    """Create a sample dataframe with OHLCV data for testing."""
    dates = pd.date_range("2023-01-01", periods=10)
    data = {
        "open": [100.0, 102.0, 101.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 110.0],
        "high": [105.0, 106.0, 104.0, 108.0, 107.0, 109.0, 110.0, 111.0, 112.0, 115.0],
        "low": [95.0, 100.0, 99.0, 102.0, 101.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "close": [102.0, 101.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 110.0, 112.0],
        "tick_volume": [1000, 1200, 1100, 1500, 1300, 1400, 1600, 1700, 1800, 2000]
    }
    return pd.DataFrame(data, index=dates)

def test_spread(sample_df):
    """Test spread calculation."""
    df_result = spread(sample_df)

    # Check if 'spread' column exists
    assert "spread" in df_result.columns

    # Check calculation
    expected_spread = sample_df["high"] - sample_df["low"]
    pd.testing.assert_series_equal(df_result["spread"], expected_spread, check_names=False)

    # Check that original df is not modified
    assert "spread" not in sample_df.columns

def test_candle_information(sample_df):
    """Test candle_information features."""
    df_result = candle_information(sample_df)

    # Check if columns are added
    expected_cols = ["candle_way", "fill", "amplitude"]
    for col in expected_cols:
        assert col in df_result.columns

    # Check logic of candle_way
    expected_candle_way = (sample_df["close"] > sample_df["open"]).astype(int)
    pd.testing.assert_series_equal(df_result["candle_way"], expected_candle_way, check_names=False)

    # Check fill calculation
    rng = (sample_df["high"] - sample_df["low"]).replace(0, np.nan)
    expected_fill = (sample_df["close"] - sample_df["open"]).abs() / (rng + 1e-5)
    pd.testing.assert_series_equal(df_result["fill"], expected_fill, check_names=False)

def test_log_transform(sample_df):
    """Test log_transform logic."""
    col_to_transform = "close"
    n_period = 2
    df_result = log_transform(sample_df, col_to_transform, n_period)

    # Check columns
    assert f"log_{col_to_transform}" in df_result.columns
    assert f"log_ret_{n_period}" in df_result.columns

    # Check log calculation
    expected_log = np.log(sample_df[col_to_transform].clip(lower=1e-12))
    pd.testing.assert_series_equal(df_result[f"log_{col_to_transform}"], expected_log, check_names=False)

    # Check diff calculation
    expected_diff = expected_log.diff(n_period)
    pd.testing.assert_series_equal(df_result[f"log_ret_{n_period}"], expected_diff, check_names=False)

def test_mathematical_derivatives(sample_df):
    """Test velocity and acceleration logic."""
    col_to_derive = "close"
    df_result = mathematical_derivatives(sample_df, col_to_derive)

    # Check columns
    assert "velocity" in df_result.columns
    assert "acceleration" in df_result.columns

    # Check calculation
    expected_velocity = sample_df[col_to_derive].diff()
    expected_acceleration = expected_velocity.diff()

    pd.testing.assert_series_equal(df_result["velocity"], expected_velocity, check_names=False)
    pd.testing.assert_series_equal(df_result["acceleration"], expected_acceleration, check_names=False)

def test_parkinson_estimator(sample_df):
    """Test Parkinson estimator."""
    # Test with valid window
    vol = parkinson_estimator(sample_df)
    assert not np.isnan(vol)
    assert vol > 0

    # Test with empty window
    empty_df = pd.DataFrame()
    empty_vol = parkinson_estimator(empty_df)
    assert np.isnan(empty_vol)

def test_yang_zhang_estimator(sample_df):
    """Test Yang Zhang estimator."""
    # Test with valid window
    vol = yang_zhang_estimator(sample_df)
    assert not np.isnan(vol)
    assert vol > 0

    # Test with empty window
    empty_df = pd.DataFrame()
    empty_vol = yang_zhang_estimator(empty_df)
    assert np.isnan(empty_vol)

def test_market_regime_dc(sample_df):
    """Test market regime calculation."""
    df_result = market_regime_dc(sample_df, threshold=0.01)

    # Check column
    assert "market_regime" in df_result.columns

    # Check values
    assert df_result["market_regime"].isin([0, 1, np.nan]).all()

def test_gap_detection(sample_df):
    """Test gap detection calculation."""
    df_result = gap_detection(sample_df)

    # Check expected columns
    expected_cols = [
        "Bullish_gap_inf",
        "Bullish_gap_sup",
        "Bullish_gap_size",
        "Bearish_gap_inf",
        "Bearish_gap_sup",
        "Bearish_gap_size",
    ]
    for col in expected_cols:
        assert col in df_result.columns

def test_displacement_detection(sample_df):
    """Test displacement detection."""
    df_result = displacement_detection(sample_df, period=2)

    # Check added columns
    assert "candle_range" in df_result.columns
    assert "Variation" in df_result.columns
    assert "STD" in df_result.columns
    assert "displacement" in df_result.columns
    assert "red_displacement" in df_result.columns

    # Check values of displacement
    assert df_result["displacement"].isin([0, 1]).all()
    assert df_result["red_displacement"].isin([0, 1]).all()

def test_set_double_barrier_label(sample_df):
    """Test set double barrier label."""
    # Since our sample_df only has 10 rows and we drop NAs,
    # we need a small horizon to not drop all rows.
    df_result = set_double_barrier_label(sample_df, up=0.01, down=0.01, horizon=2)

    # Check columns
    assert "barrier_label" in df_result.columns

    # Check values
    assert df_result["barrier_label"].isin([0, 1]).all()
