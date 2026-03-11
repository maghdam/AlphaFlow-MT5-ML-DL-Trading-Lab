import pytest
import pandas as pd
import numpy as np
from features.labeling_schemes import (
    calculate_future_returns,
    create_labels_multi_bar,
    create_labels_double_barrier,
    create_labels_regime_detection,
    create_labels_volatility,
)

@pytest.fixture
def sample_data():
    """Provides a sample DataFrame for testing."""
    # Prices: 100 -> 101 -> 102 -> 100 -> 98 -> 95 -> 100
    dates = pd.date_range("2023-01-01", periods=7, freq="D")
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 100.0, 98.0, 95.0, 100.0]}, index=dates)
    return df

@pytest.fixture
def sample_data_long():
    """Provides a longer sample DataFrame for testing regime detection."""
    # 55 periods
    dates = pd.date_range("2023-01-01", periods=55, freq="D")
    # First 30 periods: trend up, then trend down
    closes = np.linspace(100, 130, 30).tolist() + np.linspace(130, 100, 25).tolist()
    df = pd.DataFrame({"close": closes}, index=dates)
    return df

def test_calculate_future_returns(sample_data):
    df_out = calculate_future_returns(sample_data, horizon=1)

    # Check shape, original is 7, shifted by 1 means 1 nan dropped => 6 rows
    assert len(df_out) == 6
    assert "future_returns" in df_out.columns

    # 100 to 101 => 0.01
    assert pytest.approx(df_out["future_returns"].iloc[0]) == 0.01
    # 101 to 102 => 1/101
    assert pytest.approx(df_out["future_returns"].iloc[1]) == 1.0 / 101.0

def test_create_labels_multi_bar(sample_data):
    df_out = create_labels_multi_bar(sample_data, horizon=2, threshold=0.01)

    # original is 7, horizon=2 means 2 nan dropped => 5 rows
    assert len(df_out) == 5
    assert "future_return_h" in df_out.columns
    assert "multi_bar_label" in df_out.columns

    # idx 0: 100 -> 102 (horizon 2), return = 0.02 >= 0.01 => label 1
    assert df_out["multi_bar_label"].iloc[0] == 1
    # idx 2: 102 -> 98 (horizon 2), return = -4/102 = -0.039 <= -0.01 => label -1
    assert df_out["multi_bar_label"].iloc[2] == -1
    # Let's check idx 1: 101 -> 100, return = -1/101 = -0.0099. threshold is 0.01
    # -0.0099 is NOT <= -0.01, so label should be 0
    assert df_out["multi_bar_label"].iloc[1] == 0

def test_create_labels_double_barrier(sample_data):
    # Prices: 100 -> 101 -> 102 -> 100 -> 98 -> 95 -> 100
    # up=0.015, down=0.015, horizon=3
    # For idx 0 (100): upper=101.5, lower=98.5
    #   horizon=3 => looks at 101, 102, 100
    #   hits 102 > 101.5 first. => label 1

    # For idx 2 (102): upper=103.53, lower=100.47
    #   horizon=3 => looks at 100, 98, 95
    #   hits 100 < 100.47 first. => label -1

    # For idx 3 (100): upper=101.5, lower=98.5
    #   horizon=3 => looks at 98, 95, 100
    #   hits 98 < 98.5 first. => label -1

    # Let's pick a case where neither is hit:
    # We need a small horizon or large barriers.

    df_out = create_labels_double_barrier(sample_data, up=0.015, down=0.015, horizon=3)

    assert "barrier_label" in df_out.columns
    assert len(df_out) == 7

    assert df_out["barrier_label"].iloc[0] == 1
    assert df_out["barrier_label"].iloc[2] == -1
    assert df_out["barrier_label"].iloc[3] == -1

    # test no hit
    df_out_nohit = create_labels_double_barrier(sample_data, up=0.1, down=0.1, horizon=1)
    # 100 -> 101 (up=110, down=90). Not hit. => label 0
    assert df_out_nohit["barrier_label"].iloc[0] == 0

def test_create_labels_regime_detection(sample_data_long):
    df_out = create_labels_regime_detection(sample_data_long, short_window=5, long_window=10)

    # dropna for long_window => 55 - 10 + 1 = 46 rows
    assert len(df_out) == 46
    assert "regime_label" in df_out.columns
    assert "ma_short" in df_out.columns
    assert "ma_long" in df_out.columns

    # Since first 30 are trending up, short MA > long MA
    assert df_out["regime_label"].iloc[0] == 1

    # The end is trending down, short MA < long MA
    assert df_out["regime_label"].iloc[-1] == -1

def test_create_labels_volatility(sample_data):
    # Prices: 100 -> 101 -> 102 -> 100 -> 98 -> 95 -> 100
    # First, calculate returns for horizon=1:
    # 0.01, 0.0099, -0.0196, -0.02, -0.0306, 0.0526
    # Volatility window 2:
    # idx 0: ret=0.01, vol=nan (if min_periods=2) but min_periods=1 => std of [0.01] = nan
    # Actually pandas std() with 1 element is NaN.
    # We should use min_periods=2 or just check the output.

    df_out = create_labels_volatility(sample_data, returns_window=1, vol_window=3)

    assert "volatility_label" in df_out.columns
    # Will drop initial NaNs. Std requires at least 2 points.
    # Future returns has 6 rows. The first row's std is NaN. So 5 rows remaining.
    assert len(df_out) == 5

    # We just ensure it runs and outputs valid values
    assert df_out["volatility_label"].isin([-1, 0, 1]).all()
    assert df_out["volatility"].isna().sum() == 0
