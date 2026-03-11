import pytest
import pandas as pd
import numpy as np

# Adjust import based on where the module actually lives.
from features.labeling_schemes import calculate_future_returns

def test_calculate_future_returns_default_horizon():
    """Test default horizon=1 calculation"""
    # Create sample dataframe
    df = pd.DataFrame({
        "close": [100.0, 105.0, 102.9, 110.0]
    })

    # Calculate returns
    result_df = calculate_future_returns(df.copy())

    # Check shape - should drop last row due to horizon=1
    assert len(result_df) == 3

    # Check if 'future_returns' column exists
    assert "future_returns" in result_df.columns

    # Expected returns
    # row 0: (105.0 - 100.0) / 100.0 = 0.05
    # row 1: (102.9 - 105.0) / 105.0 = -0.02
    # row 2: (110.0 - 102.9) / 102.9 = 0.0690...
    expected_returns = [0.05, -0.02, (110.0 - 102.9) / 102.9]

    # Assert values are close
    np.testing.assert_allclose(result_df["future_returns"].values, expected_returns)

def test_calculate_future_returns_custom_horizon():
    """Test with custom horizon=2"""
    df = pd.DataFrame({
        "close": [100.0, 105.0, 110.0, 107.8]
    })

    result_df = calculate_future_returns(df.copy(), horizon=2)

    # Should drop last 2 rows
    assert len(result_df) == 2

    # Expected returns for horizon 2
    # row 0: (110.0 - 100.0) / 100.0 = 0.10
    # row 1: (107.8 - 105.0) / 105.0 = 0.0266...
    expected_returns = [0.10, (107.8 - 105.0) / 105.0]

    np.testing.assert_allclose(result_df["future_returns"].values, expected_returns)

def test_calculate_future_returns_missing_close_column():
    """Test KeyError is raised when 'close' column is missing"""
    df = pd.DataFrame({
        "price": [100.0, 105.0]
    })

    with pytest.raises(KeyError):
        calculate_future_returns(df)

def test_calculate_future_returns_empty_dataframe():
    """Test behavior with an empty dataframe"""
    df = pd.DataFrame(columns=["close"])

    result_df = calculate_future_returns(df.copy())

    assert len(result_df) == 0
    assert "future_returns" in result_df.columns

def test_calculate_future_returns_all_nans():
    """Test behavior with a dataframe containing only NaNs"""
    df = pd.DataFrame({
        "close": [np.nan, np.nan, np.nan]
    })

    result_df = calculate_future_returns(df.copy())

    # Dropna should drop all rows
    assert len(result_df) == 0
    assert "future_returns" in result_df.columns

def test_calculate_future_returns_horizon_larger_than_data():
    """Test behavior when horizon is larger than dataframe size"""
    df = pd.DataFrame({
        "close": [100.0, 105.0]
    })

    result_df = calculate_future_returns(df.copy(), horizon=5)

    # Should drop all rows
    assert len(result_df) == 0
