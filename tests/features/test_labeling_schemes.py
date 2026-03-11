import pytest
import pandas as pd
import numpy as np
from features.labeling_schemes import calculate_future_returns

def test_calculate_future_returns_happy_path():
    # Setup data
    data = {"close": [100.0, 105.0, 102.0, 110.0, 115.0]}
    df = pd.DataFrame(data)

    # Test horizon 1
    result_h1 = calculate_future_returns(df.copy(), horizon=1)

    # Expected:
    # idx 0: (105 - 100) / 100 = 0.05
    # idx 1: (102 - 105) / 105 = -0.028571
    # idx 2: (110 - 102) / 102 = 0.078431
    # idx 3: (115 - 110) / 110 = 0.045455
    # idx 4: NaN
    assert len(result_h1) == 4
    np.testing.assert_allclose(result_h1["future_returns"].iloc[0], 0.05, atol=1e-5)
    np.testing.assert_allclose(result_h1["future_returns"].iloc[1], -0.028571, atol=1e-5)

    # Test horizon 2
    result_h2 = calculate_future_returns(df.copy(), horizon=2)
    # idx 0: (102 - 100) / 100 = 0.02
    # idx 1: (110 - 105) / 105 = 0.047619
    # idx 2: (115 - 102) / 102 = 0.127451
    # idx 3: NaN
    # idx 4: NaN
    assert len(result_h2) == 3
    np.testing.assert_allclose(result_h2["future_returns"].iloc[0], 0.02, atol=1e-5)
    np.testing.assert_allclose(result_h2["future_returns"].iloc[1], 0.047619, atol=1e-5)
    np.testing.assert_allclose(result_h2["future_returns"].iloc[2], 0.127451, atol=1e-5)

def test_calculate_future_returns_tiny_df():
    # Rationale: Testing with a tiny DataFrame length < horizon to see if it correctly returns empty or handles it gracefully.
    data = {"close": [100.0, 105.0]}
    df = pd.DataFrame(data)

    # Test horizon 5 where df length is 2
    result = calculate_future_returns(df.copy(), horizon=5)

    # Should return empty DataFrame gracefully
    assert result.empty
    assert "future_returns" in result.columns

def test_calculate_future_returns_missing_close_column():
    data = {"open": [100.0, 105.0, 102.0]}
    df = pd.DataFrame(data)

    with pytest.raises(KeyError):
        calculate_future_returns(df.copy(), horizon=1)
