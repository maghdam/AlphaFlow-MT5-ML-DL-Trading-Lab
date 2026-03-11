import pandas as pd
import numpy as np
import pytest
from features.labeling_schemes import create_labels_multi_bar

def test_create_labels_multi_bar():
    """
    Test create_labels_multi_bar correctly assigns labels based on future returns.
    """
    # Create a simple dummy dataframe
    # We want future returns over horizon=2 to be:
    # index 0: (10.5 / 10.0) - 1 = 0.05 (should be +1, since >= 0.05 is not met if threshold=0.06, wait let's use exact)

    df = pd.DataFrame({
        "close": [100.0, 100.0, 105.0, 95.0, 100.0, 100.0]
    })

    # Let's set horizon=2, threshold=0.04
    # future returns for horizon=2:
    # i=0: (105.0 - 100.0)/100.0 = 0.05 => >= 0.04 => 1
    # i=1: (95.0 - 100.0)/100.0 = -0.05 => <= -0.04 => -1
    # i=2: (100.0 - 105.0)/105.0 = -0.0476 => <= -0.04 => -1
    # i=3: (100.0 - 95.0)/95.0 = 0.0526 => >= 0.04 => 1
    # i=4: NaN
    # i=5: NaN

    labeled_df = create_labels_multi_bar(df, horizon=2, threshold=0.04)

    # Check that df wasn't modified in place
    assert "multi_bar_label" not in df.columns

    # Ensure correct columns exist in result
    assert "future_return_h" in labeled_df.columns
    assert "multi_bar_label" in labeled_df.columns

    # Since the original drops NaN, it should have 4 rows
    assert len(labeled_df) == 4

    # Check calculated future returns roughly match expected
    expected_returns = [0.05, -0.05, -0.047619047619047616, 0.052631578947368474]
    np.testing.assert_allclose(labeled_df["future_return_h"].values, expected_returns, rtol=1e-5)

    # Check assigned labels
    expected_labels = [1, -1, -1, 1]
    np.testing.assert_array_equal(labeled_df["multi_bar_label"].values, expected_labels)


def test_create_labels_multi_bar_neutral():
    """
    Test create_labels_multi_bar handles neutral labels correctly (returns inside threshold).
    """
    df = pd.DataFrame({
        "close": [100.0, 101.0, 102.0, 99.0, 100.0]
    })

    # Let's set horizon=1, threshold=0.02
    # future returns for horizon=1:
    # i=0: (101 - 100)/100 = 0.01 (neutral -> 0)
    # i=1: (102 - 101)/101 = 0.0099 (neutral -> 0)
    # i=2: (99 - 102)/102 = -0.0294 (down -> -1)
    # i=3: (100 - 99)/99 = 0.0101 (neutral -> 0)

    labeled_df = create_labels_multi_bar(df, horizon=1, threshold=0.02)

    assert len(labeled_df) == 4
    expected_labels = [0, 0, -1, 0]
    np.testing.assert_array_equal(labeled_df["multi_bar_label"].values, expected_labels)
