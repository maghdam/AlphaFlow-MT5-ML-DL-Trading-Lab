import pandas as pd
import numpy as np
from features.labeling_schemes import create_labels_multi_bar

def test_create_labels_multi_bar():
    # Toy dataframe with close prices
    df = pd.DataFrame({
        "close": [100.0, 102.0, 99.0, 99.0, 105.0]
    })

    # horizon = 1, threshold = 0.01
    # row 0: close = 100, future = 102, return = 0.02 >= 0.01 -> label = 1
    # row 1: close = 102, future = 99, return = -3/102 = -0.0294 <= -0.01 -> label = -1
    # row 2: close = 99, future = 99, return = 0.00 -> label = 0
    # row 3: close = 99, future = 105, return = 6/99 = 0.0606 >= 0.01 -> label = 1
    # row 4: close = 105, future = NaN

    res = create_labels_multi_bar(df, horizon=1, threshold=0.01)

    # Should have 4 rows because the last row is dropped due to NaN future return
    assert len(res) == 4

    expected_labels = [1, -1, 0, 1]
    np.testing.assert_array_equal(res["multi_bar_label"].values, expected_labels)

    # Check returns
    expected_returns = [0.02, -3/102, 0.0, 6/99]
    np.testing.assert_array_almost_equal(res["future_return_h"].values, expected_returns)

def test_create_labels_multi_bar_custom_horizon():
    # Test with horizon=2, threshold=0.05
    df = pd.DataFrame({
        "close": [100.0, 101.0, 105.0, 90.0, 95.0, 100.0]
    })

    # horizon = 2
    # row 0: close 100, future 105 (idx 2), return 0.05 >= 0.05 -> 1
    # row 1: close 101, future 90 (idx 3), return -11/101 = -0.1089 <= -0.05 -> -1
    # row 2: close 105, future 95 (idx 4), return -10/105 = -0.0952 <= -0.05 -> -1
    # row 3: close 90,  future 100 (idx 5), return 10/90 = 0.1111 >= 0.05 -> 1
    # row 4: NaN
    # row 5: NaN

    res = create_labels_multi_bar(df, horizon=2, threshold=0.05)

    assert len(res) == 4
    expected_labels = [1, -1, -1, 1]
    np.testing.assert_array_equal(res["multi_bar_label"].values, expected_labels)

def test_create_labels_multi_bar_exact_threshold():
    # Check boundary condition where return is exactly the threshold
    df = pd.DataFrame({
        "close": [100.0, 105.0, 95.0]
    })

    # threshold = 0.05, horizon = 1
    # row 0: return 0.05 -> 1
    # row 1: return -10/105 = -0.0952 -> -1
    res = create_labels_multi_bar(df, horizon=1, threshold=0.05)
    assert res.iloc[0]["multi_bar_label"] == 1

    res = create_labels_multi_bar(df, horizon=1, threshold=0.1)
    # return is 0.05, which is < 0.1 and > -0.1
    assert res.iloc[0]["multi_bar_label"] == 0
