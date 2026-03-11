import pandas as pd
import numpy as np   # <-- Make sure this is present


def calculate_future_returns(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Calculates future returns for a given horizon. By default, horizon=1
    means next-bar returns. The function appends a new column 'future_returns'.
    """
    df["future_returns"] = df["close"].pct_change(periods=horizon).shift(-horizon)
    return df.dropna(subset=["future_returns"])

def create_labels_multi_bar(df, horizon=5, threshold=0.005):
    """
    Creates classification labels for a multi-bar horizon.
    +1 if future return >= +threshold
    -1 if future return <= -threshold
     0 otherwise (could keep as neutral or drop).
    
    df must have a 'close' column.
    Returns a new DataFrame with:
      - 'future_return_h' (the h-bar future return)
      - 'multi_bar_label' (the classification label)
    """
    df_copy = df.copy()
    
    # 1) Compute the horizon-based future returns
    df_copy["future_return_h"] = df_copy["close"].pct_change(periods=horizon).shift(-horizon)
    
    # 2) Create classification labels
    df_copy["multi_bar_label"] = 0
    df_copy.loc[df_copy["future_return_h"] >= threshold, "multi_bar_label"] = 1
    df_copy.loc[df_copy["future_return_h"] <= -threshold, "multi_bar_label"] = -1
    
    # 3) Drop rows where future_return_h is NaN (the last 'horizon' bars)
    df_copy.dropna(subset=["future_return_h"], inplace=True)
    
    # If you prefer a pure up/down classification, do:
    # df_copy = df_copy[df_copy["multi_bar_label"] != 0]
    
    return df_copy


def create_labels_double_barrier(df, up=0.005, down=0.005, horizon=20):
    """
    Double-barrier labeling:
      - For each index i, define:
          upper_barrier = close_i * (1 + up)
          lower_barrier = close_i * (1 - down)
      - Look ahead up to 'horizon' bars to see which barrier is touched first.
      - Label = +1 if upper barrier touched first,
                -1 if lower barrier touched first,
                 0 if neither is touched within horizon.
    df must have a 'close' column.
    Returns a new DataFrame with a 'barrier_label' in {-1, 0, +1}.
    """
    df_copy = df.copy()
    closes = df_copy["close"].values
    n = len(closes)
    
    upper_barriers = closes * (1 + up)
    lower_barriers = closes * (1 - down)
    
    labels = np.zeros(n)
    unlabeled = np.ones(n, dtype=bool)
    
    for h in range(1, min(horizon, n)):
        idx = slice(0, n - h)
        future_closes = closes[h:]
        
        # Check upper barrier
        hit_upper = (future_closes >= upper_barriers[idx]) & unlabeled[idx]
        if hit_upper.any():
            hit_upper_full = np.zeros(n, dtype=bool)
            hit_upper_full[idx] = hit_upper
            labels[hit_upper_full] = 1
            unlabeled[hit_upper_full] = False

        # Check lower barrier
        hit_lower = (future_closes <= lower_barriers[idx]) & unlabeled[idx]
        if hit_lower.any():
            hit_lower_full = np.zeros(n, dtype=bool)
            hit_lower_full[idx] = hit_lower
            labels[hit_lower_full] = -1
            unlabeled[hit_lower_full] = False

        if not unlabeled.any():
            break

    df_copy["barrier_label"] = labels
    return df_copy



def create_labels_regime_detection(df, short_window=20, long_window=50):
    """
    Simple regime detection:
      +1 if short MA > long MA (up)
      -1 if short MA < long MA (down)
       0 otherwise (sideways)
    df must have 'close' column.
    Returns a new DataFrame with 'regime_label' in {-1, 0, +1}.
    """
    df_copy = df.copy()
    
    # 1) Compute short and long MAs
    df_copy["ma_short"] = df_copy["close"].rolling(short_window).mean()
    df_copy["ma_long"] = df_copy["close"].rolling(long_window).mean()
    
    # 2) Label each bar
    df_copy["regime_label"] = 0
    up_mask = df_copy["ma_short"] > df_copy["ma_long"]
    down_mask = df_copy["ma_short"] < df_copy["ma_long"]
    
    df_copy.loc[up_mask, "regime_label"] = 1
    df_copy.loc[down_mask, "regime_label"] = -1
    
    # 3) Drop rows where MAs are NaN (the first 'long_window' bars)
    df_copy.dropna(subset=["ma_short", "ma_long"], inplace=True)
    
    return df_copy


def create_labels_volatility(df: pd.DataFrame, returns_window: int = 1, vol_window: int = 20) -> pd.DataFrame:
    """
    Creates labels based on volatility and future returns.
    
    The function calculates the future returns and the rolling volatility, then
    assigns labels based on the following conditions:
    -  1: if future return > volatility
    - -1: if future return < -volatility
    -  0: otherwise

    Args:
        df (pd.DataFrame): DataFrame containing the 'close' column.
        returns_window (int): Horizon for calculating future returns.
        vol_window (int): Rolling window for calculating volatility.

    Returns:
        pd.DataFrame: A new DataFrame with a 'volatility_label' column in {-1, 0, +1}.
    """
    df_copy = df.copy()
    df_copy = calculate_future_returns(df_copy, horizon=returns_window)
    df_copy["volatility"] = df_copy["future_returns"].rolling(vol_window, min_periods=1).std()

    df_copy["volatility_label"] = 0
    df_copy.loc[df_copy["future_returns"] > df_copy["volatility"], "volatility_label"] = 1
    df_copy.loc[df_copy["future_returns"] < -df_copy["volatility"], "volatility_label"] = -1

    df_copy.dropna(subset=["volatility", "future_returns"], inplace=True)
    return df_copy
