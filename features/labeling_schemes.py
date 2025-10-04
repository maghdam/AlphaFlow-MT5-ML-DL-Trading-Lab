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
    
    labels = np.full(len(closes), np.nan)
    
    for i in range(len(closes)):
        current_price = closes[i]
        upper_barrier = current_price * (1 + up)
        lower_barrier = current_price * (1 - down)
        
        # Look ahead up to horizon bars (or until dataset ends)
        end = min(i + horizon, len(closes))
        for fwd_i in range(i+1, end):
            if closes[fwd_i] >= upper_barrier:
                labels[i] = 1
                break
            elif closes[fwd_i] <= lower_barrier:
                labels[i] = -1
                break
        # if we exit loop without setting label => neither barrier hit => 0
        if np.isnan(labels[i]):
            labels[i] = 0
    
    df_copy["barrier_label"] = labels
    return df_copy



def create_labels_double_barrier(df, up=0.005, down=0.005, horizon=20):
    """
    Double-barrier labeling:
      +1 if upper barrier is touched first,
      -1 if lower barrier is touched first,
       0 if neither is touched within horizon.
    df must have a 'close' column.
    Returns a new DataFrame with a 'barrier_label' column in {-1, 0, +1}.
    """
    df_copy = df.copy()
    closes = df_copy["close"].values
    labels = np.full(len(closes), np.nan)
    
    for i in range(len(closes)):
        current_price = closes[i]
        upper_barrier = current_price * (1 + up)
        lower_barrier = current_price * (1 - down)
        
        end = min(i + horizon, len(closes))
        for fwd_i in range(i+1, end):
            if closes[fwd_i] >= upper_barrier:
                labels[i] = 1
                break
            elif closes[fwd_i] <= lower_barrier:
                labels[i] = -1
                break
        if np.isnan(labels[i]):
            labels[i] = 0
    
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
