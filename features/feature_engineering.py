# feature_engineering.py

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

from scipy.fftpack import fft  # simple global FFT (optional)
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler  # keep for downstream pipelines


# =============================================================================
# Stationarity configuration (can be overridden at call time)
# =============================================================================
STATIONARITY_CFG: Dict = {
    "enabled": True,
    "adf_alpha": 0.05,           # want ADF p < alpha
    "kpss_alpha": 0.05,          # want KPSS p > alpha
    "keep_original": False,      # keep both original and stationary variant
    "max_diff": 2,               # maximum extra differencing attempts
    "seasonal_period": None,     # e.g., 6 for 4H bars ~ daily; None to skip
    "transform_order": ["pct_change", "diff1", "log_diff1", "seasonal_diff"],
    # columns we never transform (raw OHLCV by default)
    "exclude_cols": {"open", "high", "low", "close", "tick_volume", "volume",
                     "rolling_adf_stat", "rolling_adf_pval", "stationary_flag"},
}


# =============================================================================
# 1) TA-LIB FEATURES (ta library)
# =============================================================================
def add_all_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a wide range of technical analysis indicators using the 'ta' library.
    Modifies the DataFrame in place and returns it for chaining.
    """
    df = ta.add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="tick_volume",
        fillna=True,
    )
    return df


def create_custom_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Example custom feature: a rolling mean of the close price."""
    df["rolling_mean_10"] = df["close"].rolling(window=10).mean()
    return df


# =============================================================================
# 2) MISCELLANEOUS FEATURES
# =============================================================================
def spread(df: pd.DataFrame) -> pd.DataFrame:
    """Spread between high and low."""
    dfc = df.copy()
    dfc["spread"] = dfc["high"] - dfc["low"]
    return dfc


def auto_corr_multi(
    df: pd.DataFrame, col: str, n: int = 50, lags: List[int] = [1, 3, 5, 10]
) -> pd.DataFrame:
    """Rolling autocorrelation for multiple lags."""
    dfc = df.copy()
    for lag in lags:
        dfc[f"autocorr_{lag}"] = (
            dfc[col]
            .rolling(window=n, min_periods=n)
            .apply(lambda x: x.autocorr(lag=lag), raw=False)
        )
    return dfc


def candle_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds candle-specific features:
      - candle_way (1 if close > open else 0)
      - fill      (real body / range)
      - amplitude (abs(close - open) / open)
    """
    dfc = df.copy()
    dfc["candle_way"] = (dfc["close"] > dfc["open"]).astype(int)
    rng = (dfc["high"] - dfc["low"]).replace(0, np.nan)
    dfc["fill"] = (dfc["close"] - dfc["open"]).abs() / (rng + 1e-5)
    dfc["amplitude"] = (dfc["close"] - dfc["open"]).abs() / (dfc["open"].abs() + 1e-5)
    return dfc


def log_transform(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """
    Create log(price) and n-period log-return: log_ret_n = log(col).diff(n).
    """
    dfc = df.copy()
    # clip to avoid log(0); if strictly positive, you can drop clip
    dfc[f"log_{col}"] = np.log(dfc[col].clip(lower=1e-12))
    dfc[f"log_ret_{n}"] = dfc[f"log_{col}"].diff(n)
    return dfc


def mathematical_derivatives(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Velocity and acceleration for a given column."""
    dfc = df.copy()
    dfc["velocity"] = dfc[col].diff()
    dfc["acceleration"] = dfc["velocity"].diff()
    return dfc


# =============================================================================
# 3) VOLATILITY ESTIMATORS
# =============================================================================
def parkinson_estimator(window: pd.DataFrame) -> float:
    n = len(window)
    if n < 1:
        return np.nan
    sum_sq = np.sum(np.log(window["high"] / window["low"]) ** 2)
    return math.sqrt(sum_sq / (4 * math.log(2) * n))


def moving_parkinson_estimator(df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    dfc = df.copy()
    rolling_vol = pd.Series(dtype="float64", index=dfc.index)
    for i in range(window_size, len(dfc)):
        w = dfc.iloc[i - window_size : i]
        rolling_vol.iloc[i] = parkinson_estimator(w)
    dfc["rolling_volatility_parkinson"] = rolling_vol
    return dfc


def yang_zhang_estimator(window: pd.DataFrame) -> float:
    n = len(window)
    if n < 1:
        return np.nan
    term1 = np.log(window["high"] / window["low"]) ** 2
    term2 = np.log(window["close"] / window["open"]) ** 2
    return math.sqrt(np.mean(term1 + term2))


def moving_yang_zhang_estimator(df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    dfc = df.copy()
    rolling_vol = pd.Series(dtype="float64", index=dfc.index)
    for i in range(window_size, len(dfc)):
        w = dfc.iloc[i - window_size : i]
        rolling_vol.iloc[i] = yang_zhang_estimator(w)
    dfc["rolling_volatility_yang_zhang"] = rolling_vol
    return dfc


# =============================================================================
# 4) MARKET REGIME / DC EVENTS
# =============================================================================
def dc_event(P: float, Pext: float, threshold: float) -> int:
    var = (P - Pext) / Pext
    if var >= threshold:
        return 1
    if var <= -threshold:
        return -1
    return 0


def calculate_dc(df: pd.DataFrame, threshold: float = 0.01) -> Tuple[List[int], List[int]]:
    dfc = df.copy()
    prices = dfc["close"].values
    dc_up, dc_down = [], []
    Pext = prices[0]
    direction = 0
    for i in range(1, len(prices)):
        P = prices[i]
        flag = dc_event(P, Pext, threshold)
        if flag == 1:
            dc_up.append(i)
            direction = 1
            Pext = P
        elif flag == -1:
            dc_down.append(i)
            direction = -1
            Pext = P
        else:
            if direction == 1 and P > Pext:
                Pext = P
            elif direction == -1 and P < Pext:
                Pext = P
    return dc_up, dc_down


def calculate_trend(dc_events_up: List[int], dc_events_down: List[int], df: pd.DataFrame):
    trend_events_down = list(sorted(dc_events_down))
    trend_events_up = list(sorted(dc_events_up))
    return trend_events_down, trend_events_up


def market_regime_dc(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    dfc = df.copy()
    dc_up, dc_down = calculate_dc(dfc, threshold=threshold)
    t_down, t_up = calculate_trend(dc_up, dc_down, dfc)
    dfc["market_regime"] = np.nan
    # Map index locations to actual index labels since calculate_dc returns integer indices
    t_up_labels = dfc.index[t_up] if len(t_up) > 0 else []
    t_down_labels = dfc.index[t_down] if len(t_down) > 0 else []
    dfc.loc[t_up_labels, "market_regime"] = 1
    dfc.loc[t_down_labels, "market_regime"] = 0
    dfc["market_regime"] = dfc["market_regime"].ffill().bfill()
    return dfc


def kama_market_regime(df: pd.DataFrame, col: str = "close", n1: int = 10, n2: int = 30) -> pd.DataFrame:
    dfc = df.copy()
    short_kama = dfc[col].ewm(span=n1, adjust=False).mean()
    long_kama = dfc[col].ewm(span=n2, adjust=False).mean()
    dfc["kama_diff"] = short_kama - long_kama
    dfc["kama_trend"] = (dfc["kama_diff"] >= 0).astype(int)
    return dfc


# =============================================================================
# 5) GAP & DISPLACEMENT
# =============================================================================
def gap_detection(df: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    dfc = df.copy()
    cols = [
        "Bullish_gap_inf",
        "Bullish_gap_sup",
        "Bullish_gap_size",
        "Bearish_gap_inf",
        "Bearish_gap_sup",
        "Bearish_gap_size",
    ]
    for c in cols:
        dfc[c] = np.nan
    for i in range(lookback, len(dfc)):
        prev_high = dfc["high"].iloc[i - lookback]
        prev_low = dfc["low"].iloc[i - lookback]
        curr_high = dfc["high"].iloc[i]
        curr_low = dfc["low"].iloc[i]
        if curr_low > prev_high:
            dfc.at[dfc.index[i], "Bullish_gap_inf"] = prev_high
            dfc.at[dfc.index[i], "Bullish_gap_sup"] = curr_low
            dfc.at[dfc.index[i], "Bullish_gap_size"] = curr_low - prev_high
        if curr_high < prev_low:
            dfc.at[dfc.index[i], "Bearish_gap_inf"] = curr_high
            dfc.at[dfc.index[i], "Bearish_gap_sup"] = prev_low
            dfc.at[dfc.index[i], "Bearish_gap_size"] = prev_low - curr_high
    return dfc


def displacement_detection(
    df: pd.DataFrame, type_range: str = "standard", strenght: float = 3.0, period: int = 20
) -> pd.DataFrame:
    dfc = df.copy()
    if type_range == "standard":
        dfc["candle_range"] = (dfc["close"] - dfc["open"]).abs()
    elif type_range == "extrem":
        dfc["candle_range"] = (dfc["high"] - dfc["low"]).abs()
    else:
        raise ValueError("Invalid 'type_range'. Use 'standard' or 'extrem'.")

    dfc["Variation"] = (dfc["close"] / dfc["open"] - 1).abs()
    dfc["STD"] = dfc["candle_range"].rolling(period).std()
    dfc["displacement"] = 0
    mask = dfc["candle_range"] > strenght * dfc["STD"]
    dfc.loc[mask, "displacement"] = 1
    dfc["red_displacement"] = (dfc["displacement"].astype(int) & dfc["displacement"].shift(1).fillna(0).astype(int)).astype(int)
    return dfc


# =============================================================================
# 6) ROLLING ADF DIAGNOSTIC (optional; not used for gating)
# =============================================================================
def rolling_adf_with_flag(
    df: pd.DataFrame, col: str = "close", window_size: int = 50, p_value_threshold=0.05
) -> pd.DataFrame:
    """Compute rolling ADF p-values and a stationarity flag (diagnostic)."""
    dfc = df.copy()
    adf_stat = pd.Series(dtype="float64", index=dfc.index)
    adf_pval = pd.Series(dtype="float64", index=dfc.index)
    flag = pd.Series(dtype="float64", index=dfc.index)

    for i in range(window_size, len(dfc)):
        slice_data = dfc[col].iloc[i - window_size : i].values
        try:
            result = adfuller(slice_data, autolag="AIC")
            adf_stat.iloc[i] = result[0]
            adf_pval.iloc[i] = result[1]
            flag.iloc[i] = 1 if result[1] < p_value_threshold else 0
        except Exception:
            adf_stat.iloc[i] = np.nan
            adf_pval.iloc[i] = np.nan
            flag.iloc[i] = np.nan

    dfc["rolling_adf_stat"] = adf_stat
    dfc["rolling_adf_pval"] = adf_pval
    dfc["stationary_flag"] = flag
    return dfc


# =============================================================================
# 7) DOUBLE-BARRIER LABEL
# =============================================================================
def set_double_barrier_label(
    df: pd.DataFrame, up: float = 0.005, down: float = 0.005, horizon: int = 50
) -> pd.DataFrame:
    dfc = df.copy()
    closes = dfc["close"].values
    labels = np.full(len(closes), np.nan)

    for i in range(len(closes)):
        current = closes[i]
        upper = current * (1 + up)
        lower = current * (1 - down)
        end = min(i + horizon, len(closes))
        for j in range(i + 1, end):
            if closes[j] >= upper:
                labels[i] = 1
                break
            if closes[j] <= lower:
                labels[i] = 0
                break

    dfc["barrier_label"] = labels
    dfc.dropna(subset=["barrier_label"], inplace=True)
    return dfc


# =============================================================================
# 8) FUTURE MARKET REGIME (Directional-Change Example)
# =============================================================================
def future_DC_market_regime(df: pd.DataFrame, threshold: float = 0.03, horizon: int = 10) -> pd.DataFrame:
    dfc = df.copy()
    dfc["future_return"] = dfc["close"].shift(-horizon) / dfc["close"] - 1.0
    dfc["future_market_regime"] = np.nan
    dfc.loc[dfc["future_return"] >= threshold, "future_market_regime"] = 1
    dfc.loc[dfc["future_return"] <= -threshold, "future_market_regime"] = 0
    dfc.dropna(subset=["future_market_regime"], inplace=True)
    return dfc


# =============================================================================
# 9) Fourier features (global)
# =============================================================================
def add_fourier_features(df: pd.DataFrame, col: str = "close", n_components: int = 5) -> pd.DataFrame:
    """
    Global FFT magnitudes (same values on all rows). For truly time-local
    frequency content, implement a rolling FFT (heavier) instead.
    """
    dfc = df.copy()
    fft_vals = np.abs(fft(dfc[col].values))
    for i in range(1, n_components + 1):
        dfc[f"fft_comp_{i}"] = fft_vals[i] if i < len(fft_vals) else np.nan
    return dfc


# =============================================================================
# 10) Stationarity: ADF+KPSS with safe transforms
# =============================================================================
def _is_stationary(series: pd.Series, adf_alpha: float, kpss_alpha: float) -> Dict:
    s = series.dropna().astype(float)
    if len(s) < 30:
        return {"adf_p": np.nan, "kpss_p": np.nan, "stationary": False}
    try:
        adf_p = adfuller(s, autolag="AIC")[1]
    except Exception:
        adf_p = np.nan
    try:
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan
    ok_adf = (not np.isnan(adf_p)) and (adf_p < adf_alpha)
    ok_kpss = (not np.isnan(kpss_p)) and (kpss_p > kpss_alpha)
    return {"adf_p": adf_p, "kpss_p": kpss_p, "stationary": (ok_adf and ok_kpss)}


def _apply_transform(s: pd.Series, kind: str, seasonal_period: Optional[int]) -> pd.Series:
    if kind == "pct_change":
        return s.pct_change()
    if kind == "diff1":
        return s.diff(1)
    if kind == "log_diff1":
        return np.log1p(s.clip(lower=0)).diff(1)
    if kind == "seasonal_diff" and seasonal_period and seasonal_period > 1:
        return s.diff(seasonal_period)
    return s  # fallback


def ensure_stationary_features(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    cfg: Dict = STATIONARITY_CFG,
    exclude: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    For each numeric feature, test ADF+KPSS. If non-stationary,
    apply transforms in cfg['transform_order'] (no look-ahead), re-test,
    and keep only features that pass. Returns (df_out, report).
    """
    if not cfg.get("enabled", True):
        return df, {"enabled": False}

    df_out = df.copy()
    report: Dict = {"config": cfg, "features": {}}

    # choose candidate columns
    if cols is None:
        cols = df_out.select_dtypes(include=[np.number]).columns.tolist()

    exclude_set = set(cfg.get("exclude_cols", set()))
    if exclude:
        exclude_set.update(exclude)
    cols = [c for c in cols if c not in exclude_set]

    drop_cols, added_cols = [], []

    for col in cols:
        s = df_out[col]
        base = _is_stationary(s, cfg["adf_alpha"], cfg["kpss_alpha"])
        entry = {"original": base, "applied": None, "final_col": col}

        if base["stationary"]:
            report["features"][col] = entry
            continue

        # try configured transforms
        applied = False
        for kind in cfg["transform_order"]:
            s_t = _apply_transform(s, kind, cfg.get("seasonal_period"))
            test_t = _is_stationary(s_t, cfg["adf_alpha"], cfg["kpss_alpha"])
            if test_t["stationary"]:
                new_col = f"{col}__{kind}"
                df_out[new_col] = s_t
                entry["applied"] = {"transform": kind, **test_t}
                entry["final_col"] = new_col
                added_cols.append(new_col)
                if not cfg["keep_original"]:
                    drop_cols.append(col)
                applied = True
                break

        # fallback: deeper differencing up to max_diff
        if not applied:
            s_f = s.copy()
            for d in range(1, int(cfg.get("max_diff", 2)) + 1):
                s_f = s_f.diff(1)
                test_f = _is_stationary(s_f, cfg["adf_alpha"], cfg["kpss_alpha"])
                if test_f["stationary"]:
                    new_col = f"{col}__diff{d}"
                    df_out[new_col] = s_f
                    entry["applied"] = {"transform": f"diff{d}", **test_f}
                    entry["final_col"] = new_col
                    added_cols.append(new_col)
                    if not cfg["keep_original"]:
                        drop_cols.append(col)
                    applied = True
                    break

        report["features"][col] = entry

    if drop_cols:
        df_out = df_out.drop(columns=list(set(drop_cols)))

    # clean NaNs introduced by differencing
    df_out = df_out.dropna()

    return df_out, report


# Backward-compatible simple wrapper (kept for API parity with earlier drafts)
def apply_stationarity_test(
    df: pd.DataFrame, threshold: float = 0.05
) -> pd.DataFrame:
    """
    Legacy simple ADF-only differencing (kept for backward compatibility).
    Prefer `ensure_stationary_features` for robust ADF+KPSS handling.
    """
    cfg = STATIONARITY_CFG.copy()
    cfg["adf_alpha"] = threshold
    df_out, _ = ensure_stationary_features(df, cfg=cfg)
    return df_out


# =============================================================================
# 11) Scaling
# =============================================================================
def scale_features(df: pd.DataFrame, cols_to_scale: List[str]) -> pd.DataFrame:
    """
    Fit-transform scaling on the entire frame (risk of leakage).
    Prefer putting scalers INSIDE your ML pipeline (fit on train only),
    or use rolling_zscore below for backtests.
    """
    dfc = df.copy()
    scaler = StandardScaler()
    dfc[cols_to_scale] = scaler.fit_transform(dfc[cols_to_scale])
    return dfc


def rolling_zscore(
    df: pd.DataFrame, cols: List[str], window: int = 200, min_periods: int = 50
) -> pd.DataFrame:
    """Rolling standardization to avoid look-ahead leakage."""
    dfc = df.copy()
    mu = dfc[cols].rolling(window, min_periods=min_periods).mean()
    sd = dfc[cols].rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    dfc[cols] = (dfc[cols] - mu) / sd
    return dfc


# =============================================================================
# 12) PIPELINES
# =============================================================================
def create_features(
    df: pd.DataFrame,
    col: str = "close",
    window_size: int = 30,
    enforce_stationarity: bool = True,
    stationarity_cfg: Optional[Dict] = None,
    use_rolling_zscore: bool = True,
    zscore_window: int = 200,
) -> pd.DataFrame:
    """
    Integrated feature pipeline (TA, autocorr, volatility, Fourier, stationarity).
    Uses ADF+KPSS gating (ensure_stationary_features) and optional rolling z-score.
    """
    dfc = df.copy()

    # TA & misc
    dfc = add_all_ta_features(dfc)
    dfc = spread(dfc)
    dfc = auto_corr_multi(dfc, col="close")
    dfc = rolling_adf_with_flag(dfc, col="close")  # diagnostic

    # transforms & volatility
    dfc = log_transform(dfc, col, 5)
    dfc = moving_yang_zhang_estimator(dfc, window_size)
    dfc = moving_parkinson_estimator(dfc, window_size)

    # frequency features (global simple FFT)
    dfc = add_fourier_features(dfc, col="close")

    # Stationarity enforcement
    if enforce_stationarity:
        cfg = STATIONARITY_CFG.copy()
        if stationarity_cfg:
            cfg.update(stationarity_cfg)
        dfc, _ = ensure_stationary_features(dfc, cfg=cfg)

    # Scaling (choose one: rolling z-score here OR scaling inside ML pipeline)
    numeric_cols = dfc.select_dtypes(include=[np.number]).columns.tolist()
    if use_rolling_zscore and len(numeric_cols) > 0:
        dfc = rolling_zscore(dfc, numeric_cols, window=zscore_window)
    # else: keep raw; or scale later in your sklearn/Keras pipeline

    dfc = dfc.dropna()
    return dfc


def add_core_features(
    df: pd.DataFrame,
    enforce_stationarity: bool = True,
    stationarity_cfg: Optional[Dict] = None,
) -> pd.DataFrame:
    """Lightweight core feature set + optional stationarity enforcement."""
    dfc = df.copy()

    # Trend & momentum
    dfc["sma_20"] = dfc["close"].rolling(20).mean()
    dfc["ema_20"] = dfc["close"].ewm(span=20, adjust=False).mean()
    dfc["kama_10"] = dfc["close"].ewm(span=10, adjust=False).mean()  # placeholder
    dfc["rsi_14"] = ta.momentum.rsi(dfc["close"], window=14)
    macd = ta.trend.macd(dfc["close"])
    macd_signal = ta.trend.macd_signal(dfc["close"])
    dfc["macd_diff"] = macd - macd_signal

    # Volatility & volume
    dfc["atr_14"] = ta.volatility.average_true_range(dfc["high"], dfc["low"], dfc["close"], window=14)
    dfc["obv"] = ta.volume.on_balance_volume(dfc["close"], dfc["tick_volume"])
    dfc["rolling_std_20"] = dfc["close"].rolling(20).std()

    # Structure & candle
    dfc = spread(dfc)
    dfc = candle_information(dfc)

    # Autocorrelation
    dfc = auto_corr_multi(dfc, col="close", n=50, lags=[1, 5, 10])

    # Regime
    dfc = kama_market_regime(dfc, col="close", n1=10, n2=30)
    dfc["ma_short"] = dfc["close"].rolling(20).mean()
    dfc["ma_long"] = dfc["close"].rolling(50).mean()
    dfc["market_regime"] = 0
    dfc.loc[dfc["ma_short"] > dfc["ma_long"], "market_regime"] = 1
    dfc.loc[dfc["ma_short"] < dfc["ma_long"], "market_regime"] = -1

    # Diagnostic rolling ADF on close
    dfc = rolling_adf_with_flag(dfc, col="close", window_size=50)

    dfc = dfc.dropna().reset_index(drop=True)

    # Stationarity gating on derived features
    if enforce_stationarity:
        cfg = STATIONARITY_CFG.copy()
        if stationarity_cfg:
            cfg.update(stationarity_cfg)
        dfc, _ = ensure_stationary_features(dfc, cfg=cfg)

    return dfc
