import sys
from unittest.mock import MagicMock

# Mock MetaTrader5 before importing data_loader
mt5_mock = MagicMock()
sys.modules['MetaTrader5'] = mt5_mock

import pandas as pd
import pytest
from data.data_loader import get_data_mt5

def test_get_data_mt5_live_trading():
    """Test get_data_mt5 when start_pos is None (live trading)."""
    # Arrange
    symbol = "BTCUSD"
    n_bars = 100
    timeframe = mt5_mock.TIMEFRAME_H1

    # Mock return value of copy_rates_from_pos
    mock_rates = [
        {"time": 1600000000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
        {"time": 1600003600, "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0},
    ]
    mt5_mock.copy_rates_from_pos.return_value = mock_rates

    # Act
    df = get_data_mt5(symbol, n_bars, timeframe)

    # Assert
    mt5_mock.copy_rates_from_pos.assert_called_once_with(symbol, timeframe, 0, n_bars)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'time'
    assert len(df) == 2
    assert "open" in df.columns
    assert df.index[0] == pd.to_datetime(1600000000, unit='s')

def test_get_data_mt5_backtesting():
    """Test get_data_mt5 when start_pos is provided (backtesting)."""
    # Arrange
    mt5_mock.copy_rates_from_pos.reset_mock()
    symbol = "EURUSD"
    n_bars = 50
    timeframe = mt5_mock.TIMEFRAME_M15
    start_pos = 10

    mock_rates = [
        {"time": 1600000000, "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15},
    ]
    mt5_mock.copy_rates_from_pos.return_value = mock_rates

    # Act
    df = get_data_mt5(symbol, n_bars, timeframe, start_pos=start_pos)

    # Assert
    mt5_mock.copy_rates_from_pos.assert_called_once_with(symbol, timeframe, start_pos, n_bars)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'time'
    assert len(df) == 1

def test_get_data_mt5_no_data():
    """Test get_data_mt5 when copy_rates_from_pos returns None."""
    # Arrange
    mt5_mock.copy_rates_from_pos.reset_mock()
    symbol = "INVALID"
    n_bars = 10
    timeframe = mt5_mock.TIMEFRAME_H1

    mt5_mock.copy_rates_from_pos.return_value = None

    # Act & Assert
    with pytest.raises(ValueError, match=f"Could not retrieve data for {symbol}"):
        get_data_mt5(symbol, n_bars, timeframe)

    mt5_mock.copy_rates_from_pos.assert_called_once_with(symbol, timeframe, 0, n_bars)
