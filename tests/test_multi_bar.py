import sys
from unittest.mock import MagicMock
# Mock out MetaTrader5 before importing our module
sys.modules['MetaTrader5'] = MagicMock()

import unittest
from unittest.mock import patch
import pandas as pd
from live_trading.multi_bar import TradingApp, log_and_print

class TestTradingApp(unittest.TestCase):
    def setUp(self):
        self.app = TradingApp(symbol="EURUSD", lot_size=0.01, magic_number=123456)

    @patch("live_trading.multi_bar.mt5.copy_rates_from_pos")
    @patch("live_trading.multi_bar.log_and_print")
    def test_get_data_returns_none(self, mock_log, mock_copy_rates):
        mock_copy_rates.return_value = None

        # Test what happens when mt5.copy_rates_from_pos returns None
        result = self.app.get_data("EURUSD", 100, 16408)

        self.assertIsNone(result)
        mock_log.assert_called_once_with("Could not retrieve data for EURUSD", is_error=True)
