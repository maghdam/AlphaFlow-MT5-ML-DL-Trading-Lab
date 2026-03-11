import sys
from unittest.mock import MagicMock

# Mock dependencies before importing TradingApp
sys.modules['MetaTrader5'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['ta'] = MagicMock()
sys.modules['joblib'] = MagicMock()

import unittest
from unittest.mock import patch
from live_trading.multi_bar import TradingApp

class TestMultiBarTradingApp(unittest.TestCase):
    @patch('live_trading.multi_bar.joblib.load')
    @patch('live_trading.multi_bar.log_and_print')
    def test_load_pipeline_exception_handling(self, mock_log_and_print, mock_joblib_load):
        # Arrange
        app = TradingApp(symbol="EURUSD", lot_size=0.01, magic_number=234003)
        mock_joblib_load.side_effect = Exception("Mocked joblib load error")

        # Act
        app.load_pipeline("dummy_path.pkl")

        # Assert
        self.assertIsNone(app.pipeline)
        mock_joblib_load.assert_called_once_with("dummy_path.pkl")
        mock_log_and_print.assert_called_once_with("Failed to load pipeline: Mocked joblib load error", is_error=True)

    @patch('live_trading.multi_bar.joblib.load')
    @patch('live_trading.multi_bar.log_and_print')
    def test_load_pipeline_success(self, mock_log_and_print, mock_joblib_load):
        # Arrange
        app = TradingApp(symbol="EURUSD", lot_size=0.01, magic_number=234003)
        mock_pipeline = "mock_pipeline_instance"
        mock_joblib_load.return_value = mock_pipeline

        # Act
        app.load_pipeline("valid_path.pkl")

        # Assert
        self.assertEqual(app.pipeline, mock_pipeline)
        mock_joblib_load.assert_called_once_with("valid_path.pkl")
        mock_log_and_print.assert_called_once_with("Pipeline loaded from valid_path.pkl")

if __name__ == '__main__':
    unittest.main()
