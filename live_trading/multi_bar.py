# LIVE TRADING CODE FOR MULTI-BAR CLASSIFICATION

import sys
import os
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) SET PROJECT ROOT AND UPDATE PATH/WORKING DIRECTORY
# ---------------------------------------------------------------------------
project_root = Path.cwd().parent.parent  # Adjust if your notebook is in notebooks/time_series
sys.path.append(str(project_root))
os.chdir(str(project_root))
warnings.filterwarnings("ignore")


import warnings
warnings.filterwarnings("ignore")
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import time
import logging
import joblib

# Setup logging
logging.basicConfig(
    filename='models/saved_models/trading_app1.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_and_print(message, is_error=False):
    """
    Logs and prints a message.
    If is_error=True, logs at the ERROR level; otherwise logs at INFO level.
    """
    if is_error:
        logging.error(message)
    else:
        logging.info(message)
    print(message)

# Update the login credentials and server information accordingly
name = 66677507
key = 'ST746$nG38'
serv = 'ICMarketsSC-Demo'

# Global variables
SYMBOL = "EURUSD"
LOT_SIZE = 0.01
TIMEFRAME = mt5.TIMEFRAME_D1
N_BARS = 50000
MAGIC_NUMBER = 234003
SLEEP_TIME = 86400  # 24 hours in seconds
COMMENT_ML = "RFFV-D"

# If you still need feature selection, you can keep this helper function:
def select_features_rf_reg(X, y, estimator, max_features=20):
    """
    Example helper function for feature selection using RandomForest.
    """
    from sklearn.feature_selection import SelectFromModel
    selector = SelectFromModel(estimator=estimator, threshold=-np.inf, max_features=max_features).fit(X, y)
    X_transformed = selector.transform(X)
    selected_features_mask = selector.get_support()
    return X_transformed, selected_features_mask

class TradingApp:
    def __init__(self, symbol, lot_size, magic_number):
        self.symbol = symbol
        self.lot_size = lot_size
        self.magic_number = magic_number
        self.pipeline = None  # We'll store the loaded classification pipeline here
        self.last_retrain_time = None

    def get_data(self, symbol, n, timeframe):
        """
        Fetch 'n' bars of historical data for the given symbol and timeframe.
        """
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        if rates is None:
            log_and_print(f"Could not retrieve data for {symbol}", is_error=True)
            return None
        rates_frame = pd.DataFrame(rates)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        rates_frame.set_index('time', inplace=True)
        return rates_frame

    def add_all_ta_features(self, df):
        """
        Add technical analysis features to the DataFrame using the 'ta' library.
        """
        df = ta.add_all_ta_features(
            df, open="open", high="high", low="low", close="close", volume="tick_volume", fillna=True
        )
        return df

    def load_pipeline(self, pipeline_path):
        """
        Loads a pre-trained classification pipeline (e.g., 'best_rf_pipeline.pkl').
        This pipeline is expected to produce SHIFTED labels [0,1,2].
        """
        self.pipeline = joblib.load(pipeline_path)
        logging.info(f"Loaded pipeline from {pipeline_path}")
        log_and_print(f"Loaded pipeline from {pipeline_path}")

    def ml_signal_generation(self, symbol, n_bars, timeframe):
        """
        Generate buy/sell signals using the loaded classification pipeline.
        The pipeline outputs SHIFTED labels in {0,1,2} => we SHIFT them back to {-1,0,+1}.
        We'll interpret +1 => buy, -1 => sell, 0 => no trade.
        """
        if self.pipeline is None:
            logging.error("No pipeline loaded. Call load_pipeline(...) first.")
            return False, False, True, True

        # 1) Fetch new data
        df = self.get_data(symbol, n_bars, timeframe)

        # 2) Add TA features
        df = self.add_all_ta_features(df)
        df.fillna(method='ffill', inplace=True)

        # 3) Prepare the features
        X_new = df  # The pipeline must handle columns in the correct order.

        # 4) Predict SHIFTED classes
        preds_shifted = self.pipeline.predict(X_new)
        # SHIFT them back: 0->-1, 1->0, 2->+1
        preds = preds_shifted - 1

        # Get the latest predicted class
        latest_pred = preds[-1]
        # If latest_pred == +1 => buy signal
        # If latest_pred == -1 => sell signal
        # If 0 => do nothing
        buy_signal = (latest_pred == 1)
        sell_signal = (latest_pred == -1)

        return buy_signal, sell_signal, not buy_signal, not sell_signal

    def orders(self, symbol, lot, is_buy=True, id_position=None, sl=None, tp=None):
        """
        Place an order (BUY or SELL) for the specified symbol and lot size.
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            log_and_print(f"Symbol {symbol} not found, can't place order.", is_error=True)
            return "Symbol not found"

        # Make sure symbol is visible
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                log_and_print(f"Failed to select symbol {symbol}", is_error=True)
                return "Symbol not visible or could not be selected."

        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            log_and_print(f"Could not get tick info for {symbol}.", is_error=True)
            return "Tick info unavailable"

        # Check for valid bid/ask
        if tick_info.bid <= 0 or tick_info.ask <= 0:
            log_and_print(
                f"Zero or invalid bid/ask for {symbol}: bid={tick_info.bid}, ask={tick_info.ask}",
                is_error=True
            )
            return "Invalid prices"

        # LOT SIZE VALIDATION
        lot = max(lot, symbol_info.volume_min)
        step = symbol_info.volume_step
        if step > 0:
            remainder = lot % step
            if remainder != 0:
                lot = lot - remainder + step
        if lot > symbol_info.volume_max:
            lot = symbol_info.volume_max

        log_and_print(
            f"Adjusted lot size to {lot} (min={symbol_info.volume_min}, "
            f"step={symbol_info.volume_step}, max={symbol_info.volume_max})"
        )

        # Force ORDER_FILLING_IOC
        filling_mode = 1  # ORDER_FILLING_IOC

        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        order_price = tick_info.ask if is_buy else tick_info.bid
        deviation = 20

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "deviation": deviation,
            "magic": self.magic_number,
            "comment": COMMENT_ML,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        if id_position is not None:
            request["position"] = id_position

        log_and_print(f"Sending order request: {request}")
        result = mt5.order_send(request)

        order_type_str = "BUY" if is_buy else "SELL"
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error_message = f"Order failed for {symbol}"
            if result:
                error_message += f", retcode={result.retcode}, comment={result.comment}"
            additional_info = (
                f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Order Type: {order_type_str}\n"
                f"Lot Size: {lot}\n"
                f"SL: {sl if sl else 'None'}\n"
                f"TP: {tp if tp else 'None'}\n"
                f"Comment: {COMMENT_ML}\n"
                f"Request: {request}\n"
                f"Result: {result}"
            )
            # If you want notifications, you could log or handle them differently here.
            log_and_print(f"Order failed details: {additional_info}", is_error=True)
        else:
            success_message = f"Order successful for {symbol}, comment={result.comment}"
            additional_info = (
                f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Order Type: {order_type_str}\n"
                f"Lot Size: {lot}\n"
                f"SL: {sl if sl else 'None'}\n"
                f"TP: {tp if tp else 'None'}\n"
                f"Comment: {COMMENT_ML}"
            )
            # If you want notifications, you could log or handle them differently here.
            log_and_print(success_message)

    def get_positions_by_magic(self, symbol, magic_number):
        """
        Retrieve positions for a specific symbol and magic number.
        """
        all_positions = mt5.positions_get(symbol=symbol)
        if not all_positions:
            log_and_print("No positions found.", is_error=False)
            return []
        return [pos for pos in all_positions if pos.magic == magic_number]

    def run_strategy(self, symbol, lot, buy_signal, sell_signal):
        """
        Run the trading strategy logic based on buy/sell signals.
        """
        log_and_print("------------------------------------------------------------------")
        log_and_print(
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
            f"SYMBOL: {symbol}, BUY SIGNAL: {buy_signal}, SELL SIGNAL: {sell_signal}"
        )

        positions = self.get_positions_by_magic(symbol, self.magic_number)
        has_buy = any(pos.type == mt5.POSITION_TYPE_BUY for pos in positions)
        has_sell = any(pos.type == mt5.POSITION_TYPE_SELL for pos in positions)

        if buy_signal and not has_buy:
            if has_sell:
                log_and_print("Existing sell positions found. Attempting to close...")
                if self.close_position(symbol, is_buy=True):
                    log_and_print("Sell positions closed. Placing new buy order.")
                    self.orders(symbol, lot, is_buy=True)
                else:
                    log_and_print("Failed to close sell positions.")
            else:
                self.orders(symbol, lot, is_buy=True)
        elif sell_signal and not has_sell:
            if has_buy:
                log_and_print("Existing buy positions found. Attempting to close...")
                if self.close_position(symbol, is_buy=False):
                    log_and_print("Buy positions closed. Placing new sell order.")
                    self.orders(symbol, lot, is_buy=False)
                else:
                    log_and_print("Failed to close buy positions.")
            else:
                self.orders(symbol, lot, is_buy=False)
        else:
            log_and_print("Appropriate position already exists or no signal to act on.")

    def close_position(self, symbol, is_buy):
        """
        Closes positions of the opposite type (BUY/SELL) for this app's magic number.
        """
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            log_and_print(f"No positions to close for symbol: {symbol}")
            return False

        initial_balance = mt5.account_info().balance
        closed_any = False

        for position in positions:
            # Close positions of the opposite type with the same magic number
            if position.magic == self.magic_number and (
                (is_buy and position.type == mt5.POSITION_TYPE_SELL) or
                (not is_buy and position.type == mt5.POSITION_TYPE_BUY)
            ):
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_BUY if position.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                    "position": position.ticket,
                    "deviation": 20,
                    "magic": self.magic_number,
                    "comment": COMMENT_ML,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(close_request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    error_message = f"Failed to close position {position.ticket} for {symbol}: {result.retcode}"
                    log_and_print(error_message, is_error=True)
                    # If you want notifications, you could log or handle them differently here.
                else:
                    log_and_print(f"Successfully closed position {position.ticket} for {symbol}")
                    closed_any = True

        if closed_any:
            final_balance = mt5.account_info().balance
            profit = final_balance - initial_balance
            success_message = f"Closed positions successfully, Profit: {profit}"
            log_and_print(success_message)
            return True

        return False

    def check_and_execute_trades(self):
        """
        Convenience method to perform the entire flow:
        generate signals, run strategy, and deselect symbol.
        """
        mt5.symbol_select(self.symbol, True)
        buy, sell, _, _ = self.ml_signal_generation(self.symbol, N_BARS, TIMEFRAME)
        self.run_strategy(self.symbol, self.lot_size, buy, sell)
        mt5.symbol_select(self.symbol, False)
        log_and_print("Waiting for new signals...")

def is_market_open():
    """
    Check if the current time is within the typical Forex trading session, adjusted for CET/CEST.
    Market closes at Friday 10:00 PM CET and opens at Sunday 11:00 PM CET. 
    It is closed all day Saturday.
    """
    current_time_utc = datetime.utcnow()
    # Adjust for Central European Time (UTC+1) or Central European Summer Time (UTC+2)
    current_time_cet = (
        current_time_utc + timedelta(hours=2) 
        if time.localtime().tm_isdst 
        else current_time_utc + timedelta(hours=1)
    )

    # Friday after 10 PM CET
    if current_time_cet.weekday() == 4 and current_time_cet.hour >= 22:
        return False
    # Sunday before 11 PM CET
    elif current_time_cet.weekday() == 6 and current_time_cet.hour < 23:
        return False
    # All day Saturday
    elif current_time_cet.weekday() == 5:
        return False
    return True

if __name__ == "__main__":
    try:
        if not mt5.initialize(login=name, server=serv, password=key):
            log_and_print("Failed to initialize MetaTrader 5", is_error=True)
            exit()

        app = TradingApp(symbol=SYMBOL, lot_size=LOT_SIZE, magic_number=MAGIC_NUMBER)

        # 1) Load the classification pipeline
        pipeline_path = "models/saved_models/best_rf_mb_pipeline.pkl"
        app.load_pipeline(pipeline_path)

        while True:
            log_and_print("Checking market status...")
            if is_market_open():
                log_and_print("Market is open. Executing trades...")

                # 2) Generate signals using the loaded pipeline
                #    This pipeline is classification-based => SHIFTED labels [0,1,2]
                #    ml_signal_generation() SHIFTs them back to [-1,0,+1] for signals
                buy_signal, sell_signal, _, _ = app.ml_signal_generation(
                    symbol=app.symbol,
                    n_bars=N_BARS,
                    timeframe=TIMEFRAME
                )

                # 3) Run strategy
                app.run_strategy(app.symbol, app.lot_size, buy_signal, sell_signal)
            else:
                log_and_print("Market is closed. No actions performed.")

            time.sleep(SLEEP_TIME)

    except KeyboardInterrupt:
        log_and_print("Shutdown signal received.")
        # If you need a notification here, handle it (e.g., log, email, etc.).
    except Exception as e:
        error_message = f"An error occurred: {e}"
        log_and_print(error_message, is_error=True)
        # If you need a notification here, handle it (e.g., log, email, etc.).
    finally:
        mt5.shutdown()
        log_and_print("MetaTrader 5 shutdown completed.")
        # If you need a notification here, handle it (e.g., log, email, etc.).
