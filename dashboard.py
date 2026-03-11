import streamlit as st
import sqlite3
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from pathlib import Path
from typing import Optional

# --- Constants ---
N_FORWARD = 3  # Set this to match your trading bot's N_FORWARD
DB_PATH = Path("live_signals.db")
TABLE_NAME = "signals"
COL_TIMESTAMP = "timestamp"
COL_SYMBOL = "symbol"
COL_PREDICTION = "prediction"

# --- Page Config ---
st.set_page_config(page_title="AlphaFlow Live Signals", layout="wide")
st.title("📈 AlphaFlow Trading Bot - Live Signals")

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="refresh")

# --- Data Loading ---
@st.cache_data(ttl=30)
def load_data() -> Optional[pd.DataFrame]:
    """Load signal data from the SQLite database."""
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f'SELECT * FROM {TABLE_NAME} ORDER BY {COL_TIMESTAMP} DESC', conn)
        conn.close()
        return df
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        st.error(f"Database error: {e}")
        return None

def display_latest_signals(df: pd.DataFrame):
    st.subheader("Latest Signal Per Symbol")
    latest = (
        df.sort_values(COL_TIMESTAMP)
        .groupby(COL_SYMBOL)
        .tail(1)
        .sort_values(COL_SYMBOL)
        .reset_index(drop=True)
    )
    st.dataframe(latest[[COL_SYMBOL, COL_PREDICTION, COL_TIMESTAMP]], use_container_width=True)

def display_recent_signals(df: pd.DataFrame):
    st.subheader(f"Latest {N_FORWARD} Signals Per Symbol")
    recent = (
        df.sort_values([COL_SYMBOL, COL_TIMESTAMP])
        .groupby(COL_SYMBOL)
        .tail(N_FORWARD)
        .sort_values([COL_SYMBOL, COL_TIMESTAMP])
        .reset_index(drop=True)
    )
    st.dataframe(recent[[COL_SYMBOL, COL_PREDICTION, COL_TIMESTAMP]], use_container_width=True)

    # Optional: Show a quick mini-forecast chart per symbol
    st.write("---")
    st.subheader("Mini Signal Forecasts (per symbol)")
    for symbol, group in df.sort_values(COL_TIMESTAMP).groupby(COL_SYMBOL):
        mini_df = group.tail(N_FORWARD)
        # Only show if there is more than one unique value
        if mini_df[COL_PREDICTION].nunique() > 1:
            st.write(f"**{symbol}**")
            st.line_chart(mini_df.set_index(COL_TIMESTAMP)[[COL_PREDICTION]], height=100, use_container_width=True)

def display_signal_history(df: pd.DataFrame):
    with st.expander("Show Full Signal History"):
        st.dataframe(df[[COL_SYMBOL, COL_PREDICTION, COL_TIMESTAMP]], use_container_width=True)

def display_signal_distribution(df: pd.DataFrame):
    st.subheader("Signal Distribution (All Time)")
    signal_counts = df.groupby([COL_SYMBOL, COL_PREDICTION]).size().unstack(fill_value=0)
    st.bar_chart(signal_counts)

def main():
    """Main function to run the Streamlit dashboard."""
    # Show signal legend and refresh time
    st.markdown("""
    | Signal | Meaning |
    |--------|---------|
    | -1     | **Sell**|
    | 0      | **Flat**|
    | 1      | **Buy** |
    """)
    st.caption(f"Last refreshed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} (server time, probably UTC)")

    df = load_data()

    if df is None or df.empty:
        st.warning("No signals found in the database yet. Please wait for signals to be generated.")
        return

    view_mode = st.radio(
        "View Mode",
        ["Latest Signal Per Symbol", f"Latest {N_FORWARD} Signals Per Symbol"],
        horizontal=True,
    )

    if view_mode == "Latest Signal Per Symbol":
        display_latest_signals(df)
    else:
        display_recent_signals(df)

    display_signal_history(df)
    display_signal_distribution(df)

if __name__ == "__main__":
    main()
