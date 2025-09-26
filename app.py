from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from datetime import datetime
from db import db_manager
from logging_config import get_logger
from bybit_client import BybitClient

# Logging using centralized system
logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AlgoTrader Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if "trading_mode" not in st.session_state:
    st.session_state.trading_mode = "virtual"
if "engine_initialized" not in st.session_state:
    st.session_state.engine_initialized = False
if "wallet_cache" not in st.session_state:
    st.session_state.wallet_cache = {}  # Store balances per mode
if "bybit_client" not in st.session_state:
    st.session_state.bybit_client = None
if "engine" not in st.session_state:
    st.session_state.engine = None

# --- Initialize trading engine ---
def initialize_engine():
    try:
        from engine import TradingEngine
        if not st.session_state.engine_initialized:
            st.session_state.engine = TradingEngine()
            st.session_state.engine_initialized = True
            logger.info("Trading engine initialized successfully")
        return True
    except Exception as e:
        st.error(f"Failed to initialize trading engine: {e}")
        logger.error(f"Engine initialization failed: {e}", exc_info=True)
        return False

# --- Initialize Bybit client ---
def initialize_bybit():
    if st.session_state.bybit_client is None:
        st.session_state.bybit_client = BybitClient()
        if st.session_state.bybit_client._test_connection():
            logger.info("Bybit client connected successfully")
        else:
            st.warning("Bybit client connection failed. Check API keys.")
            logger.error("Bybit client connection test failed")

# --- Fetch wallet balance ---
def get_wallet_balance() -> dict:
    """
    Fetch wallet balance based on the selected trading mode.
    Returns a dict with capital, available, and used balances.
    Always safe, fallback to defaults.
    """
    mode = st.session_state.trading_mode
    default_virtual = {"capital": 100.0, "available": 100.0, "used": 0.0}
    default_real = {"capital": 0.0, "available": 0.0, "used": 0.0}

    # Check cache
    if mode in st.session_state.wallet_cache:
        logger.info(f"Returning cached {mode} balance: {st.session_state.wallet_cache[mode]}")
        return st.session_state.wallet_cache[mode]

    balance_data = default_virtual if mode == "virtual" else default_real
    try:
        if mode == "virtual":
            wallet = st.session_state.engine.db.get_wallet_balance("virtual") \
                if st.session_state.engine else None
            if wallet:
                balance_data = {
                    "capital": getattr(wallet, "capital", default_virtual["capital"]),
                    "available": getattr(wallet, "available", default_virtual["available"]),
                    "used": getattr(wallet, "used", default_virtual["used"])
                }
                logger.info(f"Fetched virtual wallet balance: {balance_data}")
        else:  # real mode
            initialize_bybit()
            client = st.session_state.bybit_client
            if client and client.is_connected():
                # Sync real balance with Bybit
                st.session_state.engine.sync_real_balance()
                wallet = st.session_state.engine.db.get_wallet_balance("real")
                if wallet:
                    balance_data = {
                        "capital": getattr(wallet, "capital", default_real["capital"]),
                        "available": getattr(wallet, "available", default_real["available"]),
                        "used": getattr(wallet, "used", default_real["used"])
                    }
                    logger.info(
                        f"Fetched real wallet balance after sync: capital=${balance_data['capital']:.2f}, "
                        f"available=${balance_data['available']:.2f}, used=${balance_data['used']:.2f}"
                    )
                else:
                    logger.warning("Failed to retrieve real balance after sync")
                    st.error("❌ Failed to retrieve real balance. Check Bybit account or API permissions.")
            else:
                wallet = st.session_state.engine.db.get_wallet_balance("real") \
                    if st.session_state.engine else None
                if wallet:
                    balance_data = {
                        "capital": getattr(wallet, "capital", default_real["capital"]),
                        "available": getattr(wallet, "available", default_real["available"]),
                        "used": getattr(wallet, "used", default_real["used"])
                    }
                    logger.warning(f"Bybit client not connected, using DB real balance: {balance_data}")
                st.warning("Bybit API not connected. Check API keys in .env file.")
    except Exception as e:
        logger.error(f"Error fetching {mode} wallet: {e}", exc_info=True)
        balance_data = default_virtual if mode == "virtual" else default_real

    # Cache balance for this session
    st.session_state.wallet_cache[mode] = balance_data
    logger.info(f"Cached {mode} balance: {balance_data}")

    # Conditional messaging for real balance
    if mode == "real":
        if balance_data["available"] <= 0:
            st.warning("Real available balance is low or zero. Deposit funds on Bybit.")
    return balance_data

def main():
    initialize_engine()
    # Load saved trading mode from DB
    if "trading_mode" not in st.session_state or st.session_state.trading_mode is None:
        saved_mode = st.session_state.engine.db.get_setting("trading_mode")
        st.session_state.trading_mode = saved_mode if saved_mode in ["virtual", "real"] else "virtual"
        logger.info(f"Loaded trading mode from DB: {st.session_state.trading_mode}")

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")

        # Mode selector with confirmation for real mode
        mode_options = ["Virtual", "Real"]
        selected_mode_index = 0 if st.session_state.trading_mode == "virtual" else 1
        selected_mode = st.selectbox(
            "Trading Mode",
            mode_options,
            index=selected_mode_index
        )
        confirm_real = False
        if selected_mode.lower() != st.session_state.trading_mode:
            if selected_mode.lower() == "real":
                confirm_real = st.checkbox("Confirm: I understand this enables LIVE trading on Bybit")
                if not confirm_real:
                    st.warning("Must confirm to switch to real mode.")
            if selected_mode.lower() != "real" or confirm_real:
                st.session_state.trading_mode = selected_mode.lower()
                st.session_state.engine.db.save_setting("trading_mode", st.session_state.trading_mode)
                st.session_state.wallet_cache.clear()
                logger.info(f"Switched to {st.session_state.trading_mode} mode, cleared cache")

                # Sync if switching to real mode
                if st.session_state.trading_mode == "real":
                    initialize_bybit()
                    if st.session_state.bybit_client and st.session_state.bybit_client.is_connected():
                        st.session_state.engine.sync_real_balance()
                        st.session_state.engine.sync_real_trades()  # Fixed: Added trade sync
                        logger.info("Real balance and trades synced after mode switch")
                st.rerun()

        # Status
        engine_status = "🟢 Online" if st.session_state.engine_initialized else "🔴 Offline"
        st.markdown(f"**Engine Status:** {engine_status}")
        mode_color = "🟢" if st.session_state.trading_mode == "virtual" else "🟡"
        st.markdown(f"**Trading Mode:** {mode_color} {st.session_state.trading_mode.title()}")

        # API Connection Status
        initialize_bybit()
        api_status = "✅ Connected" if st.session_state.bybit_client and st.session_state.bybit_client.is_connected() else "❌ Disconnected"
        st.markdown(f"**API Status:** {api_status}")

        st.divider()

        # Pages
        pages = {
            "📊 Dashboard": "pages/dashboard.py",
            "🎯 Signals": "pages/signals.py",
            "📈 Trades": "pages/trades.py",
            "📊 Performance": "pages/performance.py",
            "⚙️ Settings": "pages/settings.py"
        }

        for name, path in pages.items():
            if st.button(name):
                st.switch_page(path)

        st.divider()

        # Wallet Balance
        balance = get_wallet_balance()  # Use fixed function
        current_mode = st.session_state.trading_mode
        capital_val = balance["capital"]
        available_val = max(balance["available"], 0.0)
        used_val = capital_val - available_val
        if abs(used_val) < 0.01:
            used_val = 0.0

        if current_mode == "virtual":
            st.metric("💻 Virtual Capital", f"${capital_val:.2f}")
            st.metric("💻 Virtual Available", f"${available_val:.2f}")
            st.metric("💻 Virtual Used", f"${used_val:.2f}")
        else:
            st.metric("🏦 Real Capital", f"${capital_val:.2f}")
            st.metric("🏦 Real Available", f"${available_val:.2f}")
            st.metric("🏦 Real Used Margin", f"${used_val:.2f}")

        # Last updated
        st.markdown(
            f"<small style='color:#888;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>",
            unsafe_allow_html=True
        )

        # Emergency Stop - Fixed: Also stop automated trader if running
        if st.button("🛑 Emergency Stop"):
            st.session_state.wallet_cache.clear()
            if "automated_trader" in st.session_state:  # Assume initialized elsewhere; add if needed
                st.session_state.automated_trader.stop()
            st.success("All automated trading stopped and cache cleared")
            logger.info("Emergency stop triggered, cache cleared")

    # --- Main dashboard ---
    try:
        from pages.dashboard import main as dashboard_main
        dashboard_main()
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
        logger.error(f"Dashboard error: {e}", exc_info=True)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#888;'>AlgoTrader Pro v1.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()