from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from datetime import datetime
from db import db_manager
from logging_config import get_logger
from bybit_client import BybitClient
from license_manager import check_license, format_expiration_date

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
if "license_valid" not in st.session_state:
    st.session_state.license_valid = False
if "license_key" not in st.session_state:
    st.session_state.license_key = None

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

# --- Helper: does DB already have any REAL trades? ---
def db_has_any_trade(mode: str = "real") -> bool:
    """
    Return True if the DB already contains at least one trade for the given mode.
    Tries several common DB methods and fails closed (False) without raising.
    """
    try:
        db = st.session_state.engine.db if st.session_state.engine else None
        if not db:
            return False

        # Try a few likely APIs, quietly skipping if missing
        try:
            return (db.get_trade_count(mode) or 0) > 0
        except Exception:
            pass
        try:
            return (db.count_trades(mode) or 0) > 0
        except Exception:
            pass
        try:
            return bool(db.has_trades(mode))
        except Exception:
            pass
        try:
            return db.get_last_trade(mode) is not None
        except Exception:
            pass
        try:
            trades = db.get_trades(mode=mode, limit=1)
            return bool(trades)
        except Exception:
            pass
        try:
            trades = db.get_trades(mode)
            return bool(trades)
        except Exception:
            pass

        return False
    except Exception as e:
        logger.error(f"db_has_any_trade check failed: {e}", exc_info=True)
        return False

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
            wallet = db_manager.get_wallet_balance("virtual")
            if wallet:
                balance_data = {
                    "capital": wallet.capital,
                    "available": wallet.available,
                    "used": wallet.used
                }
                logger.info(f"Fetched virtual wallet balance: {balance_data}")
        else:  # real mode
            initialize_bybit()
            client = st.session_state.bybit_client
            if client and client.is_connected():
                # Sync real balance with Bybit
                st.session_state.engine.sync_real_balance()
                wallet = db_manager.get_wallet_balance("real")
                if wallet:
                    balance_data = {
                        "capital": wallet.capital,
                        "available": wallet.available,
                        "used": wallet.used
                    }
                    logger.info(
                        f"Fetched real wallet balance after sync: capital=${balance_data['capital']:.2f}, "
                        f"available=${balance_data['available']:.2f}, used=${balance_data['used']:.2f}"
                    )
                else:
                    logger.warning("Failed to retrieve real balance after sync")
                    st.error("❌ Failed to retrieve real balance. Check Bybit account or API permissions.")
            else:
                wallet = db_manager.get_wallet_balance("real")
                if wallet:
                    balance_data = {
                        "capital": wallet.capital,
                        "available": wallet.available,
                        "used": wallet.used
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
    if mode == "real" and balance_data["available"] <= 0:
        st.warning("Real available balance is low or zero. Deposit funds on Bybit.")
    return balance_data

def main():
    # Check license before proceeding
    is_valid, license_result = check_license()
    if not is_valid:
        return  # Stop execution if license is invalid

    initialize_engine()
    # Load saved trading mode from DB
    if "trading_mode" not in st.session_state or st.session_state.trading_mode is None:
        saved_mode = db_manager.get_setting("trading_mode")
        st.session_state.trading_mode = saved_mode if saved_mode in ["virtual", "real"] else "virtual"
        logger.info(f"Loaded trading mode from DB: {st.session_state.trading_mode}")

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🎛️ Trading Controls")

        # --- Mode selector ---
        mode_options = ["Virtual", "Real"]
        selected_mode_index = 0 if st.session_state.trading_mode == "virtual" else 1
        selected_mode = st.selectbox("Trading Mode", mode_options, index=selected_mode_index)

        confirm_real = False
        if selected_mode.lower() != st.session_state.trading_mode:
            if selected_mode.lower() == "real":
                confirm_real = st.checkbox("Confirm: I understand this enables LIVE trading on Bybit")
                if not confirm_real:
                    st.warning("Must confirm to switch to real mode.")
            if selected_mode.lower() != "real" or confirm_real:
                st.session_state.trading_mode = selected_mode.lower()
                db_manager.save_setting("trading_mode", st.session_state.trading_mode)
                st.session_state.wallet_cache.clear()
                logger.info(f"Switched to {st.session_state.trading_mode} mode, cleared cache")

                if st.session_state.trading_mode == "real":
                    initialize_bybit()
                    if st.session_state.bybit_client and st.session_state.bybit_client.is_connected():
                        st.session_state.engine.sync_real_balance()
                        if db_has_any_trade("real"):
                            st.session_state.engine.sync_real_trades()
                            logger.info("Real trades synced after mode switch (DB already had trades).")
                        else:
                            logger.info("Skipped real trade sync: no existing real trades in DB yet.")
                            st.info("Live mode enabled. Trade sync will start after the first real trade exists in the database.")
                st.rerun()

        # --- Engine & API status ---
        engine_status = "🟢 Online" if st.session_state.engine_initialized else "🔴 Offline"
        st.markdown(f"**Engine Status:** {engine_status}")
        mode_color = "🟢" if st.session_state.trading_mode == "virtual" else "🟡"
        st.markdown(f"**Trading Mode:** {mode_color} {st.session_state.trading_mode.title()}")

        initialize_bybit()
        api_status = "✅ Connected" if st.session_state.bybit_client and st.session_state.bybit_client.is_connected() else "❌ Disconnected"
        st.markdown(f"**API Status:** {api_status}")

        # --- License Info ---
        st.divider()
        st.markdown(f"**License Status:** {'✅ Valid' if license_result['valid'] else '❌ Invalid'}")
        st.markdown(f"**License Tier:** {license_result.get('tier', 'N/A')}")
        st.markdown(f"**Expires:** {license_result.get('formatted_expiration_date', 'N/A')}")

        st.divider()

        # --- Page Navigation ---
        st.markdown("### 📂 Pages")
        pages = {
            "📊 Dashboard": "dashboard",
            "🎯 Signals": "signals",
            "📈 Trades": "trades",
            "📊 Performance": "performance",
            "⚙️ Settings": "settings"
        }

        # Use radio buttons for user-friendly navigation
        if "active_page" not in st.session_state:
            st.session_state.active_page = "📊 Dashboard"

        selected_page = st.radio("Navigate", list(pages.keys()), index=list(pages.keys()).index(st.session_state.active_page))
        st.session_state.active_page = selected_page

        # --- Wallet Balance ---
        st.divider()
        balance = get_wallet_balance()
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

        # --- Emergency Stop ---
        st.divider()
        if st.button("🛑 Emergency Stop"):
            st.session_state.wallet_cache.clear()
            if "automated_trader" in st.session_state:
                st.session_state.automated_trader.stop()
            st.success("All automated trading stopped and cache cleared")
            logger.info("Emergency stop triggered, cache cleared")

    # --- Main content based on selected page ---
    try:
        if st.session_state.active_page == "📊 Dashboard":
            from pages.dashboard import main as dashboard_main
            dashboard_main()
        elif st.session_state.active_page == "🎯 Signals":
            from pages.signals import main as signals_main
            signals_main()
        elif st.session_state.active_page == "📈 Trades":
            from pages.trades import main as trades_main
            trades_main()
        elif st.session_state.active_page == "📊 Performance":
            from pages.performance import main as performance_main
            performance_main()
        elif st.session_state.active_page == "⚙️ Settings":
            from pages.settings import main as settings_main
            settings_main()
    except Exception as e:
        st.error(f"Error loading page: {e}")
        logger.error(f"Page load error: {e}", exc_info=True)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#888;'>AlgoTrader Pro v1.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()