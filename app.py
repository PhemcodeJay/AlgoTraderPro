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
if "has_real_trades" not in st.session_state:
    st.session_state.has_real_trades = None  # Cache for db_has_any_trade

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
    if st.session_state.bybit_client is None or not st.session_state.bybit_client.is_connected():
        try:
            st.session_state.bybit_client = BybitClient()
            if st.session_state.bybit_client._test_connection():
                logger.info("Bybit client connected successfully")
                return True
            else:
                st.warning("Bybit client connection failed. Check API keys in .env file.")
                logger.error("Bybit client connection test failed")
                st.session_state.bybit_client = None
                return False
        except Exception as e:
            st.error(f"Failed to initialize Bybit client: {e}")
            logger.error(f"Bybit client initialization failed: {e}", exc_info=True)
            st.session_state.bybit_client = None
            return False
    return True

# --- Helper: does DB already have any REAL trades? ---
def db_has_any_trade(mode: str = "real") -> bool:
    """
    Return True if the DB contains at least one trade for the given mode.
    Uses cached result if available, otherwise queries DB.
    """
    if mode == "real" and st.session_state.has_real_trades is not None:
        logger.debug(f"Using cached has_real_trades: {st.session_state.has_real_trades}")
        return st.session_state.has_real_trades

    try:
        db = st.session_state.engine.db if st.session_state.engine else None
        if not db:
            logger.warning("No database access for trade check")
            return False

        trades = db.get_trades(mode=mode, limit=1)
        has_trades = bool(trades)
        if mode == "real":
            st.session_state.has_real_trades = has_trades
            logger.info(f"Cached has_real_trades: {has_trades}")
        return has_trades
    except Exception as e:
        logger.error(f"db_has_any_trade check failed for {mode}: {e}", exc_info=True)
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
        logger.debug(f"Returning cached {mode} balance: {st.session_state.wallet_cache[mode]}")
        return st.session_state.wallet_cache[mode]

    balance_data = default_virtual if mode == "virtual" else default_real
    try:
        if not st.session_state.engine:
            logger.warning("Engine not initialized for balance fetch")
            return balance_data

        if mode == "virtual":
            wallet = st.session_state.engine.db.get_wallet_balance("virtual")
            if wallet:
                balance_data = {
                    "capital": getattr(wallet, "capital", default_virtual["capital"]),
                    "available": getattr(wallet, "available", default_virtual["available"]),
                    "used": getattr(wallet, "used", default_virtual["used"])
                }
                logger.info(f"Fetched virtual wallet balance: {balance_data}")
        else:  # real mode
            if initialize_bybit() and st.session_state.bybit_client.is_connected():
                # Sync real balance with Bybit
                if st.session_state.engine.sync_real_balance():
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
                    logger.warning("Real balance sync failed")
                    st.error("❌ Real balance sync failed. Using last known balance.")
                    wallet = st.session_state.engine.db.get_wallet_balance("real")
                    if wallet:
                        balance_data = {
                            "capital": getattr(wallet, "capital", default_real["capital"]),
                            "available": getattr(wallet, "available", default_real["available"]),
                            "used": getattr(wallet, "used", default_real["used"])
                        }
                        logger.info(f"Using DB real balance: {balance_data}")
            else:
                logger.warning("Bybit client not connected for real balance")
                st.warning("Bybit API not connected. Check API keys in .env file.")
                wallet = st.session_state.engine.db.get_wallet_balance("real")
                if wallet:
                    balance_data = {
                        "capital": getattr(wallet, "capital", default_real["capital"]),
                        "available": getattr(wallet, "available", default_real["available"]),
                        "used": getattr(wallet, "used", default_real["used"])
                    }
                    logger.info(f"Using DB real balance (API disconnected): {balance_data}")

    except Exception as e:
        logger.error(f"Error fetching {mode} wallet: {e}", exc_info=True)
        st.error(f"Error fetching {mode} balance: {e}")
        balance_data = default_virtual if mode == "virtual" else default_real

    # Cache balance for this session
    st.session_state.wallet_cache[mode] = balance_data
    logger.info(f"Cached {mode} balance: {balance_data}")

    # Conditional messaging for real balance
    if mode == "real" and balance_data["available"] <= 0:
        st.warning("Real available balance is low or zero. Deposit funds on Bybit.")

    return balance_data

def main():
    # Initialize engine
    if not initialize_engine():
        st.stop()

    # Load saved trading mode from DB
    if "trading_mode" not in st.session_state or st.session_state.trading_mode is None:
        try:
            saved_mode = st.session_state.engine.db.get_setting("trading_mode")
            st.session_state.trading_mode = saved_mode if saved_mode in ["virtual", "real"] else "virtual"
            logger.info(f"Loaded trading mode from DB: {st.session_state.trading_mode}")
        except Exception as e:
            logger.error(f"Failed to load trading mode from DB: {e}", exc_info=True)
            st.session_state.trading_mode = "virtual"

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🎛️ Trading Controls")

        # --- Mode selector ---
        mode_options = ["Virtual", "Real"]
        selected_mode_index = 0 if st.session_state.trading_mode == "virtual" else 1
        selected_mode = st.selectbox(
            "Trading Mode",
            mode_options,
            index=selected_mode_index,
            help="Switch between virtual (simulated) and real (live) trading. Real mode requires a connected Bybit account."
        )

        if selected_mode.lower() != st.session_state.trading_mode:
            if selected_mode.lower() == "real":
                st.warning("⚠️ Real mode enables LIVE trading on Bybit. Ensure sufficient funds and valid API keys.")
            try:
                st.session_state.trading_mode = selected_mode.lower()
                st.session_state.engine.db.save_setting("trading_mode", st.session_state.trading_mode)
                st.session_state.wallet_cache.clear()
                st.session_state.has_real_trades = None  # Reset trade cache
                logger.info(f"Switched to {st.session_state.trading_mode} mode, cleared cache")
                
                if st.session_state.trading_mode == "real":
                    if initialize_bybit() and st.session_state.bybit_client.is_connected():
                        st.session_state.engine.sync_real_balance()
                        if db_has_any_trade("real"):
                            st.session_state.engine.sync_real_trades()
                            logger.info("Real trades synced after mode switch (DB has trades)")
                            st.success("✅ Switched to real mode. Trades and balance synced.")
                        else:
                            logger.info("Skipped real trade sync: no existing real trades in DB")
                            st.success("✅ Switched to real mode. Ready for live trading.")
                    else:
                        st.error("❌ Failed to connect to Bybit API. Reverting to virtual mode.")
                        st.session_state.trading_mode = "virtual"
                        st.session_state.engine.db.save_setting("trading_mode", "virtual")
                        st.session_state.wallet_cache.clear()
                        st.session_state.has_real_trades = None
                else:
                    st.success(f"✅ Switched to {st.session_state.trading_mode} mode.")
                
                st.rerun()
            except Exception as e:
                st.error(f"Failed to switch mode: {e}")
                logger.error(f"Mode switch to {selected_mode.lower()} failed: {e}", exc_info=True)
                st.session_state.trading_mode = "virtual"
                st.session_state.engine.db.save_setting("trading_mode", "virtual")
                st.rerun()

        # --- Engine & API status ---
        engine_status = "🟢 Online" if st.session_state.engine_initialized else "🔴 Offline"
        st.markdown(f"**Engine Status:** {engine_status}")
        mode_color = "🟢" if st.session_state.trading_mode == "virtual" else "🟡"
        st.markdown(f"**Trading Mode:** {mode_color} {st.session_state.trading_mode.title()}")

        api_status = "✅ Connected" if st.session_state.bybit_client and st.session_state.bybit_client.is_connected() else "❌ Disconnected"
        st.markdown(f"**API Status:** {api_status}")

        st.divider()

        # --- Lower Section: Page Navigation ---
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
        if selected_page != st.session_state.active_page:
            st.session_state.active_page = selected_page
            page_file = pages[selected_page]
            if page_file != "dashboard":
                try:
                    st.switch_page(f"pages/{page_file}.py")
                except Exception as e:
                    st.error(f"Failed to navigate to {page_file}: {e}")
                    logger.error(f"Navigation to {page_file} failed: {e}", exc_info=True)

        # --- Wallet Balance ---
        st.divider()
        balance = get_wallet_balance()
        current_mode = st.session_state.trading_mode
        capital_val = balance["capital"]
        available_val = max(balance["available"], 0.0)
        used_val = max(capital_val - available_val, 0.0)
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
            st.session_state.has_real_trades = None
            if "automated_trader" in st.session_state:
                try:
                    import asyncio
                    asyncio.run(st.session_state.automated_trader.stop())
                    logger.info("Automated trader stopped")
                except Exception as e:
                    logger.error(f"Failed to stop automated trader: {e}", exc_info=True)
            st.success("All automated trading stopped and cache cleared")
            logger.info("Emergency stop triggered, cache cleared")
            st.rerun()

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