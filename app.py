from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import asyncio
from datetime import datetime
from db import db_manager
from logging_config import get_logger
from bybit_client import BybitClient
from engine import TradingEngine
from utils import sync_real_wallet_balance
from engine import create_engine

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
        if st.session_state.bybit_client.is_connected():
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
            wallet = db_manager.get_wallet_balance("virtual")
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
                sync_real_wallet_balance(client)
                wallet = db_manager.get_wallet_balance("real")
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
                wallet = db_manager.get_wallet_balance("real")
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
        if balance_data["available"] == 0.0 and balance_data["capital"] > 0.0:
            st.info("Real available balance is $0.00. Funds may be in use (e.g., open positions).")
        elif balance_data["available"] == 0.0 and balance_data["capital"] == 0.0 and st.session_state.bybit_client and st.session_state.bybit_client.is_connected():
            st.warning("No funds available in Bybit account. Verify account balance or API permissions.")

    return balance_data

# --- Main App ---
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header { background: linear-gradient(135deg, #1e1e2f 0%, #2a2a4a 100%); padding:2rem; border-radius:10px; text-align:center; margin-bottom:2rem; border:2px solid #00ff88;}
    </style>
    """, unsafe_allow_html=True)

    # --- Logo Row ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=150)

    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1 style="color:#00ff88; margin:0; font-size:3rem;">🚀 AlgoTrader Pro</h1>
        <p style="color:#888; margin:0; font-size:1.2rem;">Advanced Cryptocurrency Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)

    if not initialize_engine():
        st.stop()

    # Load saved trading mode from DB
    if "trading_mode" not in st.session_state or st.session_state.trading_mode is None:
        saved_mode = db_manager.get_setting("trading_mode")
        st.session_state.trading_mode = saved_mode if saved_mode in ["virtual", "real"] else "virtual"
        logger.info(f"Loaded trading mode from DB: {st.session_state.trading_mode}")

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")

        # Mode selector
        mode_options = ["Virtual", "Real"]
        selected_mode = st.selectbox(
            "Trading Mode",
            mode_options,
            index=0 if st.session_state.trading_mode == "virtual" else 1
        )

        # Update session state and persist to DB if changed
        if selected_mode.lower() != st.session_state.trading_mode:
            st.session_state.trading_mode = selected_mode.lower()
            db_manager.save_setting("trading_mode", st.session_state.trading_mode)
            st.session_state.wallet_cache.clear()
            logger.info(f"Switched to {st.session_state.trading_mode} mode, cleared cache")

            # Sync real balance if switching to real mode
            if st.session_state.trading_mode == "real":
                initialize_bybit()
                if st.session_state.bybit_client and st.session_state.bybit_client.is_connected():
                    sync_real_wallet_balance(st.session_state.bybit_client)
                    logger.info("Real balance synced after mode switch")
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
        balance_data = get_wallet_balance()
        if st.session_state.trading_mode == "virtual":
            st.metric("💻 Virtual Capital", f"${balance_data['capital']:.2f}")
            st.metric("💻 Virtual Available", f"${balance_data['available']:.2f}")
            st.metric("💻 Virtual Used", f"${balance_data['used']:.2f}")
        else:
            st.metric("🏦 Real Capital", f"${balance_data['capital']:.2f}")
            st.metric("🏦 Real Available", f"${balance_data['available']:.2f}")
            st.metric("🏦 Real Used Margin", f"${balance_data['used']:.2f}")

        # Last updated
        st.markdown(
            f"<small style='color:#888;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>",
            unsafe_allow_html=True
        )

        # Emergency Stop
        if st.button("🛑 Emergency Stop"):
            if st.session_state.engine:
                success = st.session_state.engine.emergency_stop("User-initiated emergency stop")
                if success:
                    st.session_state.wallet_cache.clear()
                    st.success("All automated trading stopped and cache cleared")
                    logger.info("Emergency stop triggered successfully, cache cleared")
                else:
                    st.error("Failed to execute emergency stop. Check logs for details.")
                    logger.error("Emergency stop execution failed")
            else:
                st.error("Trading engine not initialized")
                logger.error("Emergency stop attempted but engine not initialized")

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
    engine = create_engine()
    engine.run_trading_cycle()
