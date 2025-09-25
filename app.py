from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import asyncio
from datetime import datetime, timezone
from db import db_manager
from logging_config import get_logger
from bybit_client import BybitClient
from trading_engine import TradingEngine
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
def initialize_session_state():
    defaults = {
        "trading_mode": "virtual",
        "engine_initialized": False,
        "wallet_cache": {},
        "engine": None,
        "trading_task": None,
        "trader_status": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Initialize trading engine ---
async def initialize_engine():
    try:
        if not st.session_state.engine_initialized:
            st.session_state.engine = create_engine()
            st.session_state.engine_initialized = True
            st.session_state.trader_status = await st.session_state.engine.trader.get_status()
            logger.info("Trading engine initialized successfully")
        return True
    except Exception as e:
        st.error(f"Failed to initialize trading engine: {e}")
        logger.error(f"Engine initialization failed: {e}", exc_info=True)
        return False

# --- Fetch wallet balance ---
def get_wallet_balance() -> dict:
    """
    Fetch wallet balance based on the selected trading mode from DB only.
    Returns a dict with capital, available, used, and updated_at.
    Always safe, fallback to defaults. No auto-sync.
    """
    mode = st.session_state.trading_mode
    default_virtual = {"capital": 100.0, "available": 100.0, "used": 0.0, "updated_at": None}
    default_real = {"capital": 0.0, "available": 0.0, "used": 0.0, "updated_at": None}

    # Check cache
    if mode in st.session_state.wallet_cache:
        logger.info(f"Returning cached {mode} balance: {st.session_state.wallet_cache[mode]}")
        return st.session_state.wallet_cache[mode]

    try:
        wallet = db_manager.get_wallet_balance(mode)
        if wallet:
            balance_data = wallet.to_dict()
            logger.info(f"Fetched {mode} wallet balance: {balance_data}")
        else:
            balance_data = default_virtual if mode == "virtual" else default_real
            logger.warning(f"No {mode} wallet found in DB, using defaults")
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
        elif balance_data["available"] == 0.0 and balance_data["capital"] == 0.0:
            st.warning("No funds available in Bybit account. Verify account balance or API permissions.")

    return balance_data

# --- Async trading cycle ---
async def run_trading_cycle():
    try:
        while st.session_state.engine_initialized:
            await asyncio.sleep(5)  # Adjust based on settings
            if st.session_state.engine:
                st.session_state.engine.run_trading_cycle()
                # Update trader status
                st.session_state.trader_status = await st.session_state.engine.trader.get_status()
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}", exc_info=True)

# --- Main App ---
def main():
    initialize_session_state()

    # Custom CSS
    st.markdown("""
    <style>
    .main-header { background: linear-gradient(135deg, #1e1e2f 0%, #2a2a4a 100%); padding:2rem; border-radius:10px; text-align:center; margin-bottom:2rem; border:2px solid #00ff88;}
    </style>
    """, unsafe_allow_html=True)

    # --- Logo Row ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=100)

    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1 style="color:#00ff88; margin:0; font-size:3rem;">🚀 AlgoTrader Pro</h1>
        <p style="color:#888; margin:0; font-size:1.2rem;">Advanced Cryptocurrency Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize engine asynchronously
    if not asyncio.run(initialize_engine()):
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
            st.rerun()

        # Status
        engine_status = "🟢 Online" if st.session_state.engine_initialized else "🔴 Offline"
        st.markdown(f"**Engine Status:** {engine_status}")
        mode_color = "🟢" if st.session_state.trading_mode == "virtual" else "🟡"
        st.markdown(f"**Trading Mode:** {mode_color} {st.session_state.trading_mode.title()}")

        # API Connection Status
        api_status = "✅ Connected" if st.session_state.engine and st.session_state.engine.client and st.session_state.engine.client.is_connected() else "❌ Disconnected"
        st.markdown(f"**API Status:** {api_status}")

        # Trader Status
        trader_status = st.session_state.trader_status
        if trader_status:
            status_icon = "🟢" if trader_status.get("is_running", False) else "🔴"
            st.markdown(f"**Trader Status:** {status_icon} {'Running' if trader_status.get('is_running', False) else 'Stopped'}")
            st.markdown(f"**Open Positions:** {trader_status.get('current_positions', 0)}/{trader_status.get('max_positions', 0)}")
            st.markdown(f"**Total PnL:** ${trader_status.get('stats', {}).get('total_pnl', 0):.2f}")

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
        last_synced = balance_data.get("updated_at")
        if last_synced:
            last_synced = datetime.fromisoformat(last_synced).strftime('%Y-%m-%d %H:%M:%S') + " UTC"
        else:
            last_synced = "Never"
        help_text = f"Last synced: {last_synced}"
        if st.session_state.trading_mode == "virtual":
            st.metric("💻 Virtual Capital", f"${balance_data['capital']:.2f}")
            st.metric("💻 Virtual Available", f"${balance_data['available']:.2f}")
            st.metric("💻 Virtual Used", f"${balance_data['used']:.2f}")
        else:
            st.metric("🏦 Real Capital", f"${balance_data['capital']:.2f}", help=help_text)
            st.metric("🏦 Real Available", f"${balance_data['available']:.2f}", help=help_text)
            st.metric("🏦 Real Used Margin", f"${balance_data['used']:.2f}", help=help_text)

        # Sync button for real mode
        if st.session_state.trading_mode == "real":
            if st.button("🔄 Sync Real Balance"):
                if st.session_state.engine and st.session_state.engine.client:
                    sync_real_wallet_balance(st.session_state.engine.client)
                    st.session_state.wallet_cache.clear()
                    st.rerun()
                else:
                    st.error("Engine not initialized")

        # Last updated
        st.markdown(
            f"<small style='color:#888;'>Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</small>",
            unsafe_allow_html=True
        )

        # Trading Controls
        async def handle_start_trading():
            if st.session_state.engine and st.session_state.engine.trader:
                success = await st.session_state.engine.trader.start(st)
                if success:
                    st.success("Automated trading started")
                    logger.info("Automated trading started via UI")
                    st.session_state.trader_status = await st.session_state.engine.trader.get_status()
                else:
                    st.error("Failed to start trading. Check logs for details.")
                    logger.error("Failed to start automated trading")
            else:
                st.error("Trading engine or trader not initialized")
                logger.error("Start trading attempted but engine/trader not initialized")

        async def handle_stop_trading():
            if st.session_state.engine and st.session_state.engine.trader:
                success = await st.session_state.engine.trader.stop()
                if success:
                    st.success("Automated trading stopped")
                    logger.info("Automated trading stopped via UI")
                    st.session_state.trader_status = await st.session_state.engine.trader.get_status()
                else:
                    st.error("Failed to stop trading. Check logs for details.")
                    logger.error("Failed to stop automated trading")
            else:
                st.error("Trading engine or trader not initialized")
                logger.error("Stop trading attempted but engine/trader not initialized")

        async def handle_emergency_stop():
            if st.session_state.engine:
                success = await st.session_state.engine.emergency_stop("User-initiated emergency stop")
                if success:
                    st.session_state.wallet_cache.clear()
                    st.success("All automated trading stopped and cache cleared")
                    logger.info("Emergency stop triggered successfully, cache cleared")
                    st.session_state.trader_status = await st.session_state.engine.trader.get_status()
                else:
                    st.error("Failed to execute emergency stop. Check logs for details.")
                    logger.error("Emergency stop execution failed")
            else:
                st.error("Trading engine not initialized")
                logger.error("Emergency stop attempted but engine not initialized")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Trading"):
                asyncio.run(handle_start_trading())
        with col2:
            if st.button("⏸️ Stop Trading"):
                asyncio.run(handle_stop_trading())
        if st.button("🛑 Emergency Stop"):
            asyncio.run(handle_emergency_stop())

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
        f"<div style='text-align:center;color:#888;'>AlgoTrader Pro v1.0 | Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()