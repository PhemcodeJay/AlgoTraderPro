from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from datetime import datetime
from db import db_manager
from logging_config import get_logger
from bybit_client import BybitClient
import os
import requests
import platform
import socket

# URL of your Render license server
LICENSE_SERVER_URL = os.getenv("RENDER_SERVER_URL", "https://your-render-server.onrender.com")

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
if "license_valid" not in st.session_state:
    st.session_state.license_valid = False
if "license_tier" not in st.session_state:
    st.session_state.license_tier = None
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
    try:
        db = st.session_state.engine.db if st.session_state.engine else None
        if not db:
            return False

        try: return (db.get_trade_count(mode) or 0) > 0
        except Exception: pass
        try: return (db.count_trades(mode) or 0) > 0
        except Exception: pass
        try: return bool(db.has_trades(mode))
        except Exception: pass
        try: return db.get_last_trade(mode) is not None
        except Exception: pass
        try: return bool(db.get_trades(mode=mode, limit=1))
        except Exception: pass
        try: return bool(db.get_trades(mode))
        except Exception: pass

        return False
    except Exception as e:
        logger.error(f"db_has_any_trade check failed: {e}", exc_info=True)
        return False

# --- Fetch wallet balance ---
def get_wallet_balance() -> dict:
    mode = st.session_state.trading_mode
    default_virtual = {"capital": 100.0, "available": 100.0, "used": 0.0}
    default_real = {"capital": 0.0, "available": 0.0, "used": 0.0}

    if mode in st.session_state.wallet_cache:
        return st.session_state.wallet_cache[mode]

    balance_data = default_virtual if mode == "virtual" else default_real
    try:
        if mode == "virtual":
            wallet = st.session_state.engine.db.get_wallet_balance("virtual") if st.session_state.engine else None
            if wallet:
                balance_data = {
                    "capital": getattr(wallet, "capital", default_virtual["capital"]),
                    "available": getattr(wallet, "available", default_virtual["available"]),
                    "used": getattr(wallet, "used", default_virtual["used"])
                }
        else:
            initialize_bybit()
            client = st.session_state.bybit_client
            if client and client.is_connected():
                st.session_state.engine.sync_real_balance()
                wallet = st.session_state.engine.db.get_wallet_balance("real")
                if wallet:
                    balance_data = {
                        "capital": getattr(wallet, "capital", default_real["capital"]),
                        "available": getattr(wallet, "available", default_real["available"]),
                        "used": getattr(wallet, "used", default_real["used"])
                    }
            else:
                wallet = st.session_state.engine.db.get_wallet_balance("real") if st.session_state.engine else None
                if wallet:
                    balance_data = {
                        "capital": getattr(wallet, "capital", default_real["capital"]),
                        "available": getattr(wallet, "available", default_real["available"]),
                        "used": getattr(wallet, "used", default_real["used"])
                    }
    except Exception as e:
        logger.error(f"Error fetching {mode} wallet: {e}", exc_info=True)
        balance_data = default_virtual if mode == "virtual" else default_real

    st.session_state.wallet_cache[mode] = balance_data
    if mode == "real" and balance_data["available"] <= 0:
        st.warning("Real available balance is low or zero. Deposit funds on Bybit.")
    return balance_data

# --- License Validation ---
def validate_license_key(key: str) -> dict:
    """Validate a license key by calling the Render license server."""
    try:
        payload = {
            "license_key": key,
            "hostname": socket.gethostname(),  # Works on all platforms
            "mac": None
        }
        resp = requests.post(f"{LICENSE_SERVER_URL}/validate", json=payload, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return {"valid": False, "message": f"Server returned {resp.status_code}", "tier": None}
    except Exception as e:
        return {"valid": False, "message": f"License validation failed: {e}", "tier": None}

def check_license():
    """Check if a valid license exists in the database or session state."""
    if st.session_state.license_valid and st.session_state.license_key:
        # Re-validate the stored license key to ensure it's still valid
        result = validate_license_key(st.session_state.license_key)
        if result.get("valid"):
            st.session_state.license_valid = True
            st.session_state.license_tier = result.get("tier", "basic")
            return True
        else:
            st.session_state.license_valid = False
            st.session_state.license_tier = None
            st.session_state.license_key = None
            st.session_state.engine.db.save_setting("license_key", None)
            logger.error(f"Stored license invalid: {result.get('message')}")
            return False

    # Try to load license key from database
    saved_license = st.session_state.engine.db.get_setting("license_key") if st.session_state.engine else None
    if saved_license:
        result = validate_license_key(saved_license)
        if result.get("valid"):
            st.session_state.license_valid = True
            st.session_state.license_tier = result.get("tier", "basic")
            st.session_state.license_key = saved_license
            logger.info(f"Loaded valid license from DB: {saved_license}")
            return True
        else:
            st.session_state.engine.db.save_setting("license_key", None)
            logger.error(f"DB license invalid: {result.get('message')}")
            return False
    return False

def license_input_form():
    """Display license input form and handle validation."""
    st.markdown("### 🔐 License Required")
    st.info("Please enter a valid license key to access AlgoTrader Pro.")
    license_key = st.text_input("Enter your License Key", type="password", key="license_input")
    if st.button("Validate License"):
        if license_key:
            result = validate_license_key(license_key)
            if result.get("valid"):
                st.session_state.license_valid = True
                st.session_state.license_tier = result.get("tier", "basic")
                st.session_state.license_key = license_key
                st.session_state.engine.db.save_setting("license_key", license_key)
                st.success(f"License validated successfully! Tier: {st.session_state.license_tier}")
                logger.info(f"License validated: {license_key}, Tier: {st.session_state.license_tier}")
                st.rerun()
            else:
                st.error(f"License invalid: {result.get('message')}")
                logger.error(f"License validation failed: {result.get('message')}")
        else:
            st.error("Please enter a license key.")
            logger.warning("License key input was empty")

def main():
    # Initialize engine
    if not initialize_engine():
        st.error("Application cannot start without a trading engine.")
        st.stop()

    # Check license
    if not check_license():
        license_input_form()
        st.stop()

    # Load saved trading mode from DB
    if "trading_mode" not in st.session_state or st.session_state.trading_mode is None:
        saved_mode = st.session_state.engine.db.get_setting("trading_mode")
        st.session_state.trading_mode = saved_mode if saved_mode in ["virtual", "real"] else "virtual"

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")
        st.markdown(f"**License Tier:** {st.session_state.license_tier or 'None'}")

        # Trading mode selector
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
                st.session_state.engine.db.save_setting("trading_mode", st.session_state.trading_mode)
                st.session_state.wallet_cache.clear()
                if st.session_state.trading_mode == "real":
                    initialize_bybit()
                    if st.session_state.bybit_client and st.session_state.bybit_client.is_connected():
                        st.session_state.engine.sync_real_balance()
                        if db_has_any_trade("real"):
                            st.session_state.engine.sync_real_trades()
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

        # --- Page Navigation (buttons only) ---
        page_buttons = ["📊 Dashboard", "🎯 Signals", "📈 Trades", "📊 Performance", "⚙️ Settings"]
        for page_name in page_buttons:
            if st.button(page_name, disabled=not st.session_state.license_valid):
                st.switch_page(page_name)

        st.divider()

        # Wallet Balance
        balance = get_wallet_balance()
        capital_val = balance["capital"]
        available_val = max(balance["available"], 0.0)
        used_val = capital_val - available_val
        if abs(used_val) < 0.01:
            used_val = 0.0

        if st.session_state.trading_mode == "virtual":
            st.metric("💻 Virtual Capital", f"${capital_val:.2f}")
            st.metric("💻 Virtual Available", f"${available_val:.2f}")
            st.metric("💻 Virtual Used", f"${used_val:.2f}")
        else:
            st.metric("🏦 Real Capital", f"${capital_val:.2f}")
            st.metric("🏦 Real Available", f"${available_val:.2f}")
            st.metric("🏦 Real Used Margin", f"${used_val:.2f}")

        # Last updated
        st.markdown(f"<small style='color:#888;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>", unsafe_allow_html=True)

        # Emergency Stop
        if st.button("🛑 Emergency Stop"):
            st.session_state.wallet_cache.clear()
            if "automated_trader" in st.session_state:
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
    st.markdown(f"<div style='text-align:center;color:#888;'>AlgoTrader Pro v1.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()