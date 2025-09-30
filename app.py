from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from datetime import datetime
from db import db_manager
from logging_config import get_logger
from bybit_client import BybitClient
from check_license import validate_license  # Import validate_license

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
                # Sync real balance with Bybit (safe to do regardless of trades)
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

# --- Validate License ---
def validate_user_license():
    """
    Validate the user's license key, prompting for a new key if none exists or if invalid.
    Returns True if the license is valid, False otherwise.
    """
    # Check for saved license key in the database
    saved_license_key = db_manager.get_setting("license_key")
    if saved_license_key and st.session_state.license_key != saved_license_key:
        st.session_state.license_key = saved_license_key

    # If no license key in session state, prompt for one
    if not st.session_state.license_key:
        st.warning("Please enter a valid license key to access AlgoTrader Pro.")
        with st.form("license_form"):
            license_key_input = st.text_input("License Key", placeholder="e.g., 123e4567-e89b-12d3-a456-426614174000")
            submitted = st.form_submit_button("Validate License")
            if submitted:
                if not license_key_input:
                    st.error("Please enter a license key.")
                    return False
                license_result = db_manager.validate_license(license_key_input)
                if license_result["valid"]:
                    st.session_state.license_valid = True
                    st.session_state.license_key = license_key_input
                    db_manager.save_setting("license_key", license_key_input)
                    st.success(f"License validated successfully! Tier: {license_result['tier']}, Expires: {license_result['formatted_expiration_date']}")
                    logger.info(f"License validated: {license_key_input}, Tier: {license_result['tier']}")
                    st.rerun()
                else:
                    st.error(f"Invalid license key: {license_result['message']}")
                    logger.error(f"License validation failed: {license_result['message']}")
                    return False
        return False

    # Validate existing license key
    license_result = db_manager.validate_license(st.session_state.license_key)
    if license_result["valid"]:
        st.session_state.license_valid = True
        logger.info(f"License {st.session_state.license_key} is valid: Tier={license_result['tier']}")
        return True
    else:
        st.error(f"License is invalid: {license_result['message']}. Please enter a new license key.")
        logger.error(f"License validation failed for {st.session_state.license_key}: {license_result['message']}")
        with st.form("license_form"):
            license_key_input = st.text_input("New License Key", placeholder="e.g., 123e4567-e89b-12d3-a456-426614174000")
            submitted = st.form_submit_button("Validate New License")
            if submitted:
                if not license_key_input:
                    st.error("Please enter a license key.")
                    return False
                new_license_result = db_manager.validate_license(license_key_input)
                if new_license_result["valid"]:
                    st.session_state.license_valid = True
                    st.session_state.license_key = license_key_input
                    db_manager.save_setting("license_key", license_key_input)
                    st.success(f"License validated successfully! Tier: {new_license_result['tier']}, Expires: {new_license_result['formatted_expiration_date']}")
                    logger.info(f"New license validated: {license_key_input}, Tier: {new_license_result['tier']}")
                    st.rerun()
                else:
                    st.error(f"Invalid license key: {new_license_result['message']}")
                    logger.error(f"New license validation failed: {new_license_result['message']}")
                    return False
        return False

def main():
    # Validate license before proceeding
    if not validate_user_license():
        return  # Stop execution if license is invalid

    initialize_engine()
    # Load saved trading mode from DB
    if "trading_mode" not in st.session_state or st.session_state.trading_mode is None:
        saved_mode = st.session_state.engine.db.get_setting("trading_mode")
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
                st.session_state.engine.db.save_setting("trading_mode", st.session_state.trading_mode)
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
        license_info = db_manager.validate_license(st.session_state.license_key)
        st.markdown(f"**License Status:** {'✅ Valid' if license_info['valid'] else '❌ Invalid'}")
        st.markdown(f"**License Tier:** {license_info.get('tier', 'N/A')}")
        st.markdown(f"**Expires:** {license_info.get('formatted_expiration_date', 'N/A')}")

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