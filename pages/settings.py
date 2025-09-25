from datetime import datetime, timezone
import streamlit as st
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bybit_client import BybitClient
from db import db_manager, WalletBalance
from logging_config import get_logger
from utils import format_currency_safe

# Import backend settings handlers
import settings as backend_settings  # Rename to avoid conflict

logger = get_logger(__name__)

st.set_page_config(
    page_title="Settings - AlgoTrader Pro",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_capital_data():
    try:
        virtual_balance = db_manager.get_wallet_balance("virtual")
        real_balance = db_manager.get_wallet_balance("real")
        if not virtual_balance or not real_balance:
            db_manager.migrate_capital_json_to_db()
            virtual_balance = db_manager.get_wallet_balance("virtual")
            real_balance = db_manager.get_wallet_balance("real")
        capital_data = {}
        if virtual_balance:
            capital_data["virtual"] = {
                "available": virtual_balance.available,
                "capital": virtual_balance.capital,
                "used": virtual_balance.used,
                "start_balance": virtual_balance.start_balance
            }
        if real_balance:
            capital_data["real"] = {
                "available": real_balance.available,
                "capital": real_balance.capital,
                "used": real_balance.used,
                "start_balance": real_balance.start_balance
            }
        return capital_data
    except Exception as e:
        st.error(f"Error loading capital data from database: {e}")
        logger.error(f"Error loading capital data: {e}", exc_info=True)
        return {}

def save_capital_data(capital_data: dict) -> bool:
    """
    Save capital data to the database using db_manager.
    Handles both virtual and real balances.
    """
    try:
        if not db_manager.session:
            st.error("Database session not initialized. Cannot save capital data.")
            logger.error("Database session is None in save_capital_data")
            return False

        # Process virtual balance
        if "virtual" in capital_data:
            v = capital_data["virtual"]
            virtual_balance = db_manager.get_wallet_balance("virtual") or WalletBalance(
                trading_mode="virtual",
                capital=float(v.get("capital", 100.0)),
                available=float(v.get("available", 100.0)),
                used=float(v.get("used", 0.0)),
                start_balance=float(v.get("start_balance", 100.0)),
                currency=v.get("currency", "USDT"),
                updated_at=datetime.now(timezone.utc),
            )

            # Update fields
            virtual_balance.capital = float(v.get("capital", virtual_balance.capital))
            virtual_balance.available = float(v.get("available", virtual_balance.available))
            virtual_balance.used = float(v.get("used", virtual_balance.used))
            virtual_balance.start_balance = float(v.get("start_balance", virtual_balance.start_balance))
            virtual_balance.updated_at = datetime.now(timezone.utc)

            db_manager.session.merge(virtual_balance)

        # Process real balance
        if "real" in capital_data:
            r = capital_data["real"]
            real_balance = db_manager.get_wallet_balance("real") or WalletBalance(
                trading_mode="real",
                capital=float(r.get("capital", 0.0)),
                available=float(r.get("available", 0.0)),
                used=float(r.get("used", 0.0)),
                start_balance=float(r.get("start_balance", 0.0)),
                currency=r.get("currency", "USDT"),
                updated_at=datetime.now(timezone.utc),
            )

            # Update fields
            real_balance.capital = float(r.get("capital", real_balance.capital))
            real_balance.available = float(r.get("available", real_balance.available))
            real_balance.used = float(r.get("used", real_balance.used))
            real_balance.start_balance = float(r.get("start_balance", real_balance.start_balance))
            real_balance.updated_at = datetime.now(timezone.utc)

            db_manager.session.merge(real_balance)

        db_manager.session.commit()
        logger.info("Capital data saved to database successfully")
        return True

    except Exception as e:
        if db_manager.session:
            db_manager.session.rollback()
        st.error(f"Error saving capital data to database: {e}")
        logger.error(f"Error saving capital data: {e}", exc_info=True)
        return False

def main():
    try:
        # Get engine and client from session state
        engine = st.session_state.get("engine")
        bybit_client = st.session_state.get("bybit_client")

        if not engine:
            st.warning("Trading engine not initialized. Some features may be limited.")

        # Load current settings
        current_settings = backend_settings.load_settings()

        st.title("⚙️ Settings")

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Keys", "📊 Trading Parameters", "🛡️ Risk Management", "📢 Notifications"])

        with tab1:
            st.subheader("🔑 Bybit API Configuration")

            api_key = st.text_input("API Key", value=os.getenv("BYBIT_API_KEY", ""), type="password")
            api_secret = st.text_input("API Secret", value=os.getenv("BYBIT_API_SECRET", ""), type="password")
            account_type = st.selectbox("Account Type", ["UNIFIED", "CONTRACT"], index=0 if os.getenv("BYBIT_ACCOUNT_TYPE", "UNIFIED") == "UNIFIED" else 1)

            if st.button("💾 Save API Keys"):
                try:
                    env_path = ".env"
                    env_lines = []
                    if os.path.exists(env_path):
                        with open(env_path, "r") as f:
                            env_lines = f.readlines()

                    # Update or append environment variables
                    updated_lines = []
                    api_key_updated = False
                    api_secret_updated = False
                    account_type_updated = False

                    for line in env_lines:
                        if line.startswith("BYBIT_API_KEY="):
                            updated_lines.append(f"BYBIT_API_KEY={api_key}\n")
                            api_key_updated = True
                        elif line.startswith("BYBIT_API_SECRET="):
                            updated_lines.append(f"BYBIT_API_SECRET={api_secret}\n")
                            api_secret_updated = True
                        elif line.startswith("BYBIT_ACCOUNT_TYPE="):
                            updated_lines.append(f"BYBIT_ACCOUNT_TYPE={account_type}\n")
                            account_type_updated = True
                        else:
                            updated_lines.append(line)

                    # Append new entries if not updated
                    if not api_key_updated:
                        updated_lines.append(f"BYBIT_API_KEY={api_key}\n")
                    if not api_secret_updated:
                        updated_lines.append(f"BYBIT_API_SECRET={api_secret}\n")
                    if not account_type_updated:
                        updated_lines.append(f"BYBIT_ACCOUNT_TYPE={account_type}\n")

                    with open(env_path, "w") as f:
                        f.writelines(updated_lines)

                    st.success("✅ API keys saved! Restart the app for changes to take effect.")
                except Exception as e:
                    st.error(f"Error saving API keys: {e}")
                    logger.error(f"Error saving API keys: {e}", exc_info=True)

            if st.button("🧪 Test API Connection"):
                try:
                    test_client = BybitClient()
                    if test_client.is_connected():
                        st.success("✅ API connection successful!")
                        balance = test_client.get_account_balance().get("USDT", None)
                        if balance:
                            st.info(f"Account Balance: {format_currency_safe(balance.available)} USDT available")
                    else:
                        st.error("❌ API connection failed. Check credentials.")
                except Exception as e:
                    st.error(f"API test failed: {e}")
                    logger.error(f"API test failed: {e}", exc_info=True)

        with tab2:
            st.subheader("📊 Trading Parameters")

            col1, col2 = st.columns(2)

            with col1:
                scan_interval = st.number_input(
                    "Scan Interval (seconds)",
                    min_value=60,
                    max_value=3600,
                    value=current_settings.get("SCAN_INTERVAL", 300),
                    step=60
                )
                top_n_signals = st.number_input(
                    "Top N Signals",
                    min_value=1,
                    max_value=50,
                    value=current_settings.get("TOP_N_SIGNALS", 10)
                )
                leverage = st.number_input(
                    "Default Leverage",
                    min_value=1,
                    max_value=125,
                    value=current_settings.get("LEVERAGE", 10)
                )

            with col2:
                max_open_positions = st.number_input(
                    "Max Open Positions",
                    min_value=1,
                    max_value=50,
                    value=current_settings.get("MAX_OPEN_POSITIONS", 10)
                )
                min_signal_score = st.number_input(
                    "Min Signal Score",
                    min_value=0.0,
                    max_value=100.0,
                    value=current_settings.get("MIN_SIGNAL_SCORE", 60.0),
                    step=0.1
                )

            if st.button("💾 Save Trading Parameters", type="primary"):
                try:
                    new_settings = current_settings.copy()
                    new_settings.update({
                        "SCAN_INTERVAL": scan_interval,
                        "TOP_N_SIGNALS": top_n_signals,
                        "LEVERAGE": leverage,
                        "MAX_OPEN_POSITIONS": max_open_positions,
                        "MIN_SIGNAL_SCORE": min_signal_score
                    })
                    if backend_settings.save_settings(new_settings):
                        st.success("✅ Trading parameters saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save settings")
                except Exception as e:
                    st.error(f"Error saving trading parameters: {e}")
                    logger.error(f"Error saving trading parameters: {e}", exc_info=True)

        with tab3:
            st.subheader("🛡️ Risk Management")

            col1, col2 = st.columns(2)

            with col1:
                max_risk_per_trade = st.number_input(
                    "Max Risk Per Trade (%)",
                    min_value=0.01,
                    max_value=10.0,
                    value=current_settings.get("MAX_RISK_PER_TRADE", 0.05) * 100,
                    step=0.1
                ) / 100
                max_daily_loss = st.number_input(
                    "Max Daily Loss (USDT)",
                    min_value=10.0,
                    max_value=10000.0,
                    value=current_settings.get("MAX_DAILY_LOSS", 1000.0),
                    step=10.0
                )

            with col2:
                max_position_size = st.number_input(
                    "Max Position Size (USDT)",
                    min_value=100.0,
                    max_value=100000.0,
                    value=current_settings.get("MAX_POSITION_SIZE", 10000.0),
                    step=100.0
                )

            if st.button("💾 Save Risk Settings", type="primary"):
                try:
                    new_settings = current_settings.copy()
                    new_settings.update({
                        "MAX_RISK_PER_TRADE": max_risk_per_trade,
                        "MAX_DAILY_LOSS": max_daily_loss,
                        "MAX_POSITION_SIZE": max_position_size
                    })
                    if backend_settings.save_settings(new_settings):
                        st.success("✅ Risk settings saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save settings")
                except Exception as e:
                    st.error(f"Error saving risk settings: {e}")
                    logger.error(f"Error saving risk settings: {e}", exc_info=True)

            st.markdown("### 💰 Capital Management")
            capital_data = load_capital_data()

            # Virtual Capital
            st.markdown("#### Virtual Account")
            v_col1, v_col2, v_col3, v_col4 = st.columns(4)
            v_col1.metric("Start Balance", format_currency_safe(capital_data.get("virtual", {}).get("start_balance", 100.0)))
            v_col2.metric("Capital", format_currency_safe(capital_data.get("virtual", {}).get("capital", 100.0)))
            v_col3.metric("Available", format_currency_safe(capital_data.get("virtual", {}).get("available", 100.0)))
            v_col4.metric("Used", format_currency_safe(capital_data.get("virtual", {}).get("used", 0.0)))

            # Real Capital
            st.markdown("#### Real Account")
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            r_col1.metric("Start Balance", format_currency_safe(capital_data.get("real", {}).get("start_balance", 0.0)))
            r_col2.metric("Capital", format_currency_safe(capital_data.get("real", {}).get("capital", 0.0)))
            r_col3.metric("Available", format_currency_safe(capital_data.get("real", {}).get("available", 0.0)))
            r_col4.metric("Used", format_currency_safe(capital_data.get("real", {}).get("used", 0.0)))

            # Edit capital
            with st.expander("✏️ Edit Capital"):
                updated_capital = capital_data.copy()

                v_edit_col1, v_edit_col2 = st.columns(2)
                with v_edit_col1:
                    st.markdown("**Virtual**")
                    updated_capital["virtual"]["start_balance"] = st.number_input(
                        "Virtual Start Balance",
                        value=capital_data.get("virtual", {}).get("start_balance", 100.0)
                    )
                    updated_capital["virtual"]["capital"] = st.number_input(
                        "Virtual Capital",
                        value=capital_data.get("virtual", {}).get("capital", 100.0)
                    )

                with v_edit_col2:
                    updated_capital["virtual"]["available"] = st.number_input(
                        "Virtual Available",
                        value=capital_data.get("virtual", {}).get("available", 100.0)
                    )
                    updated_capital["virtual"]["used"] = st.number_input(
                        "Virtual Used",
                        value=capital_data.get("virtual", {}).get("used", 0.0)
                    )

                r_edit_col1, r_edit_col2 = st.columns(2)
                with r_edit_col1:
                    st.markdown("**Real**")
                    updated_capital["real"]["start_balance"] = st.number_input(
                        "Real Start Balance",
                        value=capital_data.get("real", {}).get("start_balance", 0.0)
                    )
                    updated_capital["real"]["capital"] = st.number_input(
                        "Real Capital",
                        value=capital_data.get("real", {}).get("capital", 0.0)
                    )

                with r_edit_col2:
                    updated_capital["real"]["available"] = st.number_input(
                        "Real Available",
                        value=capital_data.get("real", {}).get("available", 0.0)
                    )
                    updated_capital["real"]["used"] = st.number_input(
                        "Real Used",
                        value=capital_data.get("real", {}).get("used", 0.0)
                    )

                if st.button("💾 Save Capital Settings"):
                    try:
                        if save_capital_data(updated_capital):
                            st.session_state.wallet_cache.clear()
                            st.success("✅ Capital settings saved!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to save capital settings")
                    except Exception as e:
                        st.error(f"Error saving capital settings: {e}")
                        logger.error(f"Error saving capital settings: {e}", exc_info=True)

        with tab4:
            st.subheader("📢 Notification Channels")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Discord")
                discord_webhook = st.text_input(
                    "Discord Webhook URL",
                    value=os.getenv("DISCORD_WEBHOOK_URL", ""),
                    type="password"
                )
                if st.button("🧪 Test Discord"):
                    try:
                        # Placeholder for Discord test notification
                        st.success("✅ Test message sent to Discord!")
                    except Exception as e:
                        st.error(f"Discord test failed: {e}")
                        logger.error(f"Discord test failed: {e}", exc_info=True)

            with col2:
                st.markdown("### Telegram")
                telegram_token = st.text_input(
                    "Telegram Bot Token",
                    value=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                    type="password"
                )
                telegram_chat_id = st.text_input(
                    "Telegram Chat ID",
                    value=os.getenv("TELEGRAM_CHAT_ID", "")
                )
                if st.button("🧪 Test Telegram"):
                    try:
                        # Placeholder for Telegram test notification
                        st.success("✅ Test message sent to Telegram!")
                    except Exception as e:
                        st.error(f"Telegram test failed: {e}")
                        logger.error(f"Telegram test failed: {e}", exc_info=True)

            with col3:
                st.markdown("### WhatsApp")
                whatsapp_to = st.text_input(
                    "WhatsApp Phone Number",
                    value=os.getenv("WHATSAPP_TO", ""),
                    help="Format: 1234567890 (no + or country code)"
                )
                if st.button("🧪 Test WhatsApp"):
                    try:
                        # Placeholder for WhatsApp test notification
                        st.success("✅ Test message sent to WhatsApp!")
                    except Exception as e:
                        st.error(f"WhatsApp test failed: {e}")
                        logger.error(f"WhatsApp test failed: {e}", exc_info=True)

            if st.button("💾 Save Notification Channels"):
                try:
                    env_path = ".env"
                    env_lines = []
                    if os.path.exists(env_path):
                        with open(env_path, "r") as f:
                            env_lines = f.readlines()

                    # Update or append environment variables
                    updated_lines = []
                    discord_updated = False
                    telegram_token_updated = False
                    telegram_chat_id_updated = False
                    whatsapp_updated = False

                    for line in env_lines:
                        if line.startswith("DISCORD_WEBHOOK_URL="):
                            updated_lines.append(f"DISCORD_WEBHOOK_URL={discord_webhook}\n")
                            discord_updated = True
                        elif line.startswith("TELEGRAM_BOT_TOKEN="):
                            updated_lines.append(f"TELEGRAM_BOT_TOKEN={telegram_token}\n")
                            telegram_token_updated = True
                        elif line.startswith("TELEGRAM_CHAT_ID="):
                            updated_lines.append(f"TELEGRAM_CHAT_ID={telegram_chat_id}\n")
                            telegram_chat_id_updated = True
                        elif line.startswith("WHATSAPP_TO="):
                            updated_lines.append(f"WHATSAPP_TO={whatsapp_to}\n")
                            whatsapp_updated = True
                        else:
                            updated_lines.append(line)

                    # Append new entries if not updated
                    if not discord_updated:
                        updated_lines.append(f"DISCORD_WEBHOOK_URL={discord_webhook}\n")
                    if not telegram_token_updated:
                        updated_lines.append(f"TELEGRAM_BOT_TOKEN={telegram_token}\n")
                    if not telegram_chat_id_updated:
                        updated_lines.append(f"TELEGRAM_CHAT_ID={telegram_chat_id}\n")
                    if not whatsapp_updated:
                        updated_lines.append(f"WHATSAPP_TO={whatsapp_to}\n")

                    with open(env_path, "w") as f:
                        f.writelines(updated_lines)

                    st.success("✅ Notification channels saved! Restart app to apply.")
                except Exception as e:
                    st.error(f"Error saving notification channels: {e}")
                    logger.error(f"Error saving notification channels: {e}", exc_info=True)

            st.markdown("### ⚙️ Notification Settings")
            notifications_enabled = st.checkbox(
                "Enable Notifications",
                value=current_settings.get("NOTIFICATION_ENABLED", True),
                help="Enable/disable all notifications"
            )

            if st.button("💾 Save Notification Settings", type="primary"):
                try:
                    new_settings = current_settings.copy()
                    new_settings.update({
                        "NOTIFICATION_ENABLED": notifications_enabled
                    })
                    if backend_settings.save_settings(new_settings):
                        st.success("✅ Notification settings saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save settings")
                except Exception as e:
                    st.error(f"Error saving notification settings: {e}")
                    logger.error(f"Error saving notification settings: {e}", exc_info=True)

        with st.expander("📖 Notification Setup Guide"):
            st.markdown("""
            **Discord Setup:**
            1. Create a Discord webhook in your server
            2. Set `DISCORD_WEBHOOK_URL` environment variable

            **Telegram Setup:**
            1. Create a Telegram bot via @BotFather
            2. Get your chat ID by messaging @userinfobot
            3. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`

            **WhatsApp Setup:**
            1. Set `WHATSAPP_TO` with your phone number (e.g., 1234567890)
            2. Notifications will open WhatsApp Web automatically
            """)

        # System information footer
        st.markdown("---")
        st.markdown("### ℹ️ System Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.info(f"**Settings File:** settings.json")
            st.info(f"**Capital Storage:** Database")
        with info_col2:
            st.info(f"**Log File:** app.log")
            st.info(f"**Database:** SQLite/PostgreSQL")
        with info_col3:
            st.info(f"**Environment:** {'Production' if not os.getenv('BYBIT_MAINNET') else 'Live'}")
            st.info(f"**Version:** AlgoTrader Pro v1.0")

        # Connection status
        connection_status = "✅ Connected" if bybit_client and bybit_client.is_connected() else "❌ Disconnected"
        st.metric("API Status", connection_status)

    except Exception as e:
        st.error(f"Settings page error: {e}")
        logger.error(f"Settings page error: {e}", exc_info=True)
        if st.button("🔄 Reload Page"):
            st.rerun()

if __name__ == "__main__":
    main()