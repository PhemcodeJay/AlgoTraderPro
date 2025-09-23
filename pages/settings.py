from datetime import datetime
import streamlit as st
import os
import sys
from db import db_manager, WalletBalance

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import TradingEngine
from bybit_client import BybitClient
from settings import load_settings, save_settings, validate_env

# Configure logging
# Logging using centralized system
from logging_config import get_logger
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
        return {}

def save_capital_data(capital_data: dict) -> bool:
    """
    Save capital data to the database using db_manager.
    Handles both virtual and real balances.
    """
    try:
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
                updated_at=datetime.utcnow(),
            )

            # Update fields from input
            virtual_balance.capital = float(v.get("capital", virtual_balance.capital))
            virtual_balance.available = float(v.get("available", virtual_balance.available))
            virtual_balance.used = float(v.get("used", virtual_balance.used))
            virtual_balance.start_balance = float(v.get("start_balance", virtual_balance.start_balance))
            virtual_balance.currency = v.get("currency", "USDT")
            virtual_balance.updated_at = datetime.utcnow()

            try:
                db_manager.update_wallet_balance(virtual_balance)
            except Exception as e:
                st.error(f"Failed to update virtual balance: {e}")
                return False

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
                updated_at=datetime.utcnow(),
            )

            # Update fields from input
            real_balance.capital = float(r.get("capital", real_balance.capital))
            real_balance.available = float(r.get("available", real_balance.available))
            real_balance.used = float(r.get("used", real_balance.used))
            real_balance.start_balance = float(r.get("start_balance", real_balance.start_balance))
            real_balance.currency = r.get("currency", "USDT")
            real_balance.updated_at = datetime.utcnow()

            try:
                db_manager.update_wallet_balance(real_balance)
            except Exception as e:
                st.error(f"Failed to update real balance: {e}")
                return False

        return True

    except Exception as e:
        st.error(f"Error saving capital data: {e}")
        return False

def main():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">⚙️ Settings</h1>
        <p style="color: #888; margin: 0;">Configure Trading Parameters & System Settings</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings Navigation")

        # System status
        env_valid = validate_env()
        status_color = "🟢" if env_valid else "🔴"
        st.metric("Environment", f"{status_color} {'Valid' if env_valid else 'Issues'}")

        # Quick actions
        if st.button("🔄 Reload Settings", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if st.button("📊 Dashboard", use_container_width=True):
            st.switch_page("app.py")

        st.divider()

        # Warning about real trading
        st.warning("⚠️ Changes to real trading settings require careful consideration and proper API configuration.")

    try:
        # Initialize engine and Bybit client
        engine = TradingEngine() if "engine" not in st.session_state else st.session_state.engine
        bybit_client = BybitClient()

        # Load settings and capital data
        current_settings = load_settings()
        capital_data = load_capital_data()

        # Main settings tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Trading", "🔍 Signal Generation", "💰 Capital Management", "🔑 API Configuration", "🔔 Notifications"
        ])

        with tab1:
            st.subheader("🎯 Trading Configuration")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Risk Management")
                leverage = st.number_input(
                    "Leverage",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(current_settings.get("LEVERAGE", 10)),
                    help="Maximum leverage for trades"
                )
                risk_pct = st.number_input(
                    "Risk per Trade (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=float(current_settings.get("RISK_PCT", 0.02)) * 100,
                    step=0.1,
                    help="Percentage of balance to risk per trade"
                )
                max_positions = st.number_input(
                    "Maximum Open Positions",
                    min_value=1.0,
                    max_value=20.0,
                    value=float(current_settings.get("MAX_POSITIONS", 5)),
                    help="Maximum number of concurrent open positions"
                )

            with col2:
                st.markdown("### 🎯 Take Profit & Stop Loss")
                tp_percent = st.number_input(
                    "Take Profit (%)",
                    min_value=10.0,
                    max_value=100.0,
                    value=float(current_settings.get("TP_PERCENT", 25.0)),
                    step=1.0,
                    help="Default take profit percentage"
                )
                sl_percent = st.number_input(
                    "Stop Loss (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=float(current_settings.get("SL_PERCENT", 5.0)),
                    step=1.0,
                    help="Default stop loss percentage"
                )
                max_drawdown = st.number_input(
                    "Maximum Drawdown (%)",
                    min_value=5.0,
                    max_value=50.0,
                    value=abs(float(current_settings.get("MAX_DRAWDOWN_PCT", -20.0))),
                    step=1.0,
                    help="Maximum allowed portfolio drawdown"
                )

            with st.expander("🔧 Advanced Trading Settings"):
                entry_buffer = st.number_input(
                    "Entry Buffer (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_settings.get("ENTRY_BUFFER_PCT", 0.002)) * 100,
                    step=0.01,
                    help="Buffer percentage for entry price adjustments"
                )
                use_websocket = st.checkbox(
                    "Use WebSocket for Real-time Data",
                    value=current_settings.get("USE_WEBSOCKET", True),
                    help="Enable WebSocket connections for faster price updates"
                )
                auto_trading = st.checkbox(
                    "Enable Automated Trading",
                    value=current_settings.get("AUTO_TRADING_ENABLED", False),
                    help="Allow the system to execute trades automatically"
                )

            if st.button("💾 Save Trading Settings", type="primary"):
                try:
                    new_settings = current_settings.copy()
                    new_settings.update({
                        "LEVERAGE": int(leverage),
                        "RISK_PCT": risk_pct / 100,
                        "MAX_POSITIONS": int(max_positions),
                        "TP_PERCENT": tp_percent,
                        "SL_PERCENT": sl_percent,
                        "MAX_DRAWDOWN_PCT": -max_drawdown,
                        "ENTRY_BUFFER_PCT": entry_buffer / 100,
                        "USE_WEBSOCKET": use_websocket,
                        "AUTO_TRADING_ENABLED": auto_trading
                    })
                    if save_settings(new_settings):
                        st.success("✅ Trading settings saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save settings")
                except Exception as e:
                    st.error(f"Error saving settings: {e}")

        with tab2:
            st.subheader("🔍 Signal Generation Settings")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ⏱️ Timing Settings")
                scan_interval = st.number_input(
                    "Scan Interval (minutes)",
                    min_value=1.0,
                    max_value=1440.0,
                    value=float(current_settings.get("SCAN_INTERVAL", 3600) // 60),
                    help="How often to scan for new signals"
                )
                top_n_signals = st.number_input(
                    "Top N Signals",
                    min_value=1.0,
                    max_value=50.0,
                    value=float(current_settings.get("TOP_N_SIGNALS", 10)),
                    help="Number of top signals to generate"
                )
                min_signal_score = st.number_input(
                    "Minimum Signal Score",
                    min_value=30.0,
                    max_value=100.0,
                    value=float(current_settings.get("MIN_SIGNAL_SCORE", 50)),
                    help="Minimum score for signals to be considered"
                )

            with col2:
                st.markdown("### 📊 Indicators")
                rsi_oversold = st.number_input(
                    "RSI Oversold Threshold",
                    min_value=10.0,
                    max_value=40.0,
                    value=float(current_settings.get("RSI_OVERSOLD", 30)),
                    help="RSI level considered oversold"
                )
                rsi_overbought = st.number_input(
                    "RSI Overbought Threshold",
                    min_value=60.0,
                    max_value=90.0,
                    value=float(current_settings.get("RSI_OVERBOUGHT", 70)),
                    help="RSI level considered overbought"
                )
                min_volume = st.number_input(
                    "Minimum Volume (USDT)",
                    min_value=100000.0,
                    max_value=100000000.0,
                    value=float(current_settings.get("MIN_VOLUME", 1000000)),
                    help="Minimum 24h volume for symbol selection"
                )

            st.markdown("### 🎯 Symbol Selection")
            available_symbols = [
                "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT",
                "BNBUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
                "LTCUSDT", "BCHUSDT", "ATOMUSDT", "ALGOUSDT", "VETUSDT"
            ]
            current_symbols = current_settings.get("SYMBOLS", available_symbols[:7])
            selected_symbols = st.multiselect(
                "Trading Symbols",
                available_symbols,
                default=current_symbols,
                help="Select symbols to include in signal generation"
            )

            if st.button("💾 Save Signal Settings", type="primary"):
                try:
                    new_settings = current_settings.copy()
                    new_settings.update({
                        "SCAN_INTERVAL": int(scan_interval * 60),
                        "TOP_N_SIGNALS": int(top_n_signals),
                        "MIN_SIGNAL_SCORE": int(min_signal_score),
                        "RSI_OVERSOLD": int(rsi_oversold),
                        "RSI_OVERBOUGHT": int(rsi_overbought),
                        "MIN_VOLUME": int(min_volume),
                        "SYMBOLS": selected_symbols
                    })
                    if save_settings(new_settings):
                        st.success("✅ Signal settings saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save settings")
                except Exception as e:
                    st.error(f"Error saving settings: {e}")

        with tab3:
            st.subheader("💰 Capital Management")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 💼 Virtual Capital")
                virtual_capital = st.number_input(
                    "Virtual Capital (USDT)",
                    value=capital_data.get("virtual", {}).get("capital", 100.0),
                    min_value=0.0,
                    step=100.0,
                    help="Total virtual capital available"
                )
                virtual_available = st.number_input(
                    "Virtual Available (USDT)",
                    value=capital_data.get("virtual", {}).get("available", 100.0),
                    min_value=0.0,
                    step=100.0,
                    help="Available virtual balance for trading"
                )
                virtual_used = virtual_capital - virtual_available
                st.metric("Used Margin (Virtual)", f"${virtual_used:.2f}")

            with col2:
                st.markdown("### 📈 Real Capital")

                # Display API Connection Status
                api_status = "🟢 Connected" if bybit_client.is_connected() else "🔴 Disconnected"
                st.metric("API Connection", api_status)

                if not bybit_client.is_connected():
                    st.warning("Bybit API not connected. Check API keys in .env file.")

                # Display real balances as metrics (read-only)
                real_capital_value = capital_data.get("real", {}).get("capital", 0.0)
                real_available_value = capital_data.get("real", {}).get("available", 0.0)
                real_used_value = capital_data.get("real", {}).get("used", 0.0)

                st.metric("Real Capital (USDT)", f"${real_capital_value:.2f}")
                st.metric("Real Available (USDT)", f"${real_available_value:.2f}")
                st.metric("Used Margin (Real)", f"${real_used_value:.2f}")

                # Info if available == 0 but capital > 0
                if real_available_value == 0.0 and real_capital_value > 0.0:
                    st.info("Available balance is $0.00. Funds may be in use (e.g., open positions).")
                elif real_available_value == 0.0 and real_capital_value == 0.0 and bybit_client.is_connected():
                    st.warning("No funds available in Bybit account. Verify account balance or API permissions.")

                if st.button("🔄 Sync Real Balance"):
                    if not bybit_client.is_connected():
                        st.error("❌ Cannot sync: Bybit API not connected. Check API keys in .env file.")
                    else:
                        try:
                            if engine.sync_real_balance():
                                st.success("✅ Real balance synced successfully!")
                                st.rerun()  # Reload to reflect updated values
                            else:
                                st.error("❌ Failed to sync real balance. Check Bybit account or API permissions.")
                        except Exception as e:
                            st.error(f"❌ Sync failed: {e}")
                            logger.error(f"Error during real balance sync: {e}", exc_info=True)

            if st.button("💾 Save Capital Settings", type="primary"):
                new_capital = {
                    "virtual": {
                        "capital": virtual_capital,
                        "available": virtual_available,
                        "used": virtual_used,
                        "start_balance": capital_data.get("virtual", {}).get("start_balance", virtual_capital)
                    },
                    "real": {
                        "capital": real_capital_value,
                        "available": real_available_value,
                        "used": real_used_value,
                        "start_balance": capital_data.get("real", {}).get("start_balance", real_capital_value)
                    }
                }
                if save_capital_data(new_capital):
                    st.success("✅ Capital settings saved!")
                    st.rerun()
                else:
                    st.error("❌ Failed to save capital settings")

        with tab4:
            st.subheader("🔑 API Configuration")

            st.warning("⚠️ API keys are managed through environment variables for security")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📡 Bybit API")
                api_key_status = "✅ Configured" if os.getenv("BYBIT_API_KEY") else "❌ Not Set"
                st.metric("API Key", api_key_status)
                secret_status = "✅ Configured" if os.getenv("BYBIT_API_SECRET") else "❌ Not Set"
                st.metric("API Secret", secret_status)
                mainnet_mode = os.getenv("BYBIT_MAINNET", "false").lower() == "true"
                mode_text = "🧪 Mainnet" if mainnet_mode else "🔴 Testnet"
                st.metric("Trading Mode", mode_text)

            with col2:
                st.markdown("### 🔗 Connection Status")
                if st.button("🔍 Test API Connection"):
                    with st.spinner("Testing connection..."):
                        try:
                            connection_ok = bybit_client._test_connection()
                            if connection_ok:
                                st.success("✅ API connection successful!")
                            else:
                                st.error("❌ API connection failed")
                        except Exception as e:
                            st.error(f"Connection test error: {e}")
                            logger.error(f"API connection test failed: {e}", exc_info=True)

                if bybit_client.is_connected():
                    st.success("✅ Currently connected to Bybit API")
                    try:
                        balances = bybit_client.get_account_balance()
                        if balances:
                            st.info(f"Account has {len(balances)} currencies")
                    except Exception as e:
                        st.warning(f"Account info error: {e}")
                else:
                    st.error("❌ Not connected to Bybit API")

            with st.expander("📖 API Configuration Guide"):
                st.markdown("""
                **To configure Bybit API:**

                1. **Create API Key:**
                   - Log into your Bybit account
                   - Go to Account & Security > API Management
                   - Create a new API key with trading permissions

                2. **Set Environment Variables:**
                   ```
                   BYBIT_API_KEY=your_api_key_here
                   BYBIT_API_SECRET=your_api_secret_here
                   BYBIT_mainnet=false  # Use true for mainnet
                   ```

                3. **Required Permissions:**
                   - Read wallet balance
                   - Place/modify/cancel orders
                   - View trading history

                4. **Security Notes:**
                   - Never share your API credentials
                   - Use IP whitelist if possible
                   - Start with mainnet for testing
                """)

        with tab5:
            st.subheader("🔔 Notification Settings")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📱 Discord")
                discord_url = os.getenv("DISCORD_WEBHOOK_URL", "")
                discord_status = "✅ Configured" if discord_url else "❌ Not Set"
                st.metric("Discord Webhook", discord_status)
                if st.button("📤 Test Discord"):
                    try:
                        from notification_pdf import send_discord
                        test_signal = [{
                            'Symbol': 'BTCUSDT',
                            'Type': 'Buy',
                            'Side': 'LONG',
                            'Score': '85.0',
                            'Entry': 45000.00,
                            'TP': 46000.00,
                            'SL': 44000.00,
                            'Market': 'Test',
                            'Time': 'Test Signal'
                        }]
                        send_discord(test_signal)
                        st.success("✅ Discord test sent!")
                    except Exception as e:
                        st.error(f"Discord test failed: {e}")

                st.markdown("### 📞 WhatsApp")
                whatsapp_number = os.getenv("WHATSAPP_TO", "")
                whatsapp_status = "✅ Configured" if whatsapp_number else "❌ Not Set"
                st.metric("WhatsApp Number", whatsapp_status)

            with col2:
                st.markdown("### 📨 Telegram")
                telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
                telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "")
                telegram_status = "✅ Configured" if telegram_token and telegram_chat else "❌ Not Set"
                st.metric("Telegram Bot", telegram_status)
                if st.button("📤 Test Telegram"):
                    try:
                        from notification_pdf import send_telegram
                        test_signal = [{
                            'Symbol': 'BTCUSDT',
                            'Type': 'Buy',
                            'Side': 'LONG',
                            'Score': '85.0',
                            'Entry': 45000.00,
                            'TP': 46000.00,
                            'SL': 44000.00,
                            'Market': 'Test',
                            'Time': 'Test Signal'
                        }]
                        send_telegram(test_signal)
                        st.success("✅ Telegram test sent!")
                    except Exception as e:
                        st.error(f"Telegram test failed: {e}")

                st.markdown("### ⚙️ Notification Settings")
                notifications_enabled = st.checkbox(
                    "Enable Notifications",
                    value=current_settings.get("NOTIFICATION_ENABLED", True),
                    help="Enable/disable all notifications"
                )

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

            if st.button("💾 Save Notification Settings", type="primary"):
                try:
                    new_settings = current_settings.copy()
                    new_settings.update({
                        "NOTIFICATION_ENABLED": notifications_enabled
                    })
                    if save_settings(new_settings):
                        st.success("✅ Notification settings saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save settings")
                except Exception as e:
                    st.error(f"Error saving settings: {e}")

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

    except Exception as e:
        st.error(f"Settings page error: {e}")
        logger.error(f"Settings page error: {e}", exc_info=True)
        if st.button("🔄 Reload Page"):
            st.rerun()

if __name__ == "__main__":
    main()