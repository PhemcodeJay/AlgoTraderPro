from datetime import datetime
import streamlit as st
import os
import sys
from db import db_manager, WalletBalance
from utils import send_discord, send_telegram

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import TradingEngine
from bybit_client import BybitClient
from settings import load_settings, save_settings, validate_env

# Configure logging
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
        default_virtual = {"available": 100.0, "capital": 100.0, "used": 0.0, "start_balance": 100.0}
        default_real = {"available": 0.0, "capital": 0.0, "used": 0.0, "start_balance": 0.0}
        
        capital_data["virtual"] = {
            "available": getattr(virtual_balance, "available", default_virtual["available"]),
            "capital": getattr(virtual_balance, "capital", default_virtual["capital"]),
            "used": getattr(virtual_balance, "used", default_virtual["used"]),
            "start_balance": getattr(virtual_balance, "start_balance", default_virtual["start_balance"])
        } if virtual_balance else default_virtual
        
        capital_data["real"] = {
            "available": getattr(real_balance, "available", default_real["available"]),
            "capital": getattr(real_balance, "capital", default_real["capital"]),
            "used": getattr(real_balance, "used", default_real["used"]),
            "start_balance": getattr(real_balance, "start_balance", default_real["start_balance"])
        } if real_balance else default_real
        
        return capital_data
    except Exception as e:
        st.error(f"Error loading capital data from database: {e}")
        logger.error(f"Error loading capital data: {e}", exc_info=True)
        return {"virtual": {"available": 100.0, "capital": 100.0, "used": 0.0, "start_balance": 100.0},
                "real": {"available": 0.0, "capital": 0.0, "used": 0.0, "start_balance": 0.0}}

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
            virtual_balance.capital = float(v.get("capital", virtual_balance.capital))
            virtual_balance.available = float(v.get("available", virtual_balance.available))
            virtual_balance.used = float(v.get("used", virtual_balance.used))
            virtual_balance.start_balance = float(v.get("start_balance", virtual_balance.start_balance))
            virtual_balance.currency = v.get("currency", "USDT")
            virtual_balance.updated_at = datetime.utcnow()
            db_manager.update_wallet_balance(virtual_balance)

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
            real_balance.capital = float(r.get("capital", real_balance.capital))
            real_balance.available = float(r.get("available", real_balance.available))
            real_balance.used = float(r.get("used", real_balance.used))
            real_balance.start_balance = float(r.get("start_balance", real_balance.start_balance))
            real_balance.currency = r.get("currency", "USDT")
            real_balance.updated_at = datetime.utcnow()
            db_manager.update_wallet_balance(real_balance)

        return True
    except Exception as e:
        st.error(f"Failed to update capital data: {e}")
        logger.error(f"Failed to update capital data: {e}", exc_info=True)
        return False

def main():
    try:
        engine = st.session_state.engine
        if not engine:
            st.error("Trading engine not initialized")
            logger.error("Trading engine not initialized")
            return

        # --- Sidebar ---
        with st.sidebar:
            st.markdown("### 🎛️ Navigation")
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
            balance_data = st.session_state.get("wallet_cache", {}).get(st.session_state.trading_mode, {"capital": 0.0, "available": 0.0, "used": 0.0})
            if st.session_state.trading_mode == "virtual":
                st.metric("💻 Virtual Capital", f"${balance_data['capital']:.2f}")
                st.metric("💻 Virtual Available", f"${balance_data['available']:.2f}")
                st.metric("💻 Virtual Used", f"${balance_data['used']:.2f}")
            else:
                st.metric("🏦 Real Capital", f"${balance_data['capital']:.2f}")
                st.metric("🏦 Real Available", f"${balance_data['available']:.2f}")
                st.metric("🏦 Real Used Margin", f"${balance_data['used']:.2f}")
            
            st.markdown(
                f"<small style='color:#888;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>",
                unsafe_allow_html=True
            )

        st.markdown("### ⚙️ Trading Settings")
        
        # Load current settings
        current_settings = load_settings()
        
        tab1, tab2, tab3 = st.tabs(["💰 Capital Management", "📈 Trading Parameters", "📬 Notifications"])
        
        with tab1:
            st.subheader("💰 Capital Management")
            capital_data = load_capital_data()
            
            st.markdown("#### Virtual Account")
            v_capital = st.number_input("Virtual Capital (USDT)", min_value=0.0, value=float(capital_data["virtual"]["capital"]), step=100.0)
            v_available = st.number_input("Virtual Available (USDT)", min_value=0.0, value=float(capital_data["virtual"]["available"]), step=100.0)
            v_used = st.number_input("Virtual Used (USDT)", min_value=0.0, value=float(capital_data["virtual"]["used"]), step=10.0)
            v_start_balance = st.number_input("Virtual Start Balance (USDT)", min_value=0.0, value=float(capital_data["virtual"]["start_balance"]), step=100.0)
            
            st.markdown("#### Real Account")
            r_capital = st.number_input("Real Capital (USDT)", min_value=0.0, value=float(capital_data["real"]["capital"]), step=100.0, disabled=True, help="Real balance is managed by Bybit API")
            r_available = st.number_input("Real Available (USDT)", min_value=0.0, value=float(capital_data["real"]["available"]), step=100.0, disabled=True, help="Real balance is managed by Bybit API")
            r_used = st.number_input("Real Used (USDT)", min_value=0.0, value=float(capital_data["real"]["used"]), step=10.0, disabled=True, help="Real balance is managed by Bybit API")
            r_start_balance = st.number_input("Real Start Balance (USDT)", min_value=0.0, value=float(capital_data["real"]["start_balance"]), step=100.0, disabled=True, help="Real balance is managed by Bybit API")
            
            if st.button("💾 Save Capital Settings", type="primary"):
                new_capital_data = {
                    "virtual": {
                        "capital": v_capital,
                        "available": v_available,
                        "used": v_used,
                        "start_balance": v_start_balance,
                        "currency": "USDT"
                    },
                    "real": {
                        "capital": r_capital,
                        "available": r_available,
                        "used": r_used,
                        "start_balance": r_start_balance,
                        "currency": "USDT"
                    }
                }
                if save_capital_data(new_capital_data):
                    st.success("✅ Capital settings saved!")
                    st.session_state.wallet_cache.clear()
                    st.rerun()
                else:
                    st.error("❌ Failed to save capital settings")
        
        with tab2:
            st.subheader("📈 Trading Parameters")
            
            scan_interval = st.number_input("Scan Interval (seconds)", min_value=60.0, value=float(current_settings.get("SCAN_INTERVAL", 3600.0)), step=60.0)
            top_n_signals = st.number_input("Top N Signals", min_value=1, value=int(current_settings.get("TOP_N_SIGNALS", 10)), step=1)
            max_loss_pct = st.number_input("Max Loss %", max_value=0.0, value=float(current_settings.get("MAX_LOSS_PCT", -15.0)), step=1.0)
            tp_percent = st.number_input("Take Profit %", min_value=0.0, value=float(current_settings.get("TP_PERCENT", 0.25)), step=0.01)
            sl_percent = st.number_input("Stop Loss %", min_value=0.0, value=float(current_settings.get("SL_PERCENT", 0.05)), step=0.01)
            max_drawdown_pct = st.number_input("Max Drawdown %", max_value=0.0, value=float(current_settings.get("MAX_DRAWDOWN_PCT", -20.0)), step=1.0)
            leverage = st.number_input("Leverage", min_value=1.0, value=float(current_settings.get("LEVERAGE", 10.0)), step=1.0)
            risk_pct = st.number_input("Risk % per Trade", min_value=0.0, value=float(current_settings.get("RISK_PCT", 0.02)), step=0.01)
            entry_buffer_pct = st.number_input("Entry Buffer %", min_value=0.0, value=float(current_settings.get("ENTRY_BUFFER_PCT", 0.002)), step=0.001)
            max_positions = st.number_input("Max Positions", min_value=1, value=int(current_settings.get("MAX_POSITIONS", 10)), step=1)
            min_signal_score = st.number_input("Min Signal Score", min_value=0.0, value=float(current_settings.get("MIN_SIGNAL_SCORE", 40.0)), step=1.0)
            auto_trading_enabled = st.checkbox("Enable Auto Trading", value=current_settings.get("AUTO_TRADING_ENABLED", True))
            
            symbols = st.multiselect(
                "Trading Symbols",
                options=["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"],
                default=current_settings.get("SYMBOLS", ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT"])
            )
            
            if st.button("💾 Save Trading Parameters", type="primary"):
                try:
                    new_settings = current_settings.copy()
                    new_settings.update({
                        "SCAN_INTERVAL": scan_interval,
                        "TOP_N_SIGNALS": top_n_signals,
                        "MAX_LOSS_PCT": max_loss_pct,
                        "TP_PERCENT": tp_percent,
                        "SL_PERCENT": sl_percent,
                        "MAX_DRAWDOWN_PCT": max_drawdown_pct,
                        "LEVERAGE": leverage,
                        "RISK_PCT": risk_pct,
                        "ENTRY_BUFFER_PCT": entry_buffer_pct,
                        "SYMBOLS": symbols,
                        "MAX_POSITIONS": max_positions,
                        "MIN_SIGNAL_SCORE": min_signal_score,
                        "AUTO_TRADING_ENABLED": auto_trading_enabled
                    })
                    if save_settings(new_settings):
                        st.success("✅ Trading parameters saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save trading parameters")
                except Exception as e:
                    st.error(f"Error saving trading parameters: {e}")
                    logger.error(f"Error saving trading parameters: {e}", exc_info=True)
        
        with tab3:
            st.subheader("📬 Notification Settings")
            
            # Validate environment variables
            if not validate_env():
                st.error("❌ Missing required environment variables (BYBIT_API_KEY, BYBIT_API_SECRET). Check .env file.")
            
            discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
            discord_status = "✅ Configured" if discord_webhook else "❌ Not Set"
            st.metric("Discord Webhook", discord_status)
            if st.button("📤 Test Discord"):
                try:
                    test_signal = [{
                        "symbol": "BTCUSDT",
                        "side": "LONG",
                        "score": 85.0,
                        "entry": 45000.00,
                        "take_profit": 46000.00,
                        "stop_loss": 44000.00,
                        "strategy": "Test",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }]
                    send_discord(test_signal)
                    st.success("✅ Discord test sent!")
                except Exception as e:
                    st.error(f"Discord test failed: {e}")
                    logger.error(f"Discord test failed: {e}", exc_info=True)
            
            telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "")
            telegram_status = "✅ Configured" if telegram_token and telegram_chat else "❌ Not Set"
            st.metric("Telegram Bot", telegram_status)
            if st.button("📤 Test Telegram"):
                try:
                    test_signal = [{
                        "symbol": "BTCUSDT",
                        "side": "LONG",
                        "score": 85.0,
                        "entry": 45000.00,
                        "take_profit": 46000.00,
                        "stop_loss": 44000.00,
                        "strategy": "Test",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }]
                    send_telegram(test_signal)
                    st.success("✅ Telegram test sent!")
                except Exception as e:
                    st.error(f"Telegram test failed: {e}")
                    logger.error(f"Telegram test failed: {e}", exc_info=True)
            
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
                    logger.error(f"Error saving notification settings: {e}", exc_info=True)
        
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