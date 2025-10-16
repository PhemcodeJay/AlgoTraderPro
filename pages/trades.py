import streamlit as st
import pandas as pd
import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
import json
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import TradingEngine
from db import db_manager, TradeModel
from automated_trader import AutomatedTrader
from utils import calculate_portfolio_metrics
from signal_generator import get_usdt_symbols
from sqlalchemy import update
from exceptions import APIException

# Initialize database
db = db_manager

# Configure logging
from logging_config import get_logger
logger = get_logger(__name__)

st.set_page_config(
    page_title="Trades - AlgoTraderPro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def get_engine():
    return TradingEngine()

@st.cache_resource  
def get_automated_trader():
    engine = get_engine()
    return AutomatedTrader(engine, engine.client)

async def close_trade_safely(trade_id: str, virtual: bool = True) -> bool:
    """Close a trade with proper error handling, including closing real trades on Bybit."""
    try:
        engine = get_engine()

        # Get trade from database
        open_trades = [t for t in db_manager.get_trades(limit=1000) if t.status == "open"]
        trade = next((t for t in open_trades if str(t.id) == str(trade_id) or str(t.order_id) == str(trade_id)), None)

        if not trade:
            st.error(f"Trade {trade_id} not found")
            return False

        # Initialize variables
        current_price = engine.client.get_current_price(trade.symbol)
        pnl = 0.0

        # Handle real trades
        if not virtual:
            try:
                # Fetch position data
                positions = await engine.client.get_positions(symbol=trade.symbol)
                position = next((p for p in positions if p["side"].upper() == trade.side.upper()), None)

                if position:
                    # Close position on Bybit
                    close_side = "Sell" if trade.side.upper() in ["BUY", "LONG"] else "Buy"
                    try:
                        close_response = await engine.client.place_order(
                            symbol=trade.symbol,
                            side=close_side,
                            qty=trade.qty,
                            leverage=trade.leverage or 10
                        )
                        if "error" in close_response:
                            st.error(f"Failed to close position for {trade.symbol}: {close_response.get('error', 'Unknown error')}")
                            logger.error(f"Failed to close position for {trade.symbol}: {close_response}")
                            return False
                    except APIException as e:
                        if e.error_code == "100028":
                            logger.warning(f"Unified account error closing {trade.order_id}: {e}. Retrying with cross margin mode.")
                            close_response = await engine.client.place_order(
                                symbol=trade.symbol,
                                side=close_side,
                                qty=trade.qty,
                                leverage=trade.leverage or 10,
                                mode="CROSS"
                            )
                            if "error" in close_response:
                                st.error(f"Retry failed to close position for {trade.symbol}: {close_response.get('error', 'Unknown error')}")
                                logger.error(f"Retry failed to close position for {trade.symbol}: {close_response}")
                                return False
                            logger.info(f"Closed real position {trade.order_id} on retry")
                        else:
                            st.error(f"Failed to close position for {trade.symbol}: {e}")
                            logger.error(f"Failed to close position for {trade.symbol}: {e}", exc_info=True)
                            return False

                    # Use Bybit's unrealized PnL and mark price
                    pnl = float(position.get("unrealized_pnl", 0.0))
                    current_price = float(position.get("mark_price", current_price))
                else:
                    st.warning(f"No active position found for {trade.symbol}. Marking trade as closed.")
                    pnl = 0.0
            except Exception as e:
                st.error(f"Failed to fetch/close position for {trade.symbol}: {e}")
                logger.error(f"Failed to fetch/close position for {trade.symbol}: {e}", exc_info=True)
                return False
        else:
            # Virtual trade: Calculate PnL
            pnl = engine.calculate_virtual_pnl(trade.to_dict())

        # Update trade in database
        if not db_manager.session:
            logger.error("Database session not initialized")
            st.error("Database session not initialized")
            return False

        try:
            db_manager.session.execute(
                update(TradeModel)
                .where(TradeModel.order_id == trade.order_id)
                .values(
                    status="closed",
                    exit_price=current_price,
                    pnl=pnl,
                    closed_at=datetime.now(timezone.utc)
                )
            )
            db_manager.session.commit()
            success = True
        except Exception as e:
            db_manager.session.rollback()
            logger.error(f"Database error updating trade {trade.order_id}: {e}", exc_info=True)
            st.error(f"Database error updating trade: {e}")
            return False

        if success:
            # Update virtual balance if it's a virtual trade
            if virtual:
                engine.update_virtual_balances(pnl)

            st.success(f"✅ Trade closed successfully! PnL: ${pnl:.2f}")
            return True
        else:
            st.error("❌ Failed to close trade in database")
            return False

    except Exception as e:
        st.error(f"Error closing trade: {e}")
        logger.error(f"Error closing trade {trade_id}: {e}", exc_info=True)
        return False

def display_trade_management():
    """Display trade management interface for virtual and real trades."""
    engine = get_engine()

    # Trading mode switch
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎮 Virtual Trades")
        virtual_trades = engine.get_open_virtual_trades()

        if virtual_trades:
            for i, trade in enumerate(virtual_trades):
                with st.expander(f"{trade.symbol} {trade.side} - ${trade.entry_price:.4f}"):
                    current_price = engine.client.get_current_price(trade.symbol)
                    current_pnl = engine.calculate_virtual_pnl(trade.to_dict())

                    pnl_color = "🟢" if current_pnl > 0 else "🔴" if current_pnl < 0 else "🟡"

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.write(f"**Quantity:** {trade.qty:.6f}")
                        st.write(f"**Score:** {trade.score or 0:.1f}%")
                        st.write(f"**Current Price:** ${current_price:.4f}")
                        st.write(f"**SL:** ${trade.sl:.4f}" if trade.sl else "N/A")
                        st.write(f"**TP:** ${trade.tp:.4f}" if trade.tp else "N/A")

                    with col_b:
                        st.write(f"**Current PnL:** {pnl_color} ${current_pnl:.2f}")
                        st.write(f"**Status:** {trade.status.title()}")
                        st.write(f"**Trail:** ${trade.trail:.4f}" if trade.trail else "N/A")
                        st.write(f"**Liquidation:** ${trade.liquidation:.4f}" if trade.liquidation else "N/A")
                        st.write(f"**Margin:** ${trade.margin_usdt:.2f}" if trade.margin_usdt else "N/A")

                    if st.button("❌ Close", key=f"close_virtual_{trade.id}"):
                        asyncio.run(close_trade_safely(str(trade.id), virtual=True))
                        st.rerun()
        else:
            st.info("No open virtual trades")

    with col2:
        st.subheader("💰 Real Trades")
        # Sync real trades with Bybit and show feedback
        with st.spinner("Syncing real trades with Bybit..."):
            success = engine.sync_real_trades()
            if success:
                st.success("✅ Successfully synced real trades from Bybit")
            else:
                st.error("❌ Failed to sync real trades from Bybit")

        real_trades = engine.get_open_real_trades()

        if real_trades:
            for i, trade in enumerate(real_trades):
                with st.expander(f"{trade.symbol} {trade.side} - ${trade.entry_price:.4f}"):
                    # Fetch real-time position data from Bybit
                    try:
                        positions = asyncio.run(engine.client.get_positions(symbol=trade.symbol))
                        position = next((p for p in positions if p["side"].upper() == trade.side.upper()), None)
                        current_price = float(position.get("mark_price", engine.client.get_current_price(trade.symbol))) if position else engine.client.get_current_price(trade.symbol)
                        current_pnl = float(position.get("unrealized_pnl", 0.0)) if position else 0.0
                    except Exception as e:
                        logger.warning(f"Failed to fetch real position data for {trade.symbol}: {e}")
                        st.warning(f"Could not fetch real-time data for {trade.symbol}. Using fallback price.")
                        current_price = engine.client.get_current_price(trade.symbol)
                        current_pnl = 0.0

                    pnl_color = "🟢" if current_pnl > 0 else "🔴" if current_pnl < 0 else "🟡"

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.write(f"**Quantity:** {trade.qty:.6f}")
                        st.write(f"**Score:** {trade.score or 0:.1f}%")
                        st.write(f"**Current Price:** ${current_price:.4f}")
                        st.write(f"**SL:** ${trade.sl:.4f}" if trade.sl else "N/A")
                        st.write(f"**TP:** ${trade.tp:.4f}" if trade.tp else "N/A")

                    with col_b:
                        st.write(f"**Current PnL:** {pnl_color} ${current_pnl:.2f}")
                        st.write(f"**Status:** {trade.status.title()}")
                        st.write(f"**Trail:** ${trade.trail:.4f}" if trade.trail else "N/A")
                        st.write(f"**Liquidation:** ${trade.liquidation:.4f}" if trade.liquidation else "N/A")
                        st.write(f"**Margin:** ${trade.margin_usdt:.2f}" if trade.margin_usdt else "N/A")

                    if st.button("❌ Close", key=f"close_real_{trade.id}"):
                        asyncio.run(close_trade_safely(str(trade.id), virtual=False))
                        st.rerun()
        else:
            st.info("No open real trades on Bybit")

def display_manual_trading():
    """Display manual trading interface"""
    st.subheader("📝 Manual Trade Entry")

    engine = get_engine()
    symbols = get_usdt_symbols(50)

    col1, col2 = st.columns(2)

    with col1:
        symbol = st.selectbox("Symbol", symbols, key="manual_symbol")
        side = st.selectbox("Side", ["Buy", "Sell"], key="manual_side")
        qty = st.number_input("Quantity", min_value=0.001, value=0.01, key="manual_qty")
        order_type = st.selectbox("Order Type", ["Market", "Limit"], key="manual_order_type")
        price = st.number_input("Price (for Limit orders)", min_value=0.0, key="manual_price") if order_type == "Limit" else None

    with col2:
        leverage = st.number_input("Leverage", min_value=1, max_value=100, value=10, key="manual_leverage")
        stop_loss = st.number_input("Stop Loss Price", min_value=0.0, key="manual_sl")
        take_profit = st.number_input("Take Profit Price", min_value=0.0, key="manual_tp")
        trail = st.number_input("Trailing Stop Price", min_value=0.0, key="manual_trail")
        margin_usdt = st.number_input("Margin (USDT)", min_value=0.0, value=5.0, key="manual_margin")
        trading_mode = st.selectbox("Execution Mode", ["virtual", "real"], key="manual_mode")

    if st.button("🚀 Place Order", type="primary"):
        if qty <= 0:
            st.error("Invalid quantity")
            return

        try:
            # Get current price if market order or no price specified
            current_price = engine.client.get_current_price(symbol)
            entry_price = price if order_type == "Limit" and price else current_price

            if entry_price <= 0:
                st.error("Invalid entry price")
                return

            # Calculate trail and liquidation if not provided
            trail_value = trail if trail > 0 else (
                abs(take_profit - entry_price) / 2 if take_profit > 0 else 0.0
            )
            liquidation_value = (
                entry_price * (1 - 0.9 / leverage) if side == "Buy"
                else entry_price * (1 + 0.9 / leverage)
            ) if leverage > 0 else 0.0
            margin_value = margin_usdt if margin_usdt > 0 else (entry_price * qty) / leverage

            # Create trade data
            trade_data = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry": entry_price,
                "order_id": f"manual_{symbol}_{int(datetime.now().timestamp())}",
                "virtual": trading_mode == "virtual",
                "status": "open",
                "strategy": "Manual",
                "leverage": leverage,
                "sl": stop_loss if stop_loss > 0 else None,
                "tp": take_profit if take_profit > 0 else None,
                "trail": trail_value if trail_value > 0 else None,
                "liquidation": liquidation_value if liquidation_value > 0 else None,
                "margin_usdt": margin_value if margin_value > 0 else None,
                "margin_mode": "CROSS" if trading_mode == "real" else None
            }

            # Add to database
            success = db_manager.add_trade(trade_data)

            if not success:
                st.error("❌ Failed to place order in database")
                return

            st.success(f"✅ {trading_mode.title()} order placed: {symbol} {side} @ ${entry_price:.4f}")

            # Handle trade execution based on mode
            if trading_mode == "virtual":
                # Update balance for virtual trades
                margin_used = margin_value or (entry_price * qty) / leverage
                engine.update_virtual_balances(-margin_used, "virtual")
                # Execute virtual trade
                success = engine.execute_virtual_trade(trade_data)
                if not success:
                    st.error("❌ Failed to execute virtual trade")
                    # Rollback DB entry
                    session = db_manager._get_session()
                    session.execute(
                        update(TradeModel)
                        .where(TradeModel.order_id == trade_data["order_id"])
                        .values(status="failed")
                    )
                    session.commit()
                    return
            else:
                # Execute real trade on Bybit
                success = asyncio.run(engine.execute_real_trade([trade_data]))
                if success:
                    # Sync trades after execution
                    time.sleep(2)
                    engine.sync_real_trades()
                    logger.info(f"Synced real trades to DB after executing manual trade for {symbol}")
                    st.success(f"✅ Real trade executed and synced for {symbol}")
                else:
                    st.error("❌ Failed to execute real trade on Bybit")
                    # Rollback DB entry
                    session = db_manager._get_session()
                    session.execute(
                        update(TradeModel)
                        .where(TradeModel.order_id == trade_data["order_id"])
                        .values(status="failed")
                    )
                    session.commit()
                    return

            st.rerun()

        except Exception as e:
            st.error(f"Order placement error: {e}")
            logger.error(f"Manual order error: {e}", exc_info=True)
            # Rollback DB entry on general error
            session = db_manager._get_session()
            session.execute(
                update(TradeModel)
                .where(TradeModel.order_id == trade_data["order_id"])
                .values(status="failed")
            )
            session.commit()

def display_automation_tab():
    """Display automation controls"""
    st.subheader("🤖 Automated Trading")

    automated_trader = get_automated_trader()

    # Get current status
    try:
        status = asyncio.run(automated_trader.get_status())
        is_running = status.get("is_running", False)
    except Exception as e:
        logger.error(f"Error getting automation status: {e}")
        is_running = False
        status = {}

    # Status display
    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        status_text = "🟢 Running" if is_running else "🔴 Stopped"
        st.metric("Automation Status", status_text)

    with status_col2:
        current_positions = status.get("current_positions", 0)
        max_positions = status.get("max_positions", 5)
        st.metric("Positions", f"{current_positions}/{max_positions}")

    with status_col3:
        scan_interval = status.get("scan_interval", 300) / 60
        st.metric("Scan Interval", f"{scan_interval:.0f}min")

    # Settings
    st.markdown("### ⚙️ Automation Settings")

    settings_col1, settings_col2 = st.columns(2)

    with settings_col1:
        new_max_positions = st.number_input("Max Positions", 1, 10, max_positions, key="auto_max_pos")
        new_risk_per_trade = st.number_input("Risk per Trade (%)", 0.5, 5.0, 
                                           status.get("risk_per_trade", 0.02) * 100, 
                                           step=0.1, key="auto_risk")

    with settings_col2:
        new_scan_interval = st.number_input("Scan Interval (minutes)", 1, 60, int(scan_interval), key="auto_interval")
        min_signal_score = st.number_input("Min Signal Score", 50, 90, 65, key="auto_min_score")

    # Control buttons
    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        if st.button("🚀 Start Automation", disabled=is_running):
            # Check trading mode and warn if real
            trading_mode = db_manager.get_setting("trading_mode") or "virtual"
            if trading_mode == "real":
                st.warning("⚠️ REAL MODE: Automation will place LIVE trades on Bybit with real funds!")

            with st.spinner("Starting automation..."):
                try:
                    # Update settings
                    automated_trader.max_positions = new_max_positions
                    automated_trader.risk_per_trade = new_risk_per_trade / 100
                    automated_trader.scan_interval = new_scan_interval * 60

                    success = asyncio.run(automated_trader.start())
                    if success:
                        mode_msg = "REAL" if trading_mode == "real" else "Virtual"
                        st.success(f"✅ {mode_msg} automation started! Will execute multiple trades per scan.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start automation")
                except Exception as e:
                    st.error(f"Start error: {e}")

    with control_col2:
        if st.button("⏹️ Stop Automation", disabled=not is_running, key="stop_auto_btn"):
            with st.spinner("Stopping automation..."):
                try:
                    success = asyncio.run(automated_trader.stop())
                    if success:
                        st.success("✅ Automation stopped!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Failed to stop automation")
                except Exception as e:
                    st.error(f"Stop error: {e}")

    with control_col3:
        if st.button("🔄 Reset Stats"):
            try:
                asyncio.run(automated_trader.reset_stats())
                st.success("✅ Statistics reset!")
                st.rerun()
            except Exception as e:
                st.error(f"Reset error: {e}")

    # Countdown timer with live updates
    st.markdown("### ⏱️ Scan Countdown")
    countdown_placeholder = st.empty()

    try:
        status = asyncio.run(automated_trader.get_status())
        last_scan_str = status['stats'].get('last_scan')
        scan_interval = status.get('scan_interval', 300)

        if is_running and last_scan_str:
            last_scan = datetime.strptime(last_scan_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_scan).total_seconds()
            remaining = max(0, scan_interval - elapsed)

            mins, secs = divmod(int(remaining), 60)
            if remaining > 0:
                countdown_placeholder.info(f"⏳ Next scan in {mins:02d}:{secs:02d} | Last scan: {last_scan_str}")
            else:
                countdown_placeholder.warning("⏳ Scan in progress...")
        elif is_running:
            countdown_placeholder.info("⏳ Initial scan pending...")
        else:
            countdown_placeholder.info("⏸️ Automation stopped")

    except Exception as e:
        countdown_placeholder.error(f"Status error: {e}")

    # Auto-refresh every second when running
    if is_running:
        if 'countdown_key' not in st.session_state:
            st.session_state.countdown_key = 0
        time.sleep(1)
        st.session_state.countdown_key += 1
        st.rerun()

    # Performance summary
    if is_running or status.get("stats", {}).get("total_trades", 0) > 0:
        st.markdown("### 📊 Performance Summary")

        performance = automated_trader.get_performance_summary()

        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

        with perf_col1:
            st.metric("Total Trades", performance.get("total_trades", 0))

        with perf_col2:
            win_rate = performance.get("win_rate", 0)
            st.metric("Win Rate", f"{win_rate}%")

        with perf_col3:
            total_pnl = performance.get("total_pnl", 0)
            st.metric("Total PnL", f"${total_pnl:.2f}")

        with perf_col4:
            runtime = performance.get("runtime", "N/A")
            st.metric("Runtime", runtime)

        # Recent activity
        if is_running:
            st.markdown("### 🕐 Recent Activity")
            recent_trades = db_manager.get_trades(limit=5)

            if recent_trades:
                activity_data = []
                for trade in recent_trades:
                    activity_data.append({
                        "Time": trade.timestamp.strftime("%H:%M:%S") if trade.timestamp else "N/A",
                        "Symbol": trade.symbol,
                        "Side": trade.side,
                        "Entry": f"${trade.entry_price:.4f}",
                        "Status": trade.status.title(),
                        "Type": "Virtual" if trade.virtual else "Real"
                    })

                st.dataframe(pd.DataFrame(activity_data), height=200)
            else:
                st.info("No recent activity")

def main():
    # Apply black background theme
    st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
            color: #ffffff;
        }
        .stSidebar {
            background-color: #1a1a1a;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div>div {
            background-color: #1a1a1a;
            color: #ffffff;
            border: 1px solid #333333;
        }
        .stButton>button {
            background-color: #00ff88;
            color: #000000;
        }
        .stButton>button:hover {
            background-color: #00cc6a;
        }
        .stDataFrame {
            background-color: #1a1a1a;
            border: 1px solid #333333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00ff88 !important;
        }
        .stMetric {
            background-color: #1a1a1a;
            padding: 10px;
            border-radius: 5px;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #000000;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #2a2a2a;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #00ff88;
            color: #000000;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">💼 Trading Center</h1>
        <p style="color: #888; margin: 0;">Complete Trade Management & Automation</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("ℹ️ Trade Management Guide", expanded=False):
        st.markdown("""
        ### 📋 Page Overview

        **Five tabs for complete trade control:**

        1. **🔄 Open Positions**: Monitor and close active trades
        2. **📜 Trade History**: Review all past trades and performance
        3. **📝 Manual Trading**: Place custom trades with your own parameters
        4. **🤖 Automation**: Enable automated signal scanning and execution
        5. **📊 Statistics**: View comprehensive trading metrics

        ### 🔄 Managing Open Positions

        **Virtual Mode:**
        - View all simulated open trades
        - Close positions manually at current market price
        - Track unrealized PnL in real-time
        - TP/SL levels are displayed but not enforced automatically

        **Real Mode:**
        - Syncs with all Bybit positions, including non-bot-initiated trades
        - Close positions to execute real market orders
        - Unrealized PnL reflects actual account value
        - TP/SL orders are placed on Bybit exchange

        ### 📝 Manual Trading

        **Steps to place a manual trade:**
        1. Select trading symbol from dropdown
        2. Choose Long (buy) or Short (sell)
        3. Enter custom Entry, TP, and SL prices
        4. Click "Execute Trade" to place order

        **Tips:**
        - Use current market price as reference
        - Set realistic TP/SL based on volatility
        - In real mode, ensure sufficient balance

        ### 🤖 Automation Setup

        **How automated trading works:**
        1. Set countdown timer (minutes)
        2. Click "Start Automated Trading"
        3. System scans markets when countdown reaches zero
        4. Top signals are auto-executed (virtual or real mode)
        5. Process repeats continuously

        **Controls:**
        - **Start**: Begins countdown and automation
        - **Stop**: Halts all automated activity
        - **Countdown Display**: Shows time until next scan

        **Safety Features:**
        - Max position limits prevent over-trading
        - Risk percentage caps position sizes
        - Emergency stop button available in sidebar

        ### 📊 Understanding Statistics

        - **Win Rate**: Percentage of profitable trades
        - **Total Trades**: Number of completed trades
        - **Average PnL**: Mean profit/loss per trade
        - **Total PnL**: Cumulative profit/loss
        - **Capital Usage**: How much balance is allocated to positions

        ### ⚠️ Important Notes

        - **Virtual trades** don't affect real funds
        - **Real trades** execute on Bybit immediately
        - Always verify mode before trading
        - Use emergency stop if needed
        """)

    st.divider()

    # Initialize engine
    engine = st.session_state.get("engine")
    if not engine:
        engine = get_engine()
        st.session_state.engine = engine

    # Get current trading mode from session state
    trading_mode = st.session_state.get("trading_mode", "virtual")

    # Sidebar
    with st.sidebar:
        st.header("💼 Trading Controls")

        # Show current mode prominently
        mode_color = "🟢" if trading_mode == "virtual" else "🟡"
        st.info(f"{mode_color} **Current Mode:** {trading_mode.title()}")

        st.divider()

        # Quick stats
        try:
            open_virtual = len(engine.get_open_virtual_trades())
            open_real = len(engine.get_open_real_trades())

            st.metric("Open Virtual", open_virtual)
            st.metric("Open Real", open_real)

            # Load balance from DB
            if trading_mode == "virtual":
                wallet_balance = db.get_wallet_balance("virtual")
                capital_val = wallet_balance.capital if wallet_balance else 100.0
                available_val = wallet_balance.available if wallet_balance else 100.0
            else:
                try:
                    result = engine.client._make_request(
                        "GET",
                        "/v5/account/wallet-balance",
                        {"accountType": "UNIFIED"}
                    )
                    if result and "list" in result and result["list"]:
                        wallet = result["list"][0]
                        capital_val = float(wallet.get("totalEquity", 0.0))
                        coins = wallet.get("coin", [])
                        usdt_coin = next((c for c in coins if c.get("coin") == "USDT"), None)
                        available_val = float(usdt_coin.get("walletBalance", 0.0)) if usdt_coin else capital_val
                    else:
                        capital_val = available_val = 0.0
                except Exception as e:
                    logger.error(f"Failed to fetch real balance from Bybit: {e}")
                    capital_val = available_val = 0.0

            available_val = max(available_val, 0.0)
            used_val = max(capital_val - available_val, 0.0)

            if trading_mode == "virtual":
                st.metric("💻 Virtual Capital", f"${capital_val:.2f}")
                st.metric("💻 Virtual Available", f"${available_val:.2f}")
                st.metric("💻 Virtual Used", f"${used_val:.2f}")
            else:
                st.metric("🏦 Real Capital", f"${capital_val:.2f}")
                st.metric("🏦 Real Available", f"${available_val:.2f}")
                st.metric("🏦 Real Used Margin", f"${used_val:.2f}")

        except Exception as e:
            st.error(f"Error loading stats: {e}")

        st.divider()

        # Navigation
        if st.button("📊 Dashboard"):
            st.switch_page("app.py")

        if st.button("🎯 Generate Signals"):
            st.switch_page("pages/signals.py")

        if st.button("📈 Performance"):
            st.switch_page("pages/performance.py")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Open Positions", 
        "📜 Trade History", 
        "📝 Manual Trading", 
        "🤖 Automation", 
        "📊 Statistics"
    ])

    with tab1:
        display_trade_management()

    with tab2:
        st.subheader("📜 Trading History")

        # Get all closed trades
        engine = get_engine()
        closed_trades = engine.get_closed_virtual_trades() + engine.get_closed_real_trades()

        if closed_trades:
            # Convert to displayable format
            history_data = []
            for trade in sorted(closed_trades, key=lambda x: x.timestamp or datetime.min, reverse=True):
                pnl = trade.pnl or 0
                history_data.append({
                    "Date": trade.timestamp.strftime("%Y-%m-%d %H:%M") if trade.timestamp else "N/A",
                    "Symbol": trade.symbol,
                    "Side": trade.side,
                    "Entry": f"${trade.entry_price:.4f}",
                    "Exit": f"${trade.exit_price:.4f}" if trade.exit_price else "N/A",
                    "Qty": f"{trade.qty:.6f}",
                    "SL": f"${trade.sl:.4f}" if trade.sl else "N/A",
                    "TP": f"${trade.tp:.4f}" if trade.tp else "N/A",
                    "Trail": f"${trade.trail:.4f}" if trade.trail else "N/A",
                    "Liquidation": f"${trade.liquidation:.4f}" if trade.liquidation else "N/A",
                    "Margin": f"${trade.margin_usdt:.2f}" if trade.margin_usdt else "N/A",
                    "PnL": f"${pnl:.2f}",
                    "Mode": "Virtual" if trade.virtual else "Real",
                    "Status": "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
                })

            df = pd.DataFrame(history_data)
            
            # Pagination
            if 'history_page' not in st.session_state:
                st.session_state.history_page = 0

            items_per_page = 10
            total_pages = (len(df) - 1) // items_per_page + 1
            start_idx = st.session_state.history_page * items_per_page
            end_idx = start_idx + items_per_page
            page_df = df.iloc[start_idx:end_idx]

            # Display in card grid
            cols = st.columns(3)
            for index, row in page_df.iterrows():
                with cols[index % 3]:
                    st.markdown(f"""
                    <div style="border: 1px solid #262730; border-radius: 10px; padding: 12px; margin-bottom: 10px; background: #1E1E1E;">
                        <h4 style="margin: 0; color: #00ff88;">{row['Symbol']}</h4>
                        <p style="margin: 5px 0; font-size: 13px;"><b>{row['Side']}</b> | {row['Status']}</p>
                        <p style="margin: 5px 0; font-size: 12px;">Entry: {row['Entry']} | Exit: {row['Exit']}</p>
                        <p style="margin: 5px 0; font-size: 12px;">PnL: {row['PnL']}</p>
                        <p style="margin: 5px 0; font-size: 11px; color: #888;">{row['Date']} | {row['Mode']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Pagination controls
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Prev", key="prev_history", disabled=st.session_state.history_page == 0):
                    st.session_state.history_page -= 1
                    st.rerun()
            with col2:
                st.markdown(f"<p style='text-align: center;'>Page {st.session_state.history_page + 1} of {total_pages}</p>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ➡️", key="next_history", disabled=st.session_state.history_page >= total_pages - 1):
                    st.session_state.history_page += 1
                    st.rerun()

            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Export Trading History",
                csv,
                "trading_history.csv",
                "text/csv"
            )
        else:
            st.info("No trading history available. Start trading to see your history here!")

    with tab3:
        display_manual_trading()

    with tab4:
        display_automation_tab()

    with tab5:
        st.subheader("📊 Trading Statistics")

        # Calculate comprehensive stats
        engine = get_engine()
        all_trades = engine.get_closed_virtual_trades() + engine.get_closed_real_trades()

        if all_trades:
            metrics = calculate_portfolio_metrics([t.to_dict() for t in all_trades])

            # Main metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric("Total Trades", metrics['total_trades'])

            with metric_col2:
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

            with metric_col3:
                st.metric("Total PnL", f"${metrics['total_pnl']:.2f}")

            with metric_col4:
                st.metric("Avg PnL/Trade", f"${metrics['avg_pnl']:.2f}")

            # Additional metrics
            st.markdown("### 🎯 Detailed Statistics")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.metric("Profitable Trades", metrics['profitable_trades'])
                st.metric("Best Trade", f"${metrics['best_trade']:.2f}")

            with detail_col2:
                losing_trades = metrics['total_trades'] - metrics['profitable_trades']
                st.metric("Losing Trades", losing_trades)
                st.metric("Worst Trade", f"${metrics['worst_trade']:.2f}")

            # Performance by symbol
            st.markdown("### 📈 Performance by Symbol")

            symbol_performance = {}
            for trade in all_trades:
                symbol = trade.symbol
                pnl = trade.pnl or 0

                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {'trades': 0, 'total_pnl': 0}

                symbol_performance[symbol]['trades'] += 1
                symbol_performance[symbol]['total_pnl'] += pnl

            if symbol_performance:
                symbol_data = []
                for symbol, data in symbol_performance.items():
                    symbol_data.append({
                        "Symbol": symbol,
                        "Trades": data['trades'],
                        "Total PnL": f"${data['total_pnl']:.2f}",
                        "Avg PnL": f"${data['total_pnl'] / data['trades']:.2f}"
                    })

                st.dataframe(pd.DataFrame(symbol_data))
        else:
            st.info("No trading statistics available. Complete some trades to see detailed analytics!")

if __name__ == "__main__":
    main()