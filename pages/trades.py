import streamlit as st
import pandas as pd
import asyncio
import sys
import os
from datetime import datetime, timezone
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import TradingEngine
from db import db_manager, TradeModel
from automated_trader import AutomatedTrader
from utils import calculate_portfolio_metrics
from signal_generator import get_usdt_symbols
from sqlalchemy import update

# Initialize database
db = db_manager

# Configure logging
# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)


st.set_page_config(
    page_title="Trades - AlgoTrader Pro",
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

def close_trade_safely(trade_id: str, virtual: bool = True):
    """Close a trade with proper error handling"""
    try:
        engine = get_engine()
        
        # Get trade from database
        open_trades = [t for t in db_manager.get_trades(limit=1000) if t.status == "open"]
        trade = next((t for t in open_trades if str(t.id) == str(trade_id) or str(t.order_id) == str(trade_id)), None)
        
        if not trade:
            st.error(f"Trade {trade_id} not found")
            return False
        
        # Get current price for PnL calculation
        current_price = engine.client.get_current_price(trade.symbol)
        
        # Calculate PnL
        pnl = engine.calculate_virtual_pnl(trade.to_dict())
        
        # Update trade in database
        if not db_manager.session:
            logger.error("Database session not initialized")
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
            success = False
        
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
    """Display trade management interface"""
    engine = get_engine()
    
    # Trading mode switch
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎮 Virtual Trades")
        virtual_trades = engine.get_open_virtual_trades()
        
        if virtual_trades:
            for i, trade in enumerate(virtual_trades):
                with st.expander(f"{trade.symbol} {trade.side} - ${trade.entry_price:.4f}"):
                    col_a, col_b, col_c = st.columns([2, 2, 1])
                    
                    with col_a:
                        st.write(f"**Quantity:** {trade.qty}")
                        st.write(f"**Score:** {trade.score or 0:.1f}%")
                    
                    with col_b:
                        current_pnl = engine.calculate_virtual_pnl(trade.to_dict())
                        pnl_color = "🟢" if current_pnl > 0 else "🔴" if current_pnl < 0 else "🟡"
                        st.write(f"**Current PnL:** {pnl_color} ${current_pnl:.2f}")
                        st.write(f"**Status:** {trade.status.title()}")
                    
                    with col_c:
                        if st.button("❌ Close", key=f"close_virtual_{trade.id}"):
                            if close_trade_safely(str(trade.id), virtual=True):
                                st.rerun()
        else:
            st.info("No open virtual trades")
    
    with col2:
        st.subheader("💰 Real Trades")
        real_trades = engine.get_open_real_trades()
        
        if real_trades:
            for i, trade in enumerate(real_trades):
                with st.expander(f"{trade.symbol} {trade.side} - ${trade.entry_price:.4f}"):
                    col_a, col_b, col_c = st.columns([2, 2, 1])
                    
                    with col_a:
                        st.write(f"**Quantity:** {trade.qty}")
                        st.write(f"**Score:** {trade.score or 0:.1f}%")
                    
                    with col_b:
                        # For real trades, PnL calculation might be different
                        current_pnl = trade.pnl or 0
                        pnl_color = "🟢" if current_pnl > 0 else "🔴" if current_pnl < 0 else "🟡"
                        st.write(f"**Current PnL:** {pnl_color} ${current_pnl:.2f}")
                        st.write(f"**Status:** {trade.status.title()}")
                    
                    with col_c:
                        if st.button("❌ Close", key=f"close_real_{trade.id}"):
                            if close_trade_safely(str(trade.id), virtual=False):
                                st.rerun()
        else:
            st.info("No open real trades")

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
        
    with col2:
        order_type = st.selectbox("Order Type", ["Market", "Limit"], key="manual_order_type")
        price = st.number_input("Price (for Limit orders)", min_value=0.0, key="manual_price") if order_type == "Limit" else None
        leverage = st.number_input("Leverage", min_value=1, max_value=100, value=10, key="manual_leverage")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        stop_loss = st.number_input("Stop Loss Price", min_value=0.0, key="manual_sl")
        take_profit = st.number_input("Take Profit Price", min_value=0.0, key="manual_tp")
        
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
            
            # Create trade data
            trade_data = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_price": entry_price,
                "order_id": f"manual_{symbol}_{int(datetime.now().timestamp())}",
                "virtual": trading_mode == "virtual",
                "status": "open",
                "strategy": "Manual",
                "leverage": leverage
            }
            
            # Add to database
            success = db_manager.add_trade(trade_data)
            
            if success:
                st.success(f"✅ {trading_mode.title()} order placed: {symbol} {side} @ ${entry_price:.4f}")
                
                # Update balance for virtual trades
                if trading_mode == "virtual":
                    margin_used = (entry_price * qty) / leverage
                    engine.update_virtual_balances(-margin_used, "virtual")  # Reserve margin
                
                st.rerun()
            else:
                st.error("❌ Failed to place order")
                
        except Exception as e:
            st.error(f"Order placement error: {e}")
            logger.error(f"Manual order error: {e}")

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
            with st.spinner("Starting automation..."):
                try:
                    # Update settings
                    automated_trader.max_positions = new_max_positions
                    automated_trader.risk_per_trade = new_risk_per_trade / 100
                    automated_trader.scan_interval = new_scan_interval * 60
                    
                    success = asyncio.run(automated_trader.start())
                    if success:
                        st.success("✅ Automation started!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start automation")
                except Exception as e:
                    st.error(f"Start error: {e}")
    
    with control_col2:
        if st.button("⏹️ Stop Automation", disabled=not is_running):
            with st.spinner("Stopping automation..."):
                try:
                    success = asyncio.run(automated_trader.stop())
                    if success:
                        st.success("✅ Automation stopped!")
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
            delta_color = "normal" if total_pnl == 0 else ("inverse" if total_pnl > 0 else "off")
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
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">💼 Trading Center</h1>
        <p style="color: #888; margin: 0;">Complete Trade Management & Automation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("💼 Trading Controls")
        
        # Current trading mode display
        current_mode = st.session_state.get('trading_mode', 'virtual')
        st.metric("Current Mode", current_mode.title())
        
        # Quick stats
        try:
            engine = get_engine()
            open_virtual = len(engine.get_open_virtual_trades())
            open_real = len(engine.get_open_real_trades())
            
            st.metric("Open Virtual", open_virtual)
            st.metric("Open Real", open_real)
            
            # Load balance from DB
            wallet_balance = db.get_wallet_balance(current_mode)

            capital_val = wallet_balance.capital if wallet_balance else 0.0
            available_val = wallet_balance.available if wallet_balance else 0.0
            used_val = wallet_balance.used if wallet_balance else 0.0

            if current_mode == "virtual":
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
                    "PnL": f"${pnl:.2f}",
                    "Mode": "Virtual" if trade.virtual else "Real",
                    "Strategy": trade.strategy or "Manual",
                    "Status": "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, height=500)
            
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
                st.metric("Total P&L", f"${metrics['total_pnl']:.2f}")
            
            with metric_col4:
                st.metric("Avg P&L/Trade", f"${metrics['avg_pnl']:.2f}")
            
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