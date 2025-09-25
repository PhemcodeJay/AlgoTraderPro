import streamlit as st
import pandas as pd
import asyncio
import sys
import os
from datetime import datetime, timezone
import json
from typing import Optional, List, Dict
from sqlalchemy import update

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import create_engine
from db import db_manager, TradeModel
from automated_trader import AutomatedTrader
from utils import calculate_portfolio_metrics, format_currency_safe, format_price_safe
from signal_generator import get_usdt_symbols
from logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Trades - AlgoTrader Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_engine():
    """Initialize and cache the trading engine"""
    return create_engine()

@st.cache_resource
def get_automated_trader():
    """Initialize and cache the automated trader"""
    engine = get_engine()
    return AutomatedTrader(engine, engine.client)

def close_trade_safely(trade_id: Optional[str]) -> bool:
    """Close a trade with proper error handling"""
    if not trade_id:
        st.error("🚫 Invalid trade ID provided")
        logger.error("Trade ID is None or empty")
        return False

    try:
        engine = get_engine()
        open_trades = db_manager.get_open_trades()
        trade = next((t for t in open_trades if str(t.id) == str(trade_id) or str(t.order_id) == str(trade_id)), None)
        
        if not trade:
            st.error(f"🚫 Trade {trade_id} not found")
            logger.error(f"Trade {trade_id} not found in open trades")
            return False
        
        current_price = engine.client.get_current_price(trade.symbol)
        if current_price <= 0:
            st.error(f"⚠️ Invalid current price for {trade.symbol}")
            logger.error(f"Invalid current price for {trade.symbol}: {current_price}")
            return False
        
        trade_dict = trade.to_dict()
        trade_dict['exit_price'] = current_price
        pnl = engine.calculate_pnl(trade_dict)
        
        if not trade.virtual:
            opposite_side = "Sell" if trade.side.upper() == "BUY" else "Buy"
            close_order = engine.client.place_order(
                symbol=trade.symbol,
                side=opposite_side,
                qty=trade.qty,
                reduce_only=True
            )
            if not close_order:
                st.error(f"❌ Failed to close position on Bybit for {trade.symbol}")
                logger.error(f"Failed to close Bybit position for {trade.symbol}, order_id: {trade.order_id}")
                return False
        
        if not db_manager.session:
            logger.error("Database session not initialized")
            st.error("🚫 Database session not initialized")
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
            if trade.virtual:
                engine.update_virtual_balances(pnl, mode="virtual")
            else:
                engine.sync_real_balance()
            st.success(f"✅ Trade closed successfully! PnL: {format_currency_safe(pnl)}")
            logger.info(f"Trade closed: {trade.symbol}, PnL: {format_currency_safe(pnl)}, Mode: {'virtual' if trade.virtual else 'real'}")
            return True
        except Exception as e:
            db_manager.session.rollback()
            st.error(f"❌ Failed to close trade in database: {e}")
            logger.error(f"Database error updating trade {trade.order_id}: {e}", exc_info=True)
            return False
    
    except Exception as e:
        st.error(f"❌ Error closing trade: {e}")
        logger.error(f"Error closing trade {trade_id}: {e}", exc_info=True)
        return False

def display_trade_management():
    """Display trade management interface for open trades"""
    st.markdown("### 📈 Open Positions")
    open_trades = db_manager.get_open_trades()
    virtual_trades = [t for t in open_trades if t.virtual]
    real_trades = [t for t in open_trades if not t.virtual]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎮 Virtual Trades")
        if virtual_trades:
            trades_data = [{
                'ID': t.id,
                'Symbol': t.symbol,
                'Side': t.side,
                'Entry': format_price_safe(t.entry_price),
                'Qty': f"{t.qty:.2f}",
                'Leverage': f"{t.leverage}x",
                'Timestamp': t.timestamp.isoformat()[:19]
            } for t in virtual_trades]
            
            st.dataframe(
                pd.DataFrame(trades_data),
                column_config={
                    "Entry": st.column_config.NumberColumn(format="$ %.4f"),
                    "Qty": st.column_config.NumberColumn(format="%.2f"),
                    "Timestamp": st.column_config.DateColumn(format="YYYY-MM-DD HH:mm:ss")
                },
                use_container_width=True
            )
            
            selected_trade = st.selectbox("Select Virtual Trade to Close", 
                                        options=[f"{t['ID']} - {t['Symbol']}" for t in trades_data],
                                        key="virtual_trade_select")
            trade_id = selected_trade.split(" - ")[0] if selected_trade else None
            
            if st.button("❌ Close Selected Virtual Trade", key="close_virtual"):
                with st.spinner("Closing trade..."):
                    if trade_id and close_trade_safely(trade_id):
                        st.rerun()
        else:
            st.info("🌙 No open virtual trades")
    
    with col2:
        st.markdown("#### 💰 Real Trades")
        if real_trades:
            trades_data = [{
                'ID': t.id,
                'Symbol': t.symbol,
                'Side': t.side,
                'Entry': format_price_safe(t.entry_price),
                'Qty': f"{t.qty:.2f}",
                'Leverage': f"{t.leverage}x",
                'Timestamp': t.timestamp.isoformat()[:19]
            } for t in real_trades]
            
            st.dataframe(
                pd.DataFrame(trades_data),
                column_config={
                    "Entry": st.column_config.NumberColumn(format="$ %.4f"),
                    "Qty": st.column_config.NumberColumn(format="%.2f"),
                    "Timestamp": st.column_config.DateColumn(format="YYYY-MM-DD HH:mm:ss")
                },
                use_container_width=True
            )
            
            selected_trade = st.selectbox("Select Real Trade to Close", 
                                        options=[f"{t['ID']} - {t['Symbol']}" for t in trades_data],
                                        key="real_trade_select")
            trade_id = selected_trade.split(" - ")[0] if selected_trade else None
            
            if st.button("❌ Close Selected Real Trade", key="close_real"):
                with st.spinner("Closing trade..."):
                    if trade_id and close_trade_safely(trade_id):
                        st.rerun()
        else:
            st.info("🌙 No open real trades")

def display_trade_history():
    """Display trade history with export option"""
    st.markdown("### 📜 Trade History")
    
    closed_trades = [t for t in db_manager.get_trades(limit=500) if t.status == "closed"]
    if closed_trades:
        history_data = [{
            'Symbol': t.symbol,
            'Side': t.side,
            'Entry': format_price_safe(t.entry_price),
            'Exit': format_price_safe(t.exit_price),
            'PnL': format_currency_safe(t.pnl),
            'Mode': 'Virtual' if t.virtual else 'Real',
            'Opened': t.timestamp.isoformat()[:19],
            'Closed': t.closed_at.isoformat()[:19] if t.closed_at else 'N/A',
            'Status': '✅' if t.pnl and t.pnl > 0 else '❌' if t.pnl and t.pnl < 0 else '➖'
        } for t in closed_trades]
        
        st.dataframe(
            pd.DataFrame(history_data).sort_values('Closed', ascending=False),
            column_config={
                "Entry": st.column_config.NumberColumn(format="$ %.4f"),
                "Exit": st.column_config.NumberColumn(format="$ %.4f"),
                "PnL": st.column_config.NumberColumn(format="$ %.2f"),
                "Opened": st.column_config.DateColumn(format="YYYY-MM-DD HH:mm:ss"),
                "Closed": st.column_config.DateColumn(format="YYYY-MM-DD HH:mm:ss")
            },
            use_container_width=True,
            height=400
        )
        
        csv = pd.DataFrame(history_data).to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History",
            data=csv,
            file_name="trade_history.csv",
            mime="text/csv",
            key="download_history"
        )
    else:
        st.info("🌙 No closed trades in history. Start trading to see your history here!")

def display_manual_trading():
    """Display manual trading interface"""
    st.markdown("### 🖐️ Manual Trade Execution")
    engine = get_engine()
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbols = get_usdt_symbols(limit=50)
        symbol = st.selectbox("Symbol", options=symbols, key="manual_symbol")
        side = st.selectbox("Side", options=["Buy", "Sell"], key="manual_side")
        qty = st.number_input("Quantity", min_value=0.001, value=1.0, step=0.001, key="manual_qty")
    
    with col2:
        current_price = engine.client.get_current_price(symbol)
        st.metric("Current Price", f"{format_price_safe(current_price)}")
        real_mode = st.checkbox("Real Trading Mode", value=False, key="real_mode")
        if real_mode:
            st.warning("⚠️ Real trading enabled - this will use actual funds!")
    
    if st.button("🚀 Execute Trade", key="execute_trade"):
        with st.spinner("Executing trade..."):
            trading_mode = "real" if real_mode else "virtual"
            signal = {
                "symbol": symbol,
                "side": side.upper(),
                "entry": current_price,
                "qty": qty,
                "leverage": engine.settings.get("LEVERAGE", 10)
            }
            success = engine.execute_virtual_trade(signal, trading_mode)
            if success:
                st.success(f"✅ Trade executed successfully in {trading_mode} mode!")
                st.rerun()
            else:
                st.error("❌ Failed to execute trade")
                logger.error(f"Failed to execute {trading_mode} trade for {symbol}")

def display_automation_tab():
    """Display automation control tab"""
    st.markdown("### 🤖 Trading Automation")
    trader = get_automated_trader()
    
    status_container = st.container()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Automation", key="start_automation"):
            with st.spinner("Starting automation..."):
                success = asyncio.run(trader.start(status_container))
                if success:
                    status_container.success("✅ Automation started!")
                else:
                    status_container.error("❌ Failed to start automation")
    
    with col2:
        if st.button("🛑 Stop Automation", key="stop_automation"):
            with st.spinner("Stopping automation..."):
                success = asyncio.run(trader.stop())
                if success:
                    status_container.success("✅ Automation stopped!")
                else:
                    status_container.error("❌ Failed to stop automation")
    
    with col3:
        if st.button("🔄 Refresh Status", key="refresh_status"):
            with st.spinner("Fetching status..."):
                status = asyncio.run(trader.get_status())
                status_container.json(status)
    
    if st.button("🗑️ Reset Statistics", key="reset_stats"):
        trader.stats = {
            "signals_generated": 0,
            "trades_executed": 0,
            "profitable_trades": 0,
            "total_pnl": 0.0,
            "start_time": None,
            "last_scan": None
        }
        st.success("✅ Statistics reset!")
    
    st.markdown("#### 📊 Performance Summary")
    perf = trader.get_performance_summary()
    if perf:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", perf.get('total_trades', 0))
            st.metric("Profitable Trades", perf.get('profitable_trades', 0))
        with col2:
            st.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%")
            st.metric("Runtime", perf.get('runtime', 'N/A'))
        with col3:
            st.metric("Total PnL", format_currency_safe(perf.get('total_pnl')))
            st.metric("Avg PnL", format_currency_safe(perf.get('avg_pnl')))
    else:
        st.info("🌙 No performance data available")

def display_statistics_tab():
    """Display trading statistics tab"""
    st.markdown("### 📊 Trading Statistics")
    closed_trades = [t for t in db_manager.get_trades(limit=1000) if t.status == "closed"]
    
    if closed_trades:
        metrics = calculate_portfolio_metrics([t.to_dict() for t in closed_trades])
        
        st.markdown("#### 📈 Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Trades", metrics['total_trades'])
        
        with metric_col2:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        
        with metric_col3:
            st.metric("Total P&L", format_currency_safe(metrics['total_pnl']))
        
        with metric_col4:
            st.metric("Avg P&L/Trade", format_currency_safe(metrics['avg_pnl']))
        
        st.markdown("#### 🎯 Detailed Statistics")
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.metric("Profitable Trades", metrics['profitable_trades'])
            st.metric("Best Trade", format_currency_safe(metrics['best_trade']))
        
        with detail_col2:
            losing_trades = metrics['total_trades'] - metrics['profitable_trades']
            st.metric("Losing Trades", losing_trades)
            st.metric("Worst Trade", format_currency_safe(metrics['worst_trade']))
        
        st.markdown("#### 📊 Performance by Symbol")
        symbol_performance = {}
        for trade in closed_trades:
            symbol = trade.symbol
            pnl = trade.pnl or 0
            if symbol not in symbol_performance:
                symbol_performance[symbol] = {'trades': 0, 'total_pnl': 0}
            symbol_performance[symbol]['trades'] += 1
            symbol_performance[symbol]['total_pnl'] += pnl
        
        if symbol_performance:
            symbol_data = [{
                "Symbol": symbol,
                "Trades": data['trades'],
                "Total P&L": format_currency_safe(data['total_pnl']),
                "Avg P&L": format_currency_safe(data['total_pnl'] / data['trades'])
            } for symbol, data in symbol_performance.items()]
            
            st.dataframe(
                pd.DataFrame(symbol_data),
                column_config={
                    "Total P&L": st.column_config.NumberColumn(format="$ %.2f"),
                    "Avg P&L": st.column_config.NumberColumn(format="$ %.2f")
                },
                use_container_width=True
            )
    else:
        st.info("🌙 No trading statistics available. Complete some trades to see detailed analytics!")

def main():
    try:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
            <h1 style="color: #00ff88; margin: 0;">💼 Trade Management</h1>
            <p style="color: #888; margin: 0;">Monitor, Execute, and Analyze Your Trades</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Open Positions",
            "📜 History",
            "🖐️ Manual Trading",
            "🤖 Automation",
            "📊 Statistics"
        ])
        
        with tab1:
            display_trade_management()
        
        with tab2:
            display_trade_history()
        
        with tab3:
            display_manual_trading()
        
        with tab4:
            display_automation_tab()
        
        with tab5:
            display_statistics_tab()
        
        # Display connection status
        engine = get_engine()
        connection_status = "✅ Connected" if engine.client and engine.client.is_connected() else "❌ Disconnected"
        st.markdown("---")
        st.metric("API Status", connection_status)
    
    except Exception as e:
        st.error(f"🚫 Trades page error: {e}")
        logger.error(f"Trades page error: {e}", exc_info=True)
        st.markdown("### 🔧 Error Recovery")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry", key="retry"):
                st.rerun()
        with col2:
            if st.button("📊 Go to Dashboard", key="goto_dashboard"):
                st.switch_page("app.py")

if __name__ == "__main__":
    main()