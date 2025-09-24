import streamlit as st
import pandas as pd
import asyncio
import sys
import os
from datetime import datetime, timezone
import json
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import create_engine  # Use create_engine from engine.py
from db import db_manager, TradeModel
from automated_trader import AutomatedTrader
from utils import calculate_portfolio_metrics
from signal_generator import get_usdt_symbols
from sqlalchemy import update

# Initialize database
db = db_manager

# Configure logging
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
    return create_engine()  # Use create_engine to ensure proper initialization

@st.cache_resource
def get_automated_trader():
    engine = get_engine()
    return AutomatedTrader(engine, engine.client)

def close_trade_safely(trade_id: Optional[str]) -> bool:
    """Close a trade with proper error handling"""
    if not trade_id:
        st.error("Invalid trade ID provided")
        logger.error("Trade ID is None or empty")
        return False

    try:
        engine = get_engine()
        
        # Get trade from database
        open_trades = db_manager.get_open_trades()
        trade = next((t for t in open_trades if str(t.id) == str(trade_id) or str(t.order_id) == str(trade_id)), None)
        
        if not trade:
            st.error(f"Trade {trade_id} not found")
            logger.error(f"Trade {trade_id} not found in open trades")
            return False
        
        # Get current price for PnL calculation
        current_price = engine.client.get_current_price(trade.symbol)
        if current_price <= 0:
            st.error(f"Invalid current price for {trade.symbol}")
            logger.error(f"Invalid current price for {trade.symbol}: {current_price}")
            return False
        
        # Calculate PnL
        trade_dict = trade.to_dict()
        trade_dict['exit_price'] = current_price
        pnl = engine.calculate_pnl(trade_dict)
        
        if not trade.virtual:
            # Close position on Bybit for real trades
            opposite_side = "Sell" if trade.side.upper() == "BUY" else "Buy"
            close_order = engine.client.place_order(
                symbol=trade.symbol,
                side=opposite_side,
                qty=trade.qty,
                reduce_only=True
            )
            if not close_order:
                st.error(f"Failed to close position on Bybit for {trade.symbol}")
                logger.error(f"Failed to close Bybit position for {trade.symbol}, order_id: {trade.order_id}")
                return False
        
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
            st.error("Failed to close trade in database")
            return False
        
        if success:
            if trade.virtual:
                # Update virtual balance
                engine.update_virtual_balances(pnl, mode="virtual")
            else:
                # Sync real balance
                engine.sync_real_balance()
            
            st.success(f"✅ Trade closed successfully! PnL: ${pnl:.2f}")
            logger.info(f"Trade closed: {trade.symbol}, PnL: ${pnl:.2f}, Mode: {'virtual' if trade.virtual else 'real'}")
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
    
    open_trades = db_manager.get_open_trades()
    virtual_trades = [t for t in open_trades if t.virtual]
    real_trades = [t for t in open_trades if not t.virtual]
    
    with col1:
        st.subheader("🎮 Virtual Trades")
        
        if virtual_trades:
            for i, trade in enumerate(virtual_trades):
                with st.expander(f"Virtual Trade {i+1}: {trade.symbol}"):
                    st.write(f"Side: {trade.side}")
                    st.write(f"Quantity: {trade.qty}")
                    st.write(f"Entry Price: ${trade.entry_price:.4f}")
                    if st.button(f"Close Virtual Trade {i+1}", key=f"close_virtual_{i}"):
                        close_trade_safely(trade.id)
        else:
            st.info("No open virtual trades")
    
    with col2:
        st.subheader("💰 Real Trades")
        
        if real_trades:
            for i, trade in enumerate(real_trades):
                with st.expander(f"Real Trade {i+1}: {trade.symbol}"):
                    st.write(f"Side: {trade.side}")
                    st.write(f"Quantity: {trade.qty}")
                    st.write(f"Entry Price: ${trade.entry_price:.4f}")
                    if st.button(f"Close Real Trade {i+1}", key=f"close_real_{i}"):
                        close_trade_safely(trade.id)
        else:
            st.info("No open real trades")

def display_manual_trading():
    """Display manual trading interface"""
    engine = get_engine()
    
    st.subheader("🖐️ Manual Trade Execution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox("Symbol", get_usdt_symbols(limit=50))
        side = st.selectbox("Side", ["Buy", "Sell"])
        qty = st.number_input("Quantity", min_value=0.001, value=1.0)
    
    with col2:
        current_price = engine.client.get_current_price(symbol)
        st.metric("Current Price", f"${current_price:.4f}")
        real_mode = st.checkbox("Real Trading Mode", value=False, key="real_mode")
        if real_mode:
            st.warning("⚠️ Real trading enabled - this will use actual funds!")
    
    if st.button("Execute Trade"):
        trading_mode = "real" if real_mode else "virtual"
        signal = {
            "symbol": symbol,
            "side": side,
            "entry": current_price,
            "qty": qty,
            "leverage": engine.settings.get("LEVERAGE", 10)
        }
        success = engine.execute_virtual_trade(signal, trading_mode)
        if success:
            st.success(f"✅ Trade executed successfully in {trading_mode} mode!")
        else:
            st.error("❌ Failed to execute trade")
            logger.error(f"Failed to execute {trading_mode} trade for {symbol}")

def display_automation_tab():
    """Display automation control tab"""
    trader = get_automated_trader()
    
    st.subheader("🤖 Trading Automation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Automation"):
            success = asyncio.run(trader.start())
            if success:
                st.success("Automation started!")
    
    with col2:
        if st.button("Stop Automation"):
            success = asyncio.run(trader.stop())
            if success:
                st.success("Automation stopped!")
    
    with col3:
        if st.button("Refresh Status"):
            status = asyncio.run(trader.get_status())
            st.json(status)
    
    if st.button("Reset Statistics"):
        trader.stats = {
            "signals_generated": 0,
            "trades_executed": 0,
            "profitable_trades": 0,
            "total_pnl": 0.0,
            "start_time": None,
            "last_scan": None
        }
        st.success("Statistics reset!")
    
    st.subheader("Performance Summary")
    perf = trader.get_performance_summary()
    st.json(perf)

def main():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">💼 Trade Management</h1>
        <p style="color: #888; margin: 0;">Monitor, Execute, and Analyze Your Trades</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Open Trades", "📜 History", "🖐️ Manual Trading", "🤖 Automation", "📊 Statistics"])
    
    with tab1:
        st.subheader("📈 Open Positions")
        display_trade_management()
    
    with tab2:
        st.subheader("📜 Trading History")
        
        all_trades = db_manager.get_trades(limit=500)
        closed_trades = [t for t in all_trades if t.status == "closed"]
        
        if closed_trades:
            history_data = []
            for trade in closed_trades:
                pnl = trade.pnl if trade.pnl is not None else 0
                history_data.append({
                    "Symbol": trade.symbol,
                    "Side": trade.side,
                    "Entry": f"${trade.entry_price:.4f}" if trade.entry_price else "N/A",
                    "Exit": f"${trade.exit_price:.4f}" if trade.exit_price else "N/A",
                    "PnL": f"${pnl:.2f}",
                    "Date": trade.timestamp.isoformat() if trade.timestamp else "N/A",
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
        all_trades = db_manager.get_trades(limit=1000)
        closed_trades = [t for t in all_trades if t.status == "closed"]
        
        if closed_trades:
            metrics = calculate_portfolio_metrics([t.to_dict() for t in closed_trades])
            
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
            for trade in closed_trades:
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