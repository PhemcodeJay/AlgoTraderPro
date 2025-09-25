from typing import Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json
from datetime import datetime, timezone

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bybit_client import BybitClient
from trading_engine import TradingEngine
from utils import (
    format_currency_safe, 
    get_market_overview_data,
    calculate_portfolio_metrics
)
from db import db_manager
from signal_generator import generate_signals

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

def get_signals_safe(engine, limit=10):
    """Safe way to get recent signals"""
    try:
        symbols = engine.get_usdt_symbols()[:20] if hasattr(engine, 'get_usdt_symbols') else ["BTCUSDT", "ETHUSDT"]
        signals = generate_signals(symbols, interval="60", top_n=limit)
        return signals
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return []

def get_trades_safe(engine, limit=100):
    """Safe way to get trades"""
    try:
        trades = [t.to_dict() for t in db_manager.get_trades(limit=limit)]
        return trades
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return []

def create_market_overview_chart():
    """Create market overview chart with real data"""
    try:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT"]
        market_data = get_market_overview_data(symbols=symbols)
        
        if not market_data:
            logger.warning("No market data returned")
            return None
        
        df = pd.DataFrame(market_data)
        
        # Create bar chart for 24h price changes
        fig = px.bar(
            df,
            x='symbol',
            y='change_24h',
            title='24H Price Changes (%)',
            color='change_24h',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Symbol",
            yaxis_title="24H Change (%)"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating market overview chart: {e}", exc_info=True)
        return None

def create_portfolio_chart(engine):
    """Create portfolio performance chart"""
    try:
        all_trades = db_manager.get_trades(limit=1000)
        
        # Filter closed trades
        closed_trades = [t for t in all_trades if t.status == "closed"]
        virtual_trades = [t for t in closed_trades if t.virtual]
        real_trades = [t for t in closed_trades if not t.virtual]
        
        # Calculate cumulative PnL over time
        all_closed = virtual_trades + real_trades
        if not all_closed:
            logger.info("No closed trades available for portfolio chart")
            return None
        
        # Sort trades by timestamp
        trades_data = []
        cumulative_pnl = 0
        
        for trade in sorted(all_closed, key=lambda x: getattr(x, 'timestamp', datetime.min.replace(tzinfo=timezone.utc))):
            pnl = getattr(trade, 'pnl', 0) or 0
            cumulative_pnl += pnl
            timestamp = getattr(trade, 'timestamp', datetime.now(timezone.utc))
            virtual = getattr(trade, 'virtual', True)
            trades_data.append({
                'date': timestamp,
                'pnl': pnl,
                'cumulative_pnl': cumulative_pnl,
                'type': 'Virtual' if virtual else 'Real'
            })
        
        if not trades_data:
            logger.info("No trade data after processing")
            return None
        
        df = pd.DataFrame(trades_data)
        
        fig = go.Figure()
        
        # Add cumulative PnL line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['cumulative_pnl'],
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='#00ff88', width=3)
        ))
        
        # Add individual trade points
        colors = ['green' if pnl > 0 else 'red' for pnl in df['pnl']]
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['pnl'],
            mode='markers',
            name='Individual Trades',
            marker=dict(color=colors, size=8),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Cumulative PnL (USDT)',
            yaxis2=dict(
                title='Trade PnL (USDT)',
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating portfolio chart: {e}", exc_info=True)
        return None

def main():
    try:
        # Get engine and client from session state
        engine = st.session_state.get('engine')
        bybit_client = st.session_state.get('bybit_client')
        
        if not engine or not bybit_client:
            st.warning("Trading engine or Bybit client not initialized. Please check settings.")
            return
        
        # Get current mode
        mode = st.session_state.get('trading_mode', 'virtual')
        
        # Get signals and trades
        signals = get_signals_safe(engine)
        trades = get_trades_safe(engine)
        
        # Calculate metrics
        portfolio_metrics = calculate_portfolio_metrics(trades)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Signals", "📈 Trades", "💼 Portfolio", "⚡ Actions"])
        
        with tab1:
            st.subheader("Recent Signals")
            
            if signals:
                signals_df = pd.DataFrame([{
                    'Symbol': s.get('symbol'),
                    'Score': round(s.get('score', 0), 1),
                    'Side': s.get('side'),
                    'Entry': format_currency_safe(s.get('entry')),
                    'SL': format_currency_safe(s.get('sl')),
                    'TP': format_currency_safe(s.get('tp')),
                    'Market': s.get('market', 'N/A'),
                    'Leverage': s.get('leverage', 10)
                } for s in signals])
                
                st.dataframe(
                    signals_df.sort_values('Score', ascending=False),
                    column_config={
                        "Score": st.column_config.NumberColumn(format="%.1f"),
                        "Entry": st.column_config.NumberColumn(format="$ %.4f"),
                        "SL": st.column_config.NumberColumn(format="$ %.4f"),
                        "TP": st.column_config.NumberColumn(format="$ %.4f"),
                        "Leverage": st.column_config.NumberColumn(format="%d x")
                    },
                    use_container_width=True
                )
            else:
                st.info("No signals available")
        
        with tab2:
            st.subheader("Market Overview")
            market_fig = create_market_overview_chart()
            if market_fig:
                st.plotly_chart(market_fig, use_container_width=True)
            else:
                st.info("No market data available")
            
            st.subheader("Recent Trades")
            if trades:
                trades_df = pd.DataFrame([{
                    'Symbol': t.get('symbol'),
                    'Side': t.get('side'),
                    'Entry': format_currency_safe(t.get('entry_price')),
                    'PnL': format_currency_safe(t.get('pnl')) if t.get('status') == 'closed' else 'Open',
                    'Status': t.get('status'),
                    'Mode': 'Virtual' if t.get('virtual') else 'Real',
                    'Timestamp': t.get('timestamp')[:19] if t.get('timestamp') else 'N/A'
                } for t in trades])
                
                st.dataframe(
                    trades_df.sort_values('Status', ascending=False),
                    column_config={
                        "Entry": st.column_config.NumberColumn(format="$ %.4f"),
                        "PnL": st.column_config.NumberColumn(format="$ %.2f"),
                        "Timestamp": st.column_config.DateColumn(format="YYYY-MM-DD HH:mm:ss")
                    },
                    use_container_width=True
                )
            else:
                st.info("No trades available")
        
        with tab3:
            st.subheader("Portfolio Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", portfolio_metrics.get('total_trades', 0))
            col2.metric("Win Rate", f"{portfolio_metrics.get('win_rate', 0):.1f}%")
            col3.metric("Total PnL", f"${format_currency_safe(portfolio_metrics.get('total_pnl', 0))}")
            col4.metric("Avg PnL/Trade", f"${format_currency_safe(portfolio_metrics.get('avg_pnl', 0))}")
            
            portfolio_fig = create_portfolio_chart(engine)
            if portfolio_fig:
                st.plotly_chart(portfolio_fig, use_container_width=True)
            else:
                st.info("No portfolio data available")
            
            st.subheader("Trade Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Virtual Trades**")
                vt_data = []
                virtual_only = [t for t in trades if t.get('virtual', True)]
                
                if virtual_only:
                    for trade in virtual_only:
                        entry_price = trade.get('entry_price', 0)
                        pnl = trade.get('pnl', None)
                        status = trade.get("status", "N/A")
                        
                        # Safe formatting
                        entry_str = f"${format_currency_safe(entry_price)}"
                        pnl_str = f"${format_currency_safe(pnl)}" if pnl is not None else "Open"
                        status_str = str(status).title() if status is not None else "N/A"
                        timestamp_str = trade.get('timestamp')[:19] if trade.get('timestamp') else 'N/A'
                        
                        vt_data.append({
                            "Symbol": trade.get("symbol", "N/A"),
                            "Side": trade.get("side", "N/A"),
                            "Entry": entry_str,
                            "PnL": pnl_str,
                            "Status": status_str,
                            "Timestamp": timestamp_str
                        })
                    st.dataframe(
                        pd.DataFrame(vt_data),
                        column_config={
                            "Entry": st.column_config.NumberColumn(format="$ %.4f"),
                            "PnL": st.column_config.NumberColumn(format="$ %.2f"),
                            "Timestamp": st.column_config.DateColumn(format="YYYY-MM-DD HH:mm:ss")
                        },
                        height=200,
                        use_container_width=True
                    )
                else:
                    st.info("No virtual trades")
            
            with col2:
                st.markdown("**Real Trades**")
                real_only = [t for t in trades if not t.get('virtual', True)]
                
                if real_only:
                    rt_data = []
                    for trade in real_only:
                        entry_price = trade.get('entry_price', 0)
                        pnl = trade.get('pnl', None)
                        status = trade.get("status", "N/A")
                        
                        # Safe formatting
                        entry_str = f"${format_currency_safe(entry_price)}"
                        pnl_str = f"${format_currency_safe(pnl)}" if pnl is not None else "Open"
                        status_str = str(status).title() if status is not None else "N/A"
                        timestamp_str = trade.get('timestamp')[:19] if trade.get('timestamp') else 'N/A'
                        
                        rt_data.append({
                            "Symbol": trade.get("symbol", "N/A"),
                            "Side": trade.get("side", "N/A"),
                            "Entry": entry_str,
                            "PnL": pnl_str,
                            "Status": status_str,
                            "Timestamp": timestamp_str
                        })
                    st.dataframe(
                        pd.DataFrame(rt_data),
                        column_config={
                            "Entry": st.column_config.NumberColumn(format="$ %.4f"),
                            "PnL": st.column_config.NumberColumn(format="$ %.2f"),
                            "Timestamp": st.column_config.DateColumn(format="YYYY-MM-DD HH:mm:ss")
                        },
                        height=200,
                        use_container_width=True
                    )
                else:
                    st.info("No real trades")
        
        with tab4:
            st.subheader("⚡ Quick Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🎯 Trading")
                if st.button("Generate Signals"):
                    st.switch_page("pages/signals.py")
                if st.button("View All Trades"):
                    st.switch_page("pages/trades.py")
            
            with col2:
                st.markdown("### 📊 Analysis")
                if st.button("Performance Report"):
                    st.switch_page("pages/performance.py")
                if st.button("Trading Settings"):
                    st.switch_page("pages/settings.py")
            
            with col3:
                st.markdown("### ⚙️ System")
                if st.button("Refresh Data"):
                    st.cache_data.clear()
                    st.session_state.wallet_cache.clear()
                    if st.session_state.get("trading_mode") == "real":
                        engine.sync_real_balance()
                    st.rerun()
                
                # Connection status
                connection_status = "✅ Connected" if bybit_client and bybit_client.is_connected() else "❌ Disconnected"
                st.metric("API Status", connection_status)
        
        # Footer with system info
        st.markdown("---")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Trading Mode", st.session_state.get('trading_mode', 'virtual').title())
        with info_col2:
            st.metric("Active Signals", len(signals) if signals else 0)
        with info_col3:
            st.metric("System Status", "🟢 Online")
    
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}", exc_info=True)
        
        # Show basic error recovery options
        st.markdown("### 🔧 Error Recovery")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retry"):
                st.rerun()
        with col2:
            if st.button("Go to Settings"):
                st.switch_page("pages/settings.py")

if __name__ == "__main__":
    main()