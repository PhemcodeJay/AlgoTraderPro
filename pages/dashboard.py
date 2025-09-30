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
from engine import TradingEngine
from utils import (
    get_signals_safe, format_currency_safe, get_trades_safe, 
    get_market_overview_data, calculate_portfolio_metrics
)
from db import db_manager

# Configure logging
# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)


def create_market_overview_chart():
    """Create market overview chart with real data"""
    try:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT"]
        market_data = get_market_overview_data(symbols)
        
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
        virtual_trades = engine.get_closed_virtual_trades()
        real_trades = engine.get_closed_real_trades()
        
        # Calculate cumulative PnL over time
        all_trades = virtual_trades + real_trades
        if not all_trades:
            logger.info("No trades available for portfolio chart")
            return None
        
        # Sort trades by timestamp
        trades_data = []
        cumulative_pnl = 0
        
        for trade in sorted(all_trades, key=lambda x: getattr(x, 'timestamp', None) or datetime.min):
            pnl = getattr(trade, 'pnl', None) or 0
            cumulative_pnl += pnl
            timestamp = getattr(trade, 'timestamp', None) or datetime.now(timezone.utc)
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

def load_capital_data(bybit_client: Optional['BybitClient'] = None) -> dict:
    """Load capital data from database, syncing real balance if Bybit client is connected"""
    default_virtual = {"available": 100.0, "capital": 100.0, "used": 0.0, "start_balance": 100.0}
    default_real = {"available": 0.0, "capital": 0.0, "used": 0.0, "start_balance": 0.0}

    try:
        # --- Virtual balance ---
        virtual_balance = db_manager.get_wallet_balance("virtual")
        if not virtual_balance and os.path.exists("capital.json"):
            try:
                with open("capital.json", "r") as f:
                    json.load(f)  # just validate JSON
                db_manager.migrate_capital_json_to_db()
                virtual_balance = db_manager.get_wallet_balance("virtual")
                logger.info("Virtual capital migrated from capital.json to DB")
            except Exception as e:
                logger.warning(f"Could not migrate capital.json: {e}")

        def fmt_balance(balance, defaults):
            """Helper to safely extract balance fields with rounding"""
            if not balance:
                return defaults
            available = round(float(getattr(balance, "available", defaults["available"])), 2)
            capital = round(float(getattr(balance, "capital", defaults["capital"])), 2)
            used = round(float(getattr(balance, "used", defaults["used"])), 2)
            if abs(used) < 0.01:  # suppress floating point noise
                used = 0.0
            return {
                "available": available,
                "capital": capital,
                "used": used,
                "start_balance": float(getattr(balance, "start_balance", defaults["start_balance"])),
            }

        virtual_data = fmt_balance(virtual_balance, default_virtual)

        # --- Real balance ---
        real_data = default_real.copy()  # fallback defaults

        if bybit_client and bybit_client.is_connected():
            try:
                # Fetch real-time balance from Bybit
                result = bybit_client._make_request(
                    "GET",
                    "/v5/account/wallet-balance",
                    {"accountType": "UNIFIED"}
                )

                if result and "list" in result and result["list"]:
                    wallet = result["list"][0]
                    capital_val = float(wallet.get("totalEquity", 0.0))

                    # Look for USDT balance
                    coins = wallet.get("coin", [])
                    usdt_coin = next((c for c in coins if c.get("coin") == "USDT"), None)
                    available_val = float(usdt_coin.get("walletBalance", 0.0)) if usdt_coin else capital_val

                    # Recalculate used margin
                    used_val = capital_val - available_val
                    if abs(used_val) < 0.01:
                        used_val = 0.0

                    real_data = {
                        "capital": capital_val,
                        "available": max(available_val, 0.0),
                        "used": used_val,
                        "start_balance": capital_val  # optional: could preserve previous start balance if needed
                    }

                    logger.info(f"✅ Real balance fetched: capital=${capital_val:.2f}, available=${available_val:.2f}, used=${used_val:.2f}")

                else:
                    logger.warning("No real account data received from Bybit")
                    st.error("❌ Could not fetch real balance from Bybit.")

            except Exception as e:
                logger.error(f"Failed to fetch real balance from Bybit: {e}", exc_info=True)
                st.error(f"❌ Real balance fetch failed: {e}")

        else:
            logger.debug("Bybit client not connected, skipping real balance fetch")
            if st.session_state.get("trading_mode") == "real":
                st.warning("⚠️ Bybit API not connected. Check API keys in .env file.")

        # Conditional messages
        if real_data["available"] == 0.0 and real_data["capital"] > 0.0:
            st.info("ℹ️ Real available balance is $0.00. Funds may be tied up in open positions.")
        elif real_data["available"] == 0.0 and real_data["capital"] == 0.0 and bybit_client and bybit_client.is_connected():
            st.warning("⚠️ No funds detected in Bybit account. Verify account balance or API permissions.")

        return {"virtual": virtual_data, "real": real_data}


    except Exception as e:
        logger.error(f"Error loading capital data: {e}", exc_info=True)
        return {"virtual": default_virtual, "real": default_real}


def main():
    # Ensure trading mode is initialized
    if "trading_mode" not in st.session_state or st.session_state.trading_mode is None:
        saved_mode = db_manager.get_setting("trading_mode")
        st.session_state.trading_mode = saved_mode if saved_mode in ["virtual", "real"] else "virtual"
        logger.info(f"Initialized trading mode: {st.session_state.trading_mode}")

    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">📊 Dashboard</h1>
        <p style="color: #888; margin: 0;">Market Overview & Trading Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        # Initialize engine and Bybit client
        engine = TradingEngine()
        bybit_client = BybitClient()
        st.session_state.bybit_client = bybit_client  # Store in session state for consistency
        
        # Load capital data
        capital_data = load_capital_data(bybit_client)
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
    
        with col1:
            virtual_balance = capital_data.get("virtual", {}).get("available", 0)
            st.metric("Virtual Balance", f"${format_currency_safe(virtual_balance)}")

        with col2:
            real_balance = capital_data.get("real", {}).get("capital", 0)
            st.metric("Real Capital", f"${format_currency_safe(real_balance)}")

        with col3:
            open_positions = len(engine.get_open_virtual_trades() + engine.get_open_real_trades())
            st.metric("Open Positions", open_positions)
        
        with col4:
            # Calculate win rate from all trades
            all_trades = engine.get_closed_virtual_trades() + engine.get_closed_real_trades()
            try:
                trade_dicts = []
                for trade in all_trades:
                    if hasattr(trade, "to_dict") and callable(trade.to_dict):
                        trade_dicts.append(trade.to_dict())
                    elif isinstance(trade, dict):
                        trade_dicts.append(trade)
                    elif trade is not None:
                        trade_dicts.append(vars(trade))
                    else:
                        trade_dicts.append({})
                metrics = calculate_portfolio_metrics(trade_dicts)
                win_rate = float(metrics.get("win_rate", 0) or 0)
            except Exception:
                metrics = {"win_rate": 0, "total_pnl": 0}
                win_rate = 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col5:
            total_pnl = float(metrics.get('total_pnl', 0) or 0)
            pnl_color = "normal" if total_pnl == 0 else ("inverse" if total_pnl > 0 else "off")
            st.metric("Total PnL", f"${total_pnl:.2f}", delta=f"{total_pnl:+.2f}")
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Overview", "🎯 Recent Signals", "💼 Recent Trades", "⚡ Quick Actions"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Market Performance (24H)")
                market_chart = create_market_overview_chart()
                if market_chart:
                    st.plotly_chart(market_chart)
                else:
                    st.info("Unable to load market data")
            
            with col2:
                st.subheader("Portfolio Performance")
                portfolio_chart = create_portfolio_chart(engine)
                if portfolio_chart:
                    st.plotly_chart(portfolio_chart)
                else:
                    st.info("No trading history available")
        
        with tab2:
            st.subheader("🎯 Latest Trading Signals")
            signals = get_signals_safe(db_manager, limit=10)
            
            if signals:
                signal_data = []
                for signal in signals:
                    # Safe formatting for all values
                    score_val = signal.get('score', 0)
                    entry_val = signal.get('entry', 0)
                    created_val = signal.get("created_at", None)
                    
                    # Handle None values safely
                    score_str = f"{float(score_val or 0):.1f}%" if score_val is not None else "0.0%"
                    entry_str = f"${float(entry_val or 0):.4f}" if entry_val is not None else "$0.0000"
                    created_str = str(created_val)[:19] if created_val is not None else "N/A"
                    
                    signal_data.append({
                        "Symbol": signal.get("symbol", "N/A"),
                        "Side": signal.get("side", "N/A"),
                        "Score": score_str,
                        "Entry": entry_str,
                        "Strategy": signal.get("strategy", "Auto"),
                        "Created": created_str
                    })
                
                df = pd.DataFrame(signal_data)
                st.dataframe(df, height=300)
            else:
                st.info("No recent signals found. Generate new signals to see them here.")
        
        with tab3:
            st.subheader("💼 Recent Trading Activity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Virtual Trades**")
                virtual_trades = get_trades_safe(db_manager, limit=5)
                virtual_only = [t for t in virtual_trades if t.get('virtual', True)]
                
                if virtual_only:
                    vt_data = []
                    for trade in virtual_only:
                        entry_price = trade.get('entry_price', 0)
                        pnl = trade.get('pnl', None)
                        status = trade.get("status", "N/A")
                        
                        # Safe formatting
                        entry_str = f"${float(entry_price):.4f}" if entry_price is not None else "$0.0000"
                        pnl_str = f"${float(pnl):.2f}" if pnl is not None else "Open"
                        status_str = str(status).title() if status is not None else "N/A"
                        
                        vt_data.append({
                            "Symbol": trade.get("symbol", "N/A"),
                            "Side": trade.get("side", "N/A"),
                            "Entry": entry_str,
                            "PnL": pnl_str,
                            "Status": status_str
                        })
                    st.dataframe(pd.DataFrame(vt_data), height=200)
                else:
                    st.info("No virtual trades")
            
            with col2:
                st.markdown("**Real Trades**")
                real_only = [t for t in virtual_trades if not t.get('virtual', True)]
                
                if real_only:
                    rt_data = []
                    for trade in real_only:
                        entry_price = trade.get('entry_price', 0)
                        pnl = trade.get('pnl', None)
                        status = trade.get("status", "N/A")
                        
                        # Safe formatting
                        entry_str = f"${float(entry_price):.4f}" if entry_price is not None else "$0.0000"
                        pnl_str = f"${float(pnl):.2f}" if pnl is not None else "Open"
                        status_str = str(status).title() if status is not None else "N/A"
                        
                        rt_data.append({
                            "Symbol": trade.get("symbol", "N/A"),
                            "Side": trade.get("side", "N/A"),
                            "Entry": entry_str,
                            "PnL": pnl_str,
                            "Status": status_str
                        })
                    st.dataframe(pd.DataFrame(rt_data), height=200)
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
            if st.button("Retry Loading Dashboard"):
                st.rerun()
        with col2:
            if st.button("Go to Settings"):
                st.switch_page("pages/settings.py")

if __name__ == "__main__":
    main()