import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_engine import TradingEngine
from db import db_manager, WalletBalance
from utils import calculate_portfolio_metrics
from settings import load_settings

# Configure logging
from logging_config import get_logger
logger = get_logger(__name__)

st.set_page_config(
    page_title="Performance - AlgoTrader Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_equity_curve(trades):
    """Create an equity curve chart from trade history"""
    try:
        if not trades:
            return None
            
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x.timestamp or datetime.min)
        
        # Calculate cumulative performance
        settings = load_settings()
        initial_balance = settings.get("VIRTUAL_BALANCE", 100.0)
        equity_data = []
        cumulative_pnl = initial_balance
        
        for trade in sorted_trades:
            pnl = trade.pnl or 0
            cumulative_pnl += pnl
            equity_data.append({
                'date': trade.timestamp,
                'pnl': pnl,
                'equity': cumulative_pnl,
                'trade_number': len(equity_data) + 1
            })
        
        if not equity_data:
            return None
        
        df = pd.DataFrame(equity_data)
        
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff88', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ))
        
        # Add individual trade markers
        colors = ['green' if pnl > 0 else 'red' for pnl in df['pnl']]
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['equity'],
            mode='markers',
            name='Trades',
            marker=dict(
                color=colors,
                size=8,
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            text=[f"Trade #{row['trade_number']}<br>PnL: ${row['pnl']:.2f}" for _, row in df.iterrows()],
            hovertemplate='%{text}<br>Date: %{x}<br>Portfolio: $%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USDT)",
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating equity curve: {e}", exc_info=True)
        return None

def create_drawdown_chart(trades):
    """Create a drawdown analysis chart"""
    try:
        if not trades:
            return None
        
        sorted_trades = sorted(trades, key=lambda x: x.timestamp or datetime.min)
        
        settings = load_settings()
        initial_balance = settings.get("VIRTUAL_BALANCE", 100.0)
        drawdown_data = []
        cumulative_pnl = initial_balance
        peak_value = initial_balance
        
        for trade in sorted_trades:
            pnl = trade.pnl or 0
            cumulative_pnl += pnl
            peak_value = max(peak_value, cumulative_pnl)
            drawdown = ((cumulative_pnl - peak_value) / peak_value) * 100 if peak_value != 0 else 0
            
            drawdown_data.append({
                'date': trade.timestamp,
                'drawdown': drawdown,
                'equity': cumulative_pnl
            })
        
        if not drawdown_data:
            return None
        
        df = pd.DataFrame(drawdown_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['drawdown'],
            mode='lines',
            name='Drawdown (%)',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
        
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating drawdown chart: {e}", exc_info=True)
        return None

def main():
    """Main performance page content"""
    try:
        engine = st.session_state.engine
        if not engine:
            st.error("Trading engine not initialized")
            logger.error("Trading engine not initialized")
            return

        st.markdown("### 📈 Performance Overview")
        
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

        # --- Main content ---
        all_trades = [trade.to_dict() for trade in engine.db.get_trades(limit=1000)]
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Equity Curve",
            "📉 Drawdown Analysis",
            "📊 Trade Statistics",
            "📅 Time-Based Analysis"
        ])
        
        with tab1:
            st.subheader("📈 Equity Curve")
            equity_chart = create_equity_curve(engine.db.get_trades(limit=1000))
            if equity_chart:
                st.plotly_chart(equity_chart, use_container_width=True)
            else:
                st.info("No trade data available for equity curve.")
        
        with tab2:
            st.subheader("📉 Drawdown Analysis")
            drawdown_chart = create_drawdown_chart(engine.db.get_trades(limit=1000))
            if drawdown_chart:
                st.plotly_chart(drawdown_chart, use_container_width=True)
            else:
                st.info("No trade data available for drawdown analysis.")
        
        with tab3:
            st.subheader("📊 Trade Statistics")
            
            if all_trades:
                metrics = calculate_portfolio_metrics(all_trades)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Trades", metrics["total_trades"])
                col2.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                col3.metric("Total PnL", f"${metrics['total_pnl']:.2f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Avg PnL/Trade", f"${metrics['avg_pnl']:.2f}")
                col5.metric("Best Trade", f"${metrics['best_trade']:.2f}")
                col6.metric("Worst Trade", f"${metrics['worst_trade']:.2f}")
                
                # Recent trades table
                recent_trades = engine.db.get_trades(limit=10)
                recent_data = []
                for trade in recent_trades:
                    recent_data.append({
                        "Symbol": trade.symbol or "N/A",
                        "Side": trade.side or "N/A",
                        "Entry Price": f"${trade.entry_price:.4f}" if trade.entry_price else "$0.0000",
                        "PnL": f"${trade.pnl:.2f}" if trade.pnl else "Open",
                        "Status": trade.status.title() if trade.status else "N/A",
                        "Timestamp": trade.timestamp.strftime("%Y-%m-%d %H:%M:%S") if trade.timestamp else "N/A"
                    })
                
                recent_df = pd.DataFrame(recent_data)
                st.dataframe(recent_df, height=400)
                
                # Export option
                csv = recent_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Recent Trades",
                    csv,
                    "recent_trades.csv",
                    "text/csv"
                )
        
        with tab4:
            st.subheader("📅 Time-Based Analysis")
            
            if all_trades:
                day_performance = {}
                for trade in all_trades:
                    if trade.get("timestamp"):
                        day_name = datetime.fromisoformat(trade["timestamp"]).strftime("%A")
                        pnl = trade.get("pnl", 0) or 0
                        if day_name not in day_performance:
                            day_performance[day_name] = []
                        day_performance[day_name].append(pnl)
                
                if day_performance:
                    st.markdown("### 📅 Performance by Day of Week")
                    day_data = []
                    for day, pnls in day_performance.items():
                        day_data.append({
                            "Day": day,
                            "Trades": len(pnls),
                            "Total P&L": f"${sum(pnls):.2f}",
                            "Avg P&L": f"${np.mean(pnls):.2f}",
                            "Win Rate": f"{(len([p for p in pnls if p > 0]) / len(pnls)) * 100:.1f}%"
                        })
                    day_df = pd.DataFrame(day_data)
                    st.dataframe(day_df)
            
            monthly_performance = {}
            for trade in all_trades:
                if trade.get("timestamp"):
                    month_key = datetime.fromisoformat(trade["timestamp"]).strftime("%Y-%m")
                    pnl = trade.get("pnl", 0) or 0
                    if month_key not in monthly_performance:
                        monthly_performance[month_key] = []
                    monthly_performance[month_key].append(pnl)
            
            if monthly_performance:
                st.markdown("### 📊 Monthly Performance")
                monthly_data = []
                for month, pnls in sorted(monthly_performance.items()):
                    monthly_data.append({
                        "Month": month,
                        "Trades": len(pnls),
                        "Total P&L": f"${sum(pnls):.2f}",
                        "Avg P&L": f"${np.mean(pnls):.2f}",
                        "Best Trade": f"${max(pnls):.2f}",
                        "Worst Trade": f"${min(pnls):.2f}"
                    })
                monthly_df = pd.DataFrame(monthly_data)
                st.dataframe(monthly_df)
    
    except Exception as e:
        st.error(f"Error loading performance data: {e}")
        logger.error(f"Performance page error: {e}", exc_info=True)
        
        # Error recovery options
        st.markdown("### 🔧 Error Recovery")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Analysis"):
                st.rerun()
        with col2:
            if st.button("📊 Go to Dashboard"):
                st.switch_page("pages/dashboard.py")

if __name__ == "__main__":
    main()