import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
import os
from datetime import datetime, timezone, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import calculate_portfolio_metrics

# Configure logging
# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

from db import db_manager

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
        sorted_trades = sorted(trades, key=lambda x: getattr(x, 'timestamp', datetime.min.replace(tzinfo=timezone.utc)))
        
        # Calculate cumulative performance
        equity_data = []
        cumulative_pnl = 100.0  # Starting balance
        
        for trade in sorted_trades:
            pnl = getattr(trade, 'pnl', 0) or 0
            cumulative_pnl += pnl
            timestamp = getattr(trade, 'timestamp', datetime.now(timezone.utc))
            equity_data.append({
                'date': timestamp,
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
        logger.error(f"Error creating equity curve: {e}")
        return None

def create_drawdown_chart(trades):
    """Create a drawdown analysis chart"""
    try:
        if not trades:
            return None
        
        sorted_trades = sorted(trades, key=lambda x: getattr(x, 'timestamp', datetime.min.replace(tzinfo=timezone.utc)))
        
        drawdown_data = []
        current_peak = 100.0  # Starting balance
        running_equity = 100.0
        max_drawdown = 0
        drawdowns = []
        
        for trade in sorted_trades:
            pnl = getattr(trade, 'pnl', 0) or 0
            running_equity += pnl
            current_peak = max(current_peak, running_equity)
            drawdown = ((running_equity - current_peak) / current_peak) * 100 if current_peak > 0 else 0
            drawdowns.append(drawdown)
            max_drawdown = min(max_drawdown, drawdown)
            
            timestamp = getattr(trade, 'timestamp', datetime.now(timezone.utc))
            drawdown_data.append({
                'date': timestamp,
                'drawdown': drawdown,
                'equity': running_equity
            })
        
        if not drawdown_data:
            return None
        
        df = pd.DataFrame(drawdown_data)
        
        fig = go.Figure()
        
        # Drawdown area
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['drawdown'],
            mode='lines',
            name='Drawdown %',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"Portfolio Drawdown (Max: {abs(max_drawdown):.2f}%)",
            xaxis_title="Date",
            yaxis_title="Drawdown %",
            height=400,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating drawdown chart: {e}")
        return None

def create_performance_distribution(trades):
    """Create distribution of trade performance"""
    try:
        if not trades:
            return None
        
        pnls = [getattr(trade, 'pnl', 0) or 0 for trade in trades]
        if not pnls:
            return None
        
        fig = px.histogram(
            pnls,
            nbins=50,
            title="Trade P&L Distribution",
            labels={'value': 'P&L (USDT)'},
            color_discrete_sequence=['#00ff88']
        )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance distribution: {e}")
        return None

def main():
    try:
        # Get engine from session state
        engine = st.session_state.get('engine')
        
        if not engine:
            st.warning("Trading engine not initialized. Please check settings.")
            return
        
        # Get all trades
        all_trades = db_manager.get_trades(limit=1000)
        
        # Filter closed trades
        closed_trades = [t for t in all_trades if t.status == "closed"]
        
        # Separate virtual and real
        virtual_trades = [t for t in closed_trades if t.virtual]
        real_trades = [t for t in closed_trades if not t.virtual]
        
        # Calculate metrics
        metrics = calculate_portfolio_metrics([t.to_dict() for t in closed_trades])
        
        # Header
        st.title("📊 Performance Analysis")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview",
            "📉 Drawdown",
            "📊 Distribution",
            "📅 Time Analysis"
        ])
        
        with tab1:
            st.subheader("Performance Summary")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", metrics.get('total_trades', 0))
            col2.metric("Profitable Trades", metrics.get('profitable_trades', 0))
            col3.metric("Win Rate", f"{metrics.get('win_rate', 0)}%")
            col4.metric("Total P&L", f"${metrics.get('total_pnl', 0):.2f}")
            col5.metric("Avg P&L/Trade", f"${metrics.get('avg_pnl', 0):.2f}")
            
            # Equity curve
            equity_fig = create_equity_curve(closed_trades)
            if equity_fig:
                st.plotly_chart(equity_fig, use_container_width=True)
            else:
                st.info("No trade data available for equity curve")
        
        with tab2:
            st.subheader("Drawdown Analysis")
            
            drawdown_fig = create_drawdown_chart(closed_trades)
            if drawdown_fig:
                st.plotly_chart(drawdown_fig, use_container_width=True)
            else:
                st.info("No data available for drawdown analysis")
            
            # Additional drawdown metrics
            if closed_trades:
                pnls = [getattr(t, 'pnl', 0) or 0 for t in closed_trades]
                if pnls:
                    st.markdown("### 📉 Risk Metrics")
                    col1, col2, col3 = st.columns(3)

                    col1.metric("Largest Loss", f"${min(pnls):.2f}")
                    col2.metric("Largest Win", f"${max(pnls):.2f}")

                    if any(p < 0 for p in pnls):
                        profit_factor = abs(sum(p for p in pnls if p > 0) / sum(abs(p) for p in pnls if p < 0))
                        col3.metric("Profit Factor", f"{profit_factor:.2f}")
                    else:
                        col3.metric("Profit Factor", "∞")

        with tab3:
            st.subheader("Trade Distribution")
            
            dist_fig = create_performance_distribution(closed_trades)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
            else:
                st.info("No data available for distribution analysis")
            
            # Download data
            if closed_trades:
                trades_data = [t.to_dict() for t in closed_trades]
                csv = pd.DataFrame(trades_data).to_csv(index=False)
                st.download_button(
                    "📥 Download Recent Trades",
                    csv,
                    "recent_trades.csv",
                    "text/csv"
                )
        
        with tab4:
            st.subheader("📅 Time-Based Analysis")
            
            # Performance by day of week
            if closed_trades:
                day_performance = {}
                for trade in closed_trades:
                    if getattr(trade, 'timestamp', None):
                        day_name = trade.timestamp.strftime("%A")
                        pnl = getattr(trade, 'pnl', 0) or 0
                        
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
            
            # Monthly performance
            monthly_performance = {}
            for trade in closed_trades:
                if getattr(trade, 'timestamp', None):
                    month_key = trade.timestamp.strftime("%Y-%m")
                    pnl = getattr(trade, 'pnl', 0) or 0
                    
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
        logger.error(f"Performance page error: {e}")
        
        # Error recovery options
        st.markdown("### 🔧 Error Recovery")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Analysis"):
                st.rerun()
        with col2:
            if st.button("📊 Go to Dashboard"):
                st.switch_page("app.py")

if __name__ == "__main__":
    main()