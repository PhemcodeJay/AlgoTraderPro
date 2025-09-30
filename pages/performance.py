import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import sys
import os
from datetime import datetime, timedelta

from check_licenses import check_license
license_result = check_license()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import TradingEngine
from utils import calculate_portfolio_metrics

# Configure logging
# Logging using centralized system
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
        equity_data = []
        cumulative_pnl = 100.0  # Starting balance
        
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
        logger.error(f"Error creating equity curve: {e}")
        return None

def create_drawdown_chart(trades):
    """Create a drawdown analysis chart"""
    try:
        if not trades:
            return None
        
        sorted_trades = sorted(trades, key=lambda x: x.timestamp or datetime.min)
        
        drawdown_data = []
        cumulative_pnl = 1000.0
        peak_value = 1000.0
        
        for trade in sorted_trades:
            pnl = trade.pnl or 0
            cumulative_pnl += pnl
            
            # Update peak
            if cumulative_pnl > peak_value:
                peak_value = cumulative_pnl
            
            # Calculate drawdown
            drawdown = (cumulative_pnl - peak_value) / peak_value * 100
            
            drawdown_data.append({
                'date': trade.timestamp,
                'equity': cumulative_pnl,
                'peak': peak_value,
                'drawdown': drawdown
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
            name='Drawdown (%)',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300,
            yaxis=dict(ticksuffix="%")
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating drawdown chart: {e}")
        return None

def create_returns_distribution(trades):
    """Create returns distribution histogram"""
    try:
        if not trades:
            return None
        
        returns = [trade.pnl or 0 for trade in trades]
        
        if not returns:
            return None
        
        fig = px.histogram(
            x=returns,
            nbins=20,
            title="Trade Returns Distribution",
            labels={'x': 'PnL (USDT)', 'y': 'Frequency'},
            color_discrete_sequence=['#00ff88']
        )
        
        # Add vertical lines for mean and median
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        
        fig.add_vline(x=mean_return, line_dash="dash", line_color="orange", 
                     annotation_text=f"Mean: ${mean_return:.2f}")
        fig.add_vline(x=median_return, line_dash="dot", line_color="blue",
                     annotation_text=f"Median: ${median_return:.2f}")
        
        fig.update_layout(height=400)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating returns distribution: {e}")
        return None

def create_performance_metrics_table(trades):
    """Create a comprehensive performance metrics table"""
    try:
        if not trades:
            return pd.DataFrame()
        
        trade_dicts = [trade.to_dict() for trade in trades]
        metrics = calculate_portfolio_metrics(trade_dicts)
        
        returns = [trade.pnl or 0 for trade in trades]
        
        # Calculate additional metrics
        if len(returns) > 1:
            volatility = np.std(returns)
            sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            cumulative = np.cumsum([1000] + returns)  # Starting with 1000
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max * 100
            max_drawdown = np.min(drawdown)
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for pnl in returns:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif pnl < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_wins = 0
                consecutive_losses = 0
        
        # Create metrics table
        performance_data = {
            "Metric": [
                "Total Trades",
                "Profitable Trades", 
                "Losing Trades",
                "Win Rate (%)",
                "Total P&L (USDT)",
                "Average P&L per Trade",
                "Best Trade",
                "Worst Trade",
                "Volatility",
                "Sharpe Ratio",
                "Maximum Drawdown (%)",
                "Max Consecutive Wins",
                "Max Consecutive Losses",
                "Profit Factor"
            ],
            "Value": [
                metrics['total_trades'],
                metrics['profitable_trades'],
                metrics['total_trades'] - metrics['profitable_trades'],
                f"{metrics['win_rate']:.1f}%",
                f"${metrics['total_pnl']:.2f}",
                f"${metrics['avg_pnl']:.2f}",
                f"${metrics['best_trade']:.2f}",
                f"${metrics['worst_trade']:.2f}",
                f"{volatility:.2f}",
                f"{sharpe_ratio:.2f}",
                f"{max_drawdown:.1f}%",
                max_consecutive_wins,
                max_consecutive_losses,
                f"{abs(sum([r for r in returns if r > 0]) / sum([r for r in returns if r < 0])):.2f}" if any(r < 0 for r in returns) else "N/A"
            ]
        }
        
        return pd.DataFrame(performance_data)
        
    except Exception as e:
        logger.error(f"Error creating performance metrics: {e}")
        return pd.DataFrame()

def main():
    is_valid, result = check_license()
    if not is_valid:
        st.stop()
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">📈 Performance Analytics</h1>
        <p style="color: #888; margin: 0;">Comprehensive Trading Performance Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        engine = TradingEngine()
        
        # Sidebar controls
        with st.sidebar:
            st.header("📈 Performance Controls")
            
            # Time period filter
            time_period = st.selectbox(
                "Analysis Period", 
                ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
            )
            
            # Account type filter
            account_filter = st.selectbox(
                "Account Type", 
                ["All Accounts", "Virtual Only", "Real Only"]
            )
            
            # Display options
            st.subheader("📊 Display Options")
            show_equity_curve = st.checkbox("Equity Curve", value=True)
            show_drawdown = st.checkbox("Drawdown Analysis", value=True)
            show_distribution = st.checkbox("Returns Distribution", value=True)
            show_metrics = st.checkbox("Detailed Metrics", value=True)
            
            st.divider()
            
            # Export options
            st.subheader("📤 Export Options")
            if st.button("📊 Generate Report"):
                st.info("📊 Report generation feature coming soon!")
            
            if st.button("📥 Export Data"):
                st.info("📥 Data export feature coming soon!")
            
            st.divider()
            
            if st.button("📊 Back to Dashboard"):
                st.switch_page("app.py")
        
        # Get trades based on filters
        if account_filter == "Virtual Only":
            all_trades = engine.get_closed_virtual_trades()
        elif account_filter == "Real Only":
            all_trades = engine.get_closed_real_trades()
        else:
            all_trades = engine.get_closed_virtual_trades() + engine.get_closed_real_trades()
        
        # Apply time filter
        if time_period != "All Time" and all_trades:
            now = datetime.now()
            if time_period == "Last 7 Days":
                cutoff = now - timedelta(days=7)
            elif time_period == "Last 30 Days":
                cutoff = now - timedelta(days=30)
            elif time_period == "Last 90 Days":
                cutoff = now - timedelta(days=90)
            else:
                cutoff = datetime.min
            
            all_trades = [t for t in all_trades if t.timestamp and t.timestamp >= cutoff]
        
        if not all_trades:
            st.info("📊 No trading data available for the selected filters. Start trading to see performance analytics!")
            return
        
        # Calculate overall metrics
        trade_dicts = [trade.to_dict() for trade in all_trades]
        overall_metrics = calculate_portfolio_metrics(trade_dicts)
        
        # Key performance indicators
        st.subheader("🎯 Key Performance Indicators")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        
        with kpi_col1:
            st.metric("Total Trades", overall_metrics['total_trades'])
        
        with kpi_col2:
            win_rate = overall_metrics['win_rate']
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with kpi_col3:
            total_pnl = overall_metrics['total_pnl']
            pnl_delta = f"{total_pnl:+.2f}" if total_pnl != 0 else None
            st.metric("Total P&L", f"${total_pnl:.2f}", delta=pnl_delta)
        
        with kpi_col4:
            avg_pnl = overall_metrics['avg_pnl']
            st.metric("Avg P&L/Trade", f"${avg_pnl:.2f}")
        
        with kpi_col5:
            roi = (total_pnl / 1000) * 100  # Assuming 1000 starting balance
            st.metric("ROI", f"{roi:.1f}%")
        
        # Performance analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Charts", "📊 Detailed Metrics", "🎯 Trade Analysis", "📅 Time Analysis"
        ])
        
        with tab1:
            if show_equity_curve:
                st.subheader("📈 Portfolio Equity Curve")
                equity_chart = create_equity_curve(all_trades)
                if equity_chart:
                    st.plotly_chart(equity_chart)
                else:
                    st.info("Unable to generate equity curve")
            
            if show_drawdown:
                st.subheader("📉 Drawdown Analysis")
                drawdown_chart = create_drawdown_chart(all_trades)
                if drawdown_chart:
                    st.plotly_chart(drawdown_chart)
                else:
                    st.info("Unable to generate drawdown chart")
            
            if show_distribution:
                st.subheader("📊 Returns Distribution")
                dist_chart = create_returns_distribution(all_trades)
                if dist_chart:
                    st.plotly_chart(dist_chart)
                else:
                    st.info("Unable to generate distribution chart")
        
        with tab2:
            if show_metrics:
                st.subheader("📊 Comprehensive Performance Metrics")
                metrics_df = create_performance_metrics_table(all_trades)
                
                if not metrics_df.empty:
                    st.dataframe(metrics_df, height=500)
                else:
                    st.info("Unable to calculate detailed metrics")
        
        with tab3:
            st.subheader("🎯 Individual Trade Analysis")
            
            # Trade performance by symbol
            symbol_performance = {}
            for trade in all_trades:
                symbol = trade.symbol
                pnl = trade.pnl or 0
                
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {
                        'trades': 0, 'total_pnl': 0, 'wins': 0
                    }
                
                symbol_performance[symbol]['trades'] += 1
                symbol_performance[symbol]['total_pnl'] += pnl
                if pnl > 0:
                    symbol_performance[symbol]['wins'] += 1
            
            if symbol_performance:
                st.markdown("### 📈 Performance by Symbol")
                
                symbol_data = []
                for symbol, data in symbol_performance.items():
                    win_rate = (data['wins'] / data['trades']) * 100
                    symbol_data.append({
                        "Symbol": symbol,
                        "Trades": data['trades'],
                        "Win Rate": f"{win_rate:.1f}%",
                        "Total P&L": f"${data['total_pnl']:.2f}",
                        "Avg P&L": f"${data['total_pnl'] / data['trades']:.2f}"
                    })
                
                symbol_df = pd.DataFrame(symbol_data)
                symbol_df = symbol_df.sort_values('Total P&L', ascending=False)
                st.dataframe(symbol_df)
            
            # Recent trades table
            st.markdown("### 📜 Recent Trades")
            recent_trades = sorted(all_trades, key=lambda x: x.timestamp or datetime.min, reverse=True)[:20]
            
            recent_data = []
            for trade in recent_trades:
                pnl = trade.pnl or 0
                recent_data.append({
                    "Date": trade.timestamp.strftime("%Y-%m-%d %H:%M") if trade.timestamp else "N/A",
                    "Symbol": trade.symbol,
                    "Side": trade.side,
                    "Entry": f"${trade.entry_price:.4f}",
                    "Exit": f"${trade.exit_price:.4f}" if trade.exit_price else "N/A",
                    "P&L": f"${pnl:.2f}",
                    "Type": "Virtual" if trade.virtual else "Real",
                    "Result": "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
                })
            
            if recent_data:
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
            
            # Performance by day of week
            if all_trades:
                day_performance = {}
                for trade in all_trades:
                    if trade.timestamp:
                        day_name = trade.timestamp.strftime("%A")
                        pnl = trade.pnl or 0
                        
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
            for trade in all_trades:
                if trade.timestamp:
                    month_key = trade.timestamp.strftime("%Y-%m")
                    pnl = trade.pnl or 0
                    
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
