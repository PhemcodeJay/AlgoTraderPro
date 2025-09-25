import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime
import asyncio
from ml import MLFilter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import db_manager
from indicators import get_candles
from signal_generator import generate_signals, get_usdt_symbols, analyze_single_symbol
from notifications import send_all_notifications
from engine import create_engine
from utils import format_currency_safe, format_price_safe
from logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Signals - AlgoTrader Pro", page_icon="🎯", layout="wide")

def create_signal_chart(signal_data, client):
    """Create a candlestick chart with entry, TP, and SL lines"""
    try:
        symbol = signal_data.get('Symbol', signal_data.get('symbol', 'BTCUSDT'))
        try:
            entry = float(signal_data.get('Entry', signal_data.get('entry', 0)) or 0)
            tp = float(signal_data.get('TP', signal_data.get('tp', 0)) or 0)
            sl = float(signal_data.get('SL', signal_data.get('sl', 0)) or 0)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid price values for {symbol}: {e}")
            return None
        
        # Now you can call get_candles
        candles = get_candles(client=client, symbol=symbol, interval="60", limit=50)
        if not candles or not isinstance(candles, list) or len(candles) == 0:
            logger.warning(f"No candlestick data for {symbol}")
            return None

        df = pd.DataFrame(candles)
        required_columns = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Invalid candlestick data format for {symbol}: missing columns")
            return None

        df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        if df['time'].isna().any():
            logger.error(f"Invalid timestamp data for {symbol}")
            return None

        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))
        
        # Add signal lines
        if entry > 0:
            fig.add_hline(y=entry, line_dash="dash", line_color="blue", 
                         annotation_text=f"Entry: ${format_price_safe(entry)}")
        if tp > 0:
            fig.add_hline(y=tp, line_dash="dot", line_color="green", 
                         annotation_text=f"Take Profit: ${format_price_safe(tp)}")
        if sl > 0:
            fig.add_hline(y=sl, line_dash="dot", line_color="red", 
                         annotation_text=f"Stop Loss: ${format_price_safe(sl)}")
        
        fig.update_layout(
            title=f"{symbol} - Technical Analysis",
            yaxis_title="Price (USDT)",
            xaxis_title="Time",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating signal chart for {symbol}: {e}", exc_info=True)
        return None

def display_signal_details(signal):
    """Display detailed signal information"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Signal Details**")
        st.write(f"Symbol: {signal.get('symbol', 'N/A')}")
        st.write(f"Interval: {signal.get('interval', 'N/A')}")
        st.write(f"Type: {signal.get('signal_type', 'N/A')}")
        st.write(f"Side: {signal.get('side', 'N/A')}")
        st.write(f"Score: {signal.get('score', 0):.1f}")
    
    with col2:
        st.markdown("**💰 Price Levels**")
        st.write(f"Entry: ${format_price_safe(signal.get('entry'))}")
        st.write(f"Take Profit: ${format_price_safe(signal.get('tp'))}")
        st.write(f"Stop Loss: ${format_price_safe(signal.get('sl'))}")
        st.write(f"Trail: ${format_price_safe(signal.get('trail'))}")
        st.write(f"Liquidation: ${format_price_safe(signal.get('liquidation'))}")
    
    with col3:
        st.markdown("**⚙️ Parameters**")
        st.write(f"Leverage: {signal.get('leverage', 10)}x")
        st.write(f"Margin USDT: ${format_currency_safe(signal.get('margin_usdt'))}")
        st.write(f"Market: {signal.get('market', 'N/A')}")
        st.write(f"Strategy: {signal.get('strategy', 'Auto')}")
        st.write(f"Created: {signal.get('created_at', datetime.now().isoformat())[:19]}")

def main():
    try:
        # Initialize or get engine from session state
        if 'engine' not in st.session_state:
            st.session_state.engine = create_engine()
        
        engine = st.session_state.engine
        client = engine.client
        trader = engine.trader
        
        if not client or not client.is_connected():
            st.warning("Bybit client not connected. Some features may be limited.")
        
        st.title("🎯 Trading Signals")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📡 Signals",
            "📈 Charts",
            "📊 Trades",
            "🤖 ML Filter",
            "🤖 Automated Trading"
        ])
        
        with tab1:
            st.subheader("Generate Trading Signals")
            
            col1, col2 = st.columns(2)
            with col1:
                num_symbols = st.number_input("Number of Symbols to Scan", min_value=5, max_value=100, value=20)
            with col2:
                top_n = st.number_input("Top N Signals", min_value=1, max_value=50, value=5)
            
            if st.button("🔍 Generate Signals"):
                with st.spinner("Scanning markets..."):
                    symbols = get_usdt_symbols(limit=num_symbols)
                    if not symbols:
                        st.error("Failed to fetch symbols")
                        return
                    
                    signals = generate_signals(symbols, interval="60", top_n=top_n)
                    
                    if signals:
                        st.session_state.signals = signals
                        st.success(f"Generated {len(signals)} signals")
                    else:
                        st.warning("No signals generated")
            
            if 'signals' in st.session_state and st.session_state.signals:
                signals = st.session_state.signals
                signals_df = pd.DataFrame([{
                    'Symbol': s.get('symbol'),
                    'Score': round(s.get('score', 0), 1),
                    'Side': s.get('side'),
                    'Entry': format_price_safe(s.get('entry')),
                    'SL': format_price_safe(s.get('sl')),
                    'TP': format_price_safe(s.get('tp')),
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
                
                if st.button("📤 Send Notifications"):
                    try:
                        send_all_notifications(signals)
                        st.success("Notifications sent!")
                    except Exception as e:
                        st.error(f"Failed to send notifications: {e}")
            
            else:
                st.info("Generate signals to view them here")
        
        with tab2:
            st.subheader("Signal Charts")
            
            if 'signals' in st.session_state and st.session_state.signals:
                selected_symbol = st.selectbox(
                    "Select Signal",
                    options=[s.get('symbol') for s in st.session_state.signals]
                )
                
                selected_signal = next(
                    (s for s in st.session_state.signals if s.get('symbol') == selected_symbol),
                    None
                )
                
                if selected_signal:
                    chart = create_signal_chart(selected_signal, client)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.warning("Unable to generate chart")
                    
                    display_signal_details(selected_signal)
            else:
                st.info("Generate signals first to view charts")
        
        with tab3:
            st.subheader("Recent Trades")
            
            trades = [t.to_dict() for t in db_manager.get_trades(limit=50)]
            if trades:
                trades_df = pd.DataFrame([{
                    'Symbol': t.get('symbol'),
                    'Side': t.get('side'),
                    'Entry': format_price_safe(t.get('entry_price')),
                    'PnL': format_currency_safe(t.get('pnl')) if t.get('status') == 'closed' else 'Open',
                    'Status': t.get('status'),
                    'Mode': 'Virtual' if t.get('virtual') else 'Real',
                    'Timestamp': t.get('timestamp')[:19] if t.get('timestamp') else 'N/A'
                } for t in trades])
                
                st.dataframe(
                    trades_df.sort_values('Timestamp', ascending=False),
                    column_config={
                        "Entry": st.column_config.NumberColumn(format="$ %.4f"),
                        "PnL": st.column_config.NumberColumn(format="$ %.2f"),
                        "Timestamp": st.column_config.DateColumn(format="YYYY-MM-DD HH:mm:ss")
                    },
                    use_container_width=True
                )
            else:
                st.info("No trades available")
        
        with tab4:
            st.subheader("🤖 ML Signal Filter")
            
            ml_filter = MLFilter()
            
            if st.button("Refresh ML Status"):
                status = ml_filter.get_status()
                st.json(status)
            
            st.subheader("Live Signal Scoring")
            with st.form("ml_scoring"):
                rsi = st.number_input("RSI", value=50.0)
                macd = st.number_input("MACD Histogram", value=0.0)
                vol_ratio = st.number_input("Volume Ratio", value=1.0)
                volatility = st.number_input("Volatility", value=1.0)
                trend_score = st.number_input("Trend Score", value=0.0)
                
                if st.form_submit_button("Score Signal"):
                    features = {
                        'rsi': rsi,
                        'macd_histogram': macd,
                        'volume_ratio': vol_ratio,
                        'volatility': volatility,
                        'trend_score': trend_score
                    }
                    score = ml_filter.predict(features)
                    st.metric("ML Score", f"{score:.2f}")
            
            importance = ml_filter.get_feature_importance()
            if importance:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Importance'])
                st.bar_chart(importance_df)
            else:
                st.info("No trained ML model available")

            st.markdown("---")

            # Train ML Model button
            if st.button("Train ML Model from Trades"):
                trades = db_manager.get_trades(limit=1000)
                training_data = []

                for trade in trades:
                    # Match trade with closest signal (within 1 hour)
                    signals_for_trade = db_manager.get_signals(limit=500)  # Increased limit for better matching
                    signal = next(
                        (s for s in signals_for_trade 
                        if s.symbol == trade.symbol and abs((s.created_at - trade.timestamp).total_seconds()) < 3600),
                        None
                    )
                    if signal and signal.indicators:
                        training_data.append({
                            'indicators': signal.indicators,
                            'profit': trade.pnl if trade.pnl is not None else 0
                        })

                if not training_data:
                    st.error("No trades with matching signal indicators found for training")
                else:
                    success = ml_filter.train_model(training_data)
                    if success:
                        st.success("ML model trained successfully")
                    else:
                        st.error("Failed to train ML model. Check logs for details")
                    
            st.markdown("""
            ## 🧠 How ML Works with Signals & Trades
            1. **Signals Collection**:  
            - Signals are generated by your strategies with indicators like RSI, MACD, Bollinger Bands, volatility, etc.
            2. **ML Filtering**:  
            - ML models score signals based on historical patterns. Only signals above a threshold are considered for trading.
            3. **Training ML Model**:  
            - Uses trades or signals as historical data, with indicators as features and profit/score as target.
            - Feature importance shows which indicators influence the ML prediction the most.
            4. **Generating Trades**:  
            - Trades can be created from filtered signals, inheriting key parameters like side, entry price, leverage.
            5. **Live Scoring**:  
            - Input current market indicators to get instant ML probability scores for new signals.
            """)

        with tab5:
            st.subheader("🚀 Automated Trading Control")
            
            status_container = st.container()
            
            if st.button("Refresh Status"):
                status = asyncio.run(trader.get_status())
                status_container.json(status)
            
            if st.button("Start Automated Trading"):
                success = asyncio.run(trader.start(status_container))
                if success:
                    st.success("Automated trading started")
                else:
                    st.error("Failed to start automated trading")
            
            if st.button("Stop Automated Trading"):
                success = asyncio.run(trader.stop())
                if success:
                    st.success("Automated trading stopped")
                else:
                    st.error("Failed to stop automated trading")
            
            if st.button("Reset Statistics"):
                asyncio.run(trader.reset_stats())
                st.success("Statistics reset")
            
            st.subheader("Performance Summary")
            perf = trader.get_performance_summary()
            st.json(perf)

    except Exception as e:
        st.error(f"Signals page error: {e}")
        logger.error(f"Signals page error: {e}", exc_info=True)

if __name__ == "__main__":
    main()