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
from bybit_client import BybitClient

# Initialize the client with your API credentials
client = BybitClient()

# Configure logging
# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)


st.set_page_config(page_title="Signals - AlgoTrader Pro", page_icon="🎯", layout="wide")

def create_signal_chart(signal_data):
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
                         annotation_text=f"Entry: ${entry:.4f}")
        if tp > 0:
            fig.add_hline(y=tp, line_dash="dot", line_color="green", 
                         annotation_text=f"Take Profit: ${tp:.4f}")
        if sl > 0:
            fig.add_hline(y=sl, line_dash="dot", line_color="red", 
                         annotation_text=f"Stop Loss: ${sl:.4f}")
        
        fig.update_layout(
            title=f"{symbol} - Technical Analysis",
            yaxis_title="Price (USDT)",
            xaxis_title="Time",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating signal chart for {symbol}: {e}")
        return None

def display_signal_details(signal):
    """Display detailed signal information"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Signal Details**")
        st.write(f"**Symbol:** {signal.get('Symbol', signal.get('symbol', 'N/A'))}")
        st.write(f"**Side:** {signal.get('Side', signal.get('side', 'N/A'))}")
        st.write(f"**Type:** {signal.get('signal_type', 'N/A')}")
        st.write(f"**Score:** {signal.get('Score', '0%')}")
    
    with col2:
        st.markdown("**💰 Price Levels**")
        st.write(f"**Entry:** ${float(signal.get('Entry', signal.get('entry', 0)) or 0):.4f}")
        st.write(f"**Take Profit:** ${float(signal.get('TP', signal.get('tp', 0)) or 0):.4f}")
        st.write(f"**Stop Loss:** ${float(signal.get('SL', signal.get('sl', 0)) or 0):.4f}")
        st.write(f"**Trail Stop:** ${float(signal.get('Trail', signal.get('trail', 0)) or 0):.4f}")
    
    with col3:
        st.markdown("**⚙️ Trading Info**")
        st.write(f"**Market:** {signal.get('Market', signal.get('market', 'N/A'))}")
        st.write(f"**BB Slope:** {signal.get('bb_slope', 'N/A')}")
        st.write(f"**Leverage:** {signal.get('leverage', 10)}x")
        created_at = signal.get('Time', signal.get('created_at', 'N/A'))
        st.write(f"**Generated:** {created_at if created_at != 'N/A' else 'Unknown'}")

async def execute_real_trade(engine, signal):
    """Execute a real trade based on a signal"""
    try:
        symbol = signal.get("symbol")
        if not isinstance(symbol, str):
            logger.error("Symbol must be a non-empty string")
            return False
        
        side = signal.get("side", "Buy")
        entry_price = signal.get("entry")
        if not entry_price:
            entry_price = await engine.client.get_current_price(symbol)
        if entry_price <= 0:
            logger.error(f"Invalid entry price for {symbol}")
            return False
        
        position_size = engine.calculate_position_size(symbol, entry_price)
        trade_data = {
            "symbol": symbol,
            "side": side,
            "qty": position_size,
            "entry_price": entry_price,
            "order_id": f"real_{symbol}_{int(datetime.now().timestamp())}",
            "virtual": False,
            "status": "open",
            "score": signal.get("score"),
            "strategy": signal.get("strategy", "Auto"),
            "leverage": signal.get("leverage", 10)
        }
        order_result = await engine.client.place_order(
            symbol=symbol,
            side=side,
            qty=trade_data["qty"],
            price=entry_price,
            order_type="Market"
        )
        if order_result.get("success"):
            engine.db.add_trade(trade_data)
            engine.sync_real_balance()
            return True
        else:
            logger.error(f"Real trade failed: {order_result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Error executing real trade: {e}")
        return False

def main():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">🎯 Trading Signals</h1>
        <p style="color: #888; margin: 0;">AI-Powered Signal Generation & Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'generated_signals' not in st.session_state:
        st.session_state.generated_signals = []
        try:
            # Fetch recent signals from DB as fallback
            db_signals = db_manager.get_signals(limit=10)
            st.session_state.generated_signals = [s.to_dict() for s in db_signals if s.score >= 40]
            logger.info(f"Initialized {len(st.session_state.generated_signals)} signals from database")
        except Exception as e:
            logger.error(f"Error fetching initial signals from DB: {e}")
            st.session_state.generated_signals = []
    
    if 'signal_generation_in_progress' not in st.session_state:
        st.session_state.signal_generation_in_progress = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Signal Controls")
        
        # Trading mode selector
        trading_mode = st.selectbox(
            "Trading Mode",
            ["virtual", "real"],
            index=0 if db_manager.get_setting("trading_mode") != "real" else 1,
            help="Virtual mode uses simulated balances. Real mode trades with actual funds on Bybit."
        )
        if st.button("Save Mode"):
            db_manager.save_setting("trading_mode", trading_mode)
            st.success(f"Trading mode set to {trading_mode}")
        
        st.markdown("---")
        
        # Signal generation parameters
        st.subheader("Generation Settings")
        num_symbols = st.number_input("Number of Symbols to Scan", min_value=5, max_value=100, value=20)
        top_n = st.number_input("Top Signals to Keep", min_value=1, max_value=20, value=5)
        interval = st.selectbox("Time Interval", ["60", "15", "240", "D"])
        
        st.markdown("---")
        
        # Notification settings
        st.subheader("Notifications")
        notify_telegram = st.checkbox("Telegram", value=True)
        notify_discord = st.checkbox("Discord", value=False)
        notify_email = st.checkbox("Email", value=False)

    # Create engine and trader
    engine = create_engine()
    trader = engine.trader

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "⚡ Generate Signals", "🔍 Analyze Symbol", "🤖 ML Filtering", "🚀 Automated Trading"])

    with tab1:
        st.subheader("📊 Signals Dashboard")
        
        if not st.session_state.generated_signals:
            st.info("No signals generated yet. Use the 'Generate Signals' tab to create some.")
        else:
            df = pd.DataFrame(st.session_state.generated_signals)
            if not df.empty:
                st.dataframe(
                    df[['symbol', 'score', 'side', 'entry', 'tp', 'sl', 'market', 'created_at']],
            
                )
        
        st.subheader("Recent Trades")
        recent_trades = db_manager.get_trades(limit=20)
        if recent_trades:
            trade_df = pd.DataFrame([t.to_dict() for t in recent_trades])
            st.dataframe(
                trade_df[['symbol', 'side', 'entry_price', 'exit_price', 'pnl', 'status', 'timestamp']],
        
            )
        else:
            st.info("No recent trades")

    with tab2:
        st.subheader("⚡ Generate Trading Signals")
        
        if st.button("Generate Signals", disabled=st.session_state.signal_generation_in_progress):
            st.session_state.signal_generation_in_progress = True
            with st.spinner("Generating signals..."):
                try:
                    symbols = get_usdt_symbols(limit=int(num_symbols))
                    signals = generate_signals(
                        symbols,
                        interval=interval,
                        top_n=int(top_n),
                        trading_mode=trading_mode
                    )
                    if signals:
                        st.session_state.generated_signals = signals
                        st.success(f"Generated {len(signals)} signals")
                        
                        # Send notifications if enabled
                        notifications = []
                        if notify_telegram:
                            notifications.append("telegram")
                        if notify_discord:
                            notifications.append("discord")
                        if notify_email:
                            notifications.append("email")
                            
                        if notifications:
                            send_all_notifications(signals)
                    else:
                        st.warning("No signals generated")
                except Exception as e:
                    st.error(f"Error generating signals: {e}")
                    logger.error(f"Signal generation error: {e}")
            st.session_state.signal_generation_in_progress = False
        
        if st.session_state.generated_signals:
            st.subheader("Generated Signals")
            for signal in st.session_state.generated_signals:
                with st.expander(f"{signal.get('symbol')} - Score: {signal.get('score')}"):
                    display_signal_details(signal)
                    chart = create_signal_chart(signal)
                    if chart:
                        st.plotly_chart(chart)

    with tab3:
        st.subheader("🔍 Single Symbol Analysis")
        
        analysis_symbol = st.text_input("Enter Symbol (e.g., BTCUSDT)").upper()
        if st.button("Analyze Symbol"):
            with st.spinner("Analyzing..."):
                try:
                    analysis = analyze_single_symbol(analysis_symbol, interval)
                    if analysis:
                        st.session_state['single_analysis'] = analysis
                        st.success("Analysis complete")
                    else:
                        st.warning("No analysis data returned")
                except Exception as e:
                    st.error(f"Analysis error: {e}")
        
        if 'single_analysis' in st.session_state:
            signal = st.session_state['single_analysis']
            display_signal_details(signal)
            
            chart = create_signal_chart(signal)
            if chart:
                st.plotly_chart(chart)
            
            st.subheader("Detailed Indicators")
            st.json(signal.get('indicators', {}))
            
            trade_col1, trade_col2 = st.columns(2)
            with trade_col1:
                virtual_disabled = 'single_analysis' not in st.session_state
                if st.button("💻 Execute Virtual Trade", disabled=virtual_disabled):
                    try:
                        success = engine.execute_virtual_trade(
                            st.session_state['single_analysis'], trading_mode="virtual"
                        )
                        if success:
                            st.success(f"✅ Virtual trade executed for {analysis_symbol}")
                        else:
                            st.error("❌ Failed to execute virtual trade")
                    except Exception as e:
                        st.error(f"Virtual trade error: {e}")
            with trade_col2:
                real_disabled = 'single_analysis' not in st.session_state or trading_mode != "real"
                if st.button("💰 Execute Real Trade", disabled=real_disabled, key="execute_real_trade_tab3"):
                    try:
                        if not engine.is_trading_enabled():
                            st.error("Trading is disabled or emergency stop is active. Cannot execute trade.")
                        else:
                            success = engine.execute_virtual_trade(
                                st.session_state['single_analysis'], trading_mode="real"
                            )
                            if success:
                                st.success(f"✅ Real trade executed for {analysis_symbol}")
                            else:
                                st.error("❌ Failed to execute real trade")
                    except Exception as e:
                        st.error(f"Real trade error: {e}")

    with tab4:
        st.subheader("🤖 ML-Powered Signal Filtering")

        ml_filter = MLFilter()

        # ML Threshold slider
        threshold = st.slider(
            "ML Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Minimum ML probability for signal to pass filter"
        )

        # Fetch signals from DB
        signals = db_manager.get_signals(limit=100)
        st.write(f"Fetched {len(signals)} signals from database")

        # Filter signals with ML
        filtered_signals = ml_filter.filter_signals(signals, threshold=threshold)
        st.write(f"{len(filtered_signals)} signals passed the ML filter")
        st.dataframe(pd.DataFrame([s.to_dict() for s in filtered_signals]))

        st.markdown("---")

        # Feature Importance
        if st.button("Show Feature Importance"):
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
                signals_for_trade = db_manager.get_signals(limit=50)  # limit to recent signals
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
            st.json(status)
        
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

if __name__ == "__main__":
    main()