import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timezone
import asyncio
import uuid
from ml import MLFilter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import db_manager, Trade, Signal
from indicators import get_candles
from signal_generator import generate_signals, get_usdt_symbols, analyze_single_symbol
from notifications import send_all_notifications
from trading_engine import TradingEngine
from settings import load_settings, save_settings

# Configure logging
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
            st.error(f"Invalid price values for {symbol}")
            logger.error(f"Invalid price values for {symbol}: {e}", exc_info=True)
            return None

        # Get candlestick data
        candles = get_candles(client, symbol, "60", limit=50)
        if not candles or not isinstance(candles, list) or len(candles) == 0:
            st.warning(f"No candlestick data available for {symbol}")
            logger.warning(f"No candlestick data for {symbol}")
            return None

        df = pd.DataFrame(candles)
        required_columns = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Invalid candlestick data format for {symbol}: missing columns")
            logger.error(f"Invalid candlestick data format for {symbol}: missing columns")
            return None

        df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        if df['time'].isna().any():
            st.error(f"Invalid timestamp data for {symbol}")
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
            fig.add_hline(y=tp, line_dash="dash", line_color="green", 
                         annotation_text=f"TP: ${tp:.4f}")
        if sl > 0:
            fig.add_hline(y=sl, line_dash="dash", line_color="red", 
                         annotation_text=f"SL: ${sl:.4f}")

        fig.update_layout(
            title=f"{symbol} Signal Chart",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            showlegend=True
        )
        return fig
    except Exception as e:
        st.error(f"Error creating chart for {symbol}: {e}")
        logger.error(f"Error creating chart for {symbol}: {e}", exc_info=True)
        return None

def main():
    try:
        # Initialize session state defaults
        if "engine" not in st.session_state:
            st.session_state.engine = TradingEngine()
        if "trading_mode" not in st.session_state:
            st.session_state.trading_mode = "virtual"
        if "wallet_cache" not in st.session_state:
            st.session_state.wallet_cache = {"virtual": {"capital": 0.0, "available": 0.0, "used": 0.0}}

        engine = st.session_state.engine
        if not engine:
            st.error("Trading engine not initialized")
            logger.error("Trading engine not initialized")
            return

        # Ensure database session
        if not db_manager.session:
            try:
                db_manager.create_session()
            except Exception as e:
                st.error("Failed to initialize database session")
                logger.error(f"Failed to initialize database session: {e}", exc_info=True)
                return

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
                f"<small style='color:#888;'>Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}</small>",
                unsafe_allow_html=True
            )

        st.markdown("### 🎯 Trading Signals")
        
        tab1, tab2, tab3 = st.tabs(["📡 Current Signals", "📊 Signal Analysis", "🧠 ML Model"])

        with tab1:
            st.subheader("📡 Current Signals")
            signals = db_manager.get_signals(limit=50)
            if signals:
                signal_data = []
                for signal in signals:
                    signal_dict = signal.to_dict()
                    signal_data.append({
                        "Symbol": signal.symbol,
                        "Side": signal.side,
                        "Entry": f"${signal.entry:.4f}" if signal.entry else "N/A",
                        "TP": f"${signal.tp:.4f}" if signal.tp else "N/A",
                        "SL": f"${signal.sl:.4f}" if signal.sl else "N/A",
                        "Score": f"{signal.score:.2f}" if signal.score else "N/A",
                        "Strategy": signal.strategy or "N/A",
                        "Created At": signal.created_at.strftime("%Y-%m-%d %H:%M:%S") if signal.created_at else "N/A"
                    })
                df = pd.DataFrame(signal_data)
                st.dataframe(df)
                
                selected_signal = st.selectbox("Select Signal to View Chart", [s["Symbol"] for s in signal_data])
                selected_signal_data = next((s for s in signal_data if s["Symbol"] == selected_signal), None)
                if selected_signal_data:
                    fig = create_signal_chart(selected_signal_data, engine.client)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Generate Trade from Selected Signal"):
                    if selected_signal_data:
                        try:
                            signal = next((s for s in signals if s.symbol == selected_signal_data["Symbol"]), None)
                            if not signal:
                                st.error("Selected signal not found")
                                logger.error(f"Selected signal {selected_signal_data['Symbol']} not found")
                                return
                            
                            qty = (signal.margin_usdt / signal.entry) if signal.entry and signal.margin_usdt else 0.01
                            trade_data = {
                                "symbol": signal.symbol,
                                "side": signal.side,
                                "qty": qty,
                                "entry_price": signal.entry,
                                "take_profit": signal.tp,
                                "stop_loss": signal.sl,
                                "virtual": st.session_state.trading_mode == "virtual",
                                "timestamp": datetime.now(timezone.utc),
                                "status": "open",
                                "order_id": str(uuid.uuid4()),
                                "strategy": signal.strategy,
                                "leverage": signal.leverage,
                                "score": signal.score
                            }
                            trade = Trade(**trade_data)
                            db_manager.session.add(trade)
                            db_manager.session.commit()
                            st.success(f"✅ Trade created for {signal.symbol} ({signal.side})")
                            logger.info(f"Trade created from signal: {trade_data}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create trade: {e}")
                            logger.error(f"Failed to create trade: {e}", exc_info=True)
                    else:
                        st.warning("Please select a signal to generate a trade")
                        logger.warning("No signal selected for trade generation")
                
                if st.button("Send Notifications"):
                    try:
                        send_all_notifications([s.to_dict() for s in signals])
                        st.success("✅ Notifications sent")
                        logger.info("Notifications sent for signals")
                    except Exception as e:
                        st.error(f"Failed to send notifications: {e}")
                        logger.error(f"Failed to send notifications: {e}", exc_info=True)
            else:
                st.info("No signals available. Generate signals to view them here!")

        with tab2:
            st.subheader("📊 Signal Analysis")
            symbols = get_usdt_symbols()
            selected_symbol = st.selectbox("Select Symbol for Analysis", symbols)
            timeframe = st.selectbox("Timeframe", ["1", "5", "15", "60", "240", "D"])
            
            if st.button("Analyze Symbol"):
                try:
                    signal_dict = analyze_single_symbol(selected_symbol, timeframe)
                    if signal_dict:
                        # Convert dictionary to Signal object for consistency
                        signal_data = {
                            "symbol": signal_dict.get("symbol", selected_symbol),
                            "side": signal_dict.get("side", "BUY"),
                            "entry": signal_dict.get("entry", 0.0),
                            "tp": signal_dict.get("tp", 0.0),
                            "sl": signal_dict.get("sl", 0.0),
                            "score": signal_dict.get("score", 0.0),
                            "strategy": signal_dict.get("signal_type", "Auto"),
                            "indicators": signal_dict.get("indicators", {}),
                            "leverage": engine.settings.get("LEVERAGE", 10),
                            "margin_usdt": signal_dict.get("margin_usdt", 100.0),
                            "created_at": datetime.fromtimestamp(signal_dict.get("analysis_time", time.time()), tz=timezone.utc)
                        }
                        signal = Signal(**signal_data)
                        db_manager.session.add(signal)
                        db_manager.session.commit()
                        
                        st.write(f"**Analysis for {selected_symbol} ({timeframe}m)**")
                        st.write(f"Side: {signal.side}")
                        st.write(f"Entry: ${signal.entry:.4f}")
                        st.write(f"Take Profit: ${signal.tp:.4f}")
                        st.write(f"Stop Loss: ${signal.sl:.4f}")
                        st.write(f"Score: {signal.score:.2f}")
                        st.write(f"Strategy: {signal.strategy}")
                        
                        fig = create_signal_chart(signal.to_dict(), engine.client)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No signal generated for {selected_symbol}")
                        logger.warning(f"No signal generated for {selected_symbol} on {timeframe}m")
                except Exception as e:
                    st.error(f"Error analyzing symbol: {e}")
                    logger.error(f"Error analyzing {selected_symbol}: {e}", exc_info=True)

        with tab3:
            st.subheader("🧠 ML Model")
            ml_filter = MLFilter()
            
            if st.button("Generate Signals with ML Filter"):
                try:
                    signals = generate_signals()
                    filtered_signals = ml_filter.filter_signals(signals, threshold=0.7)
                    for signal in filtered_signals:
                        db_manager.session.add(signal)
                    db_manager.session.commit()
                    st.success(f"✅ Generated {len(filtered_signals)} high-confidence signals")
                    logger.info(f"Generated {len(filtered_signals)} high-confidence signals")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating signals: {e}")
                    logger.error(f"Error generating signals: {e}", exc_info=True)
            
            if st.button("Show Feature Importance"):
                try:
                    importance = ml_filter.get_feature_importance()
                    if importance:
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Importance'])
                        st.bar_chart(importance_df)
                    else:
                        st.info("No trained ML model available")
                        logger.info("No trained ML model available for feature importance")
                except Exception as e:
                    st.error(f"Error retrieving feature importance: {e}")
                    logger.error(f"Error retrieving feature importance: {e}", exc_info=True)

            if st.button("Train ML Model from Trades"):
                try:
                    trades = db_manager.get_trades(limit=1000)
                    training_data = []
                    for trade in trades:
                        signals_for_trade = db_manager.get_signals(limit=50)
                        signal = next(
                            (s for s in signals_for_trade 
                             if s.symbol == trade.symbol and abs((s.created_at - trade.timestamp).total_seconds()) < 3600),
                            None
                        )
                        if signal and signal.indicators:
                            training_data.append({
                                'indicators': {
                                    k: signal.indicators.get(k, 0) for k in ml_filter.feature_columns
                                },
                                'profit': trade.pnl if trade.pnl is not None else 0
                            })
                    if not training_data:
                        st.error("No trades with matching signal indicators found for training")
                        logger.error("No trades with matching signal indicators found")
                    else:
                        success = ml_filter.train_model(training_data)
                        if success:
                            st.success("✅ ML model trained successfully")
                            logger.info("ML model trained successfully")
                        else:
                            st.error("Failed to train ML model. Check logs for details")
                            logger.error("Failed to train ML model")
                except Exception as e:
                    st.error(f"Error training ML model: {e}")
                    logger.error(f"Error training ML model: {e}", exc_info=True)
            
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

    except Exception as e:
        st.error(f"Error loading signals page: {e}")
        logger.error(f"Signals page error: {e}", exc_info=True)
        
        st.markdown("### 🔧 Error Recovery")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Loading"):
                st.rerun()
        with col2:
            if st.button("📊 Go to Dashboard"):
                st.switch_page("pages/dashboard.py")

if __name__ == "__main__":
    main()