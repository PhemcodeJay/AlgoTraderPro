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
from engine import TradingEngine

# Configure logging
from logging_config import get_logger
logger = get_logger(__name__)

st.set_page_config(page_title="Signals - AlgoTrader Pro", page_icon="🎯", layout="wide")

def create_signal_chart(signal_data):
    """Create a candlestick chart with entry, TP, SL, trail, and liquidation lines"""
    try:
        symbol = signal_data.get('Symbol', signal_data.get('symbol', 'BTCUSDT'))
        try:
            entry = float(signal_data.get('Entry', signal_data.get('entry', 0)) or 0)
            tp = float(signal_data.get('TP', signal_data.get('tp', 0)) or 0)
            sl = float(signal_data.get('SL', signal_data.get('sl', 0)) or 0)
            trail = float(signal_data.get('Trail', signal_data.get('trail', 0)) or 0)
            liquidation = float(signal_data.get('Liquidation', signal_data.get('liquidation', 0)) or 0)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid price values for {symbol}: {e}")
            return None

        # Get candlestick data
        candles = get_candles(symbol, "60", limit=50)
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
        if trail > 0:
            fig.add_hline(y=trail, line_dash="dashdot", line_color="purple", 
                         annotation_text=f"Trail: ${trail:.4f}")
        if liquidation > 0:
            fig.add_hline(y=liquidation, line_dash="dashdot", line_color="orange", 
                         annotation_text=f"Liquidation: ${liquidation:.4f}")
        
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
        score = signal.get('Score', signal.get('score', '0%'))
        st.write(f"**Score:** {score if isinstance(score, str) and '%' in score else f'{float(score):.1f}%'}")
        st.write(f"**BB Slope:** {signal.get('bb_slope', signal.get('indicators', {}).get('bb_slope', 'N/A'))}")
    
    with col2:
        st.markdown("**💰 Price Levels**")
        st.write(f"**Market Price:** ${float(signal.get('indicators', {}).get('price', 0)):.4f}")
        st.write(f"**Entry:** ${float(signal.get('Entry', signal.get('entry', 0)) or 0):.4f}")
        st.write(f"**Take Profit:** ${float(signal.get('TP', signal.get('tp', 0)) or 0):.4f}")
        st.write(f"**Stop Loss:** ${float(signal.get('SL', signal.get('sl', 0)) or 0):.4f}")
        st.write(f"**Trail:** ${float(signal.get('Trail', signal.get('trail', 0)) or 0):.4f}")
    
    with col3:
        st.markdown("**🔒 Risk Management**")
        st.write(f"**Liquidation:** ${float(signal.get('Liquidation', signal.get('liquidation', 0)) or 0):.4f}")
        st.write(f"**Margin USDT:** ${float(signal.get('Margin USDT', signal.get('margin_usdt', 0)) or 0):.2f}")
        st.write(f"**Leverage:** {signal.get('leverage', 'N/A')}x")
        st.write(f"**Market Type:** {signal.get('market', 'N/A')}")
        created_at = signal.get('Time', signal.get('created_at', 'N/A'))
        st.write(f"**Generated:** {created_at if isinstance(created_at, str) and created_at != 'N/A' else str(created_at)[:19]}")

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
            "score": float(str(signal.get("score", "0%")).replace("%", "")),
            "strategy": signal.get("strategy", "Auto"),
            "leverage": signal.get("leverage", 10),
            "sl": float(signal.get("sl", 0)),
            "tp": float(signal.get("tp", 0)),
            "trail": float(signal.get("trail", 0)),
            "liquidation": float(signal.get("liquidation", 0)),
            "margin_usdt": float(signal.get("margin_usdt", 0))
        }
        order_result = await engine.client.place_order(
            symbol=symbol,
            side=side,
            qty=trade_data["qty"],
            price=entry_price,
            stop_loss=trade_data["sl"],
            take_profit=trade_data["tp"],
            leverage=trade_data["leverage"],
            mode="ISOLATED"
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
        
        trading_mode = st.selectbox(
            "Trading Mode", 
            ["virtual", "real"], 
            index=0 if st.session_state.get('trading_mode', 'virtual') == 'virtual' else 1,
            key="signals_trading_mode"
        )
        st.session_state.trading_mode = trading_mode
        
        st.divider()
        
        st.subheader("📊 Generation Settings")
        top_n_signals = st.slider("Number of Signals", 1, 20, 10)
        min_score = st.slider("Minimum Score", 30, 90, 50)
        
        available_symbols = get_usdt_symbols(100)
        selected_symbols = st.multiselect(
            "Select Symbols (leave empty for auto-selection)",
            available_symbols,
            default=[]
        )
        
        st.divider()
        
        if st.button("🎯 Generate New Signals", 
                    disabled=st.session_state.signal_generation_in_progress):
            st.session_state.signal_generation_in_progress = True
        
        if st.button("📤 Send Notifications", 
                    disabled=len(st.session_state.generated_signals) == 0):
            if st.session_state.generated_signals:
                try:
                    send_all_notifications(st.session_state.generated_signals)
                    st.success("Notifications sent!")
                except Exception as e:
                    st.error(f"Notification error: {e}")
        
        st.divider()
        
        st.subheader("🔍 Database Filters")
        symbol_filter = st.text_input("Symbol Filter", placeholder="BTC, ETH, etc.")
        side_filter = st.selectbox("Side Filter", ["All", "Buy", "Sell"])
        
        if st.button("📊 Back to Dashboard"):
            st.switch_page("app.py")
    
    # Handle signal generation
    if st.session_state.signal_generation_in_progress:
        with st.spinner("🔄 Generating signals... This may take a few minutes."):
            try:
                symbols_to_scan = selected_symbols if selected_symbols else get_usdt_symbols(50)
                signals = generate_signals(
                    symbols_to_scan, 
                    interval="60", 
                    top_n=top_n_signals,
                    trading_mode=trading_mode
                )
                filtered_signals = [s for s in signals if float(str(s.get('score', '0%')).replace('%', '')) >= min_score]
                st.session_state.generated_signals = filtered_signals
                st.success(f"✅ Generated {len(filtered_signals)} signals, saved to database")
            except Exception as e:
                st.error(f"Error generating signals: {e}")
                logger.error(f"Signal generation error: {e}")
            finally:
                st.session_state.signal_generation_in_progress = False
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🆕 Generated Signals", "💾 Database Signals", "🔍 Single Symbol Analysis", "🤖 ML Signal Filter"])
    
    with tab1:
        st.subheader("🆕 Recently Generated Signals")
        signals = st.session_state.generated_signals
        if signals:
            signals_data = []
            for s in signals:
                try:
                    signals_data.append({
                        "Symbol": s.get("Symbol", s.get("symbol", "N/A")),
                        "Side": s.get("Side", s.get("side", "N/A")),
                        "Score": f"{float(s.get('score', '0%').replace('%', '')):.1f}%",
                        "Market Price": f"${float(s.get('indicators', {}).get('price', 0)):.4f}",
                        "Entry": f"${float(s.get('Entry', s.get('entry', 0)) or 0):.4f}",
                        "Stop Loss": f"${float(s.get('SL', s.get('sl', 0)) or 0):.4f}",
                        "Take Profit": f"${float(s.get('TP', s.get('tp', 0)) or 0):.4f}",
                        "Trail": f"${float(s.get('Trail', s.get('trail', 0)) or 0):.4f}",
                        "Liquidation": f"${float(s.get('Liquidation', s.get('liquidation', 0)) or 0):.4f}",
                        "Margin USDT": f"${float(s.get('Margin USDT', s.get('margin_usdt', 0)) or 0):.2f}",
                        "Market": s.get("Market", s.get("market", "N/A")),
                        "BB Slope": s.get("bb_slope", s.get("indicators", {}).get("bb_slope", "N/A"))
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid signal: {s}, error: {e}")
                    continue

            if signals_data:
                signals_df = pd.DataFrame(signals_data)
                st.dataframe(signals_df, height=400)

                selected_idx = st.selectbox(
                    "Select signal for detailed analysis:",
                    range(len(signals)),
                    format_func=lambda idx: f"{signals[idx].get('Symbol', signals[idx].get('symbol', 'N/A'))} - {signals[idx].get('score', '0%')}"
                )

                if selected_idx is not None:
                    selected_signal = signals[selected_idx]
                    display_signal_details(selected_signal)
                    chart = create_signal_chart(selected_signal)
                    if chart:
                        st.plotly_chart(chart)
                    else:
                        st.error("Failed to generate chart for selected signal.")

                    engine = TradingEngine()
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📈 Execute Virtual Trade"):
                            try:
                                success = engine.execute_virtual_trade(selected_signal, trading_mode="virtual")
                                if success:
                                    st.success("✅ Virtual trade executed!")
                                else:
                                    st.error("❌ Failed to execute virtual trade")
                            except Exception as e:
                                st.error(f"Error executing virtual trade: {e}")
                    with col2:
                        real_disabled = st.session_state.get("trading_mode", "virtual") != "real"
                        if st.button("💰 Execute Real Trade", disabled=real_disabled, key="execute_real_trade_tab1"):
                            if real_disabled:
                                st.info("Switch to real mode to execute real trades")
                            else:
                                try:
                                    if not engine.is_trading_enabled():
                                        st.error("Trading disabled or emergency stop active")
                                    else:
                                        success = asyncio.run(execute_real_trade(engine, selected_signal))
                                        if success:
                                            st.success(f"✅ Real trade executed for {selected_signal.get('symbol', 'N/A')}")
                                        else:
                                            st.error("❌ Failed to execute real trade")
                                except Exception as e:
                                    st.error(f"Error executing real trade: {e}")
            else:
                st.warning("No valid signals available to display.")
        else:
            st.info("🎯 Click 'Generate New Signals' to start analyzing the markets!")

    with tab2:
        st.subheader("💾 Historical Signals from Database")
        try:
            db_signals = db_manager.get_signals(limit=50)
            if db_signals:
                filtered_db_signals = []
                for signal in db_signals:
                    signal_dict = signal.to_dict()
                    if symbol_filter and symbol_filter.upper() not in signal_dict.get('symbol', '').upper():
                        continue
                    if side_filter != "All" and signal_dict.get('side', '').lower() != side_filter.lower():
                        continue
                    if signal_dict.get('score', 0) < min_score:
                        continue
                    filtered_db_signals.append(signal_dict)
                
                if filtered_db_signals:
                    db_data = []
                    for s in filtered_db_signals[:30]:
                        db_data.append({
                            "Symbol": s.get("symbol", "N/A"),
                            "Side": s.get("side", "N/A"),
                            "Score": f"{float(s.get('score', 0)):.1f}%",
                            "Market Price": f"${float(s.get('indicators', {}).get('price', 0)):.4f}",
                            "Entry": f"${float(s.get('entry', 0)):.4f}",
                            "Stop Loss": f"${float(s.get('sl', 0)):.4f}",
                            "Take Profit": f"${float(s.get('tp', 0)):.4f}",
                            "Trail": f"${float(s.get('trail', 0)):.4f}",
                            "Liquidation": f"${float(s.get('liquidation', 0)):.4f}",
                            "Margin USDT": f"${float(s.get('margin_usdt', 0)):.2f}",
                            "Market": s.get("market", "N/A"),
                            "BB Slope": s.get("indicators", {}).get("bb_slope", "N/A"),
                            "Created": str(s.get("created_at", "N/A"))[:19]
                        })
                    
                    db_df = pd.DataFrame(db_data)
                    st.dataframe(db_df, height=400)
                    
                    csv = db_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Signals CSV",
                        csv,
                        "trading_signals.csv",
                        "text/csv",
                        key="download_signals"
                    )
                else:
                    st.info("No signals match the current filters")
            else:
                st.info("No signals in database. Generate some signals first!")
        except Exception as e:
            st.error(f"Error loading database signals: {e}")
            logger.error(f"Database signals error: {e}")
    
    with tab3:
        st.subheader("🔍 Single Symbol Analysis")
        col1, col2 = st.columns([1, 2])
        with col1:
            analysis_symbol = st.selectbox(
                "Select Symbol for Analysis:",
                get_usdt_symbols(50),
                key="analysis_symbol"
            )
            analysis_interval = st.selectbox(
                "Time Interval:",
                ["15", "30", "60", "240", "D"],
                index=2,
                key="analysis_interval"
            )
            if st.button("🔍 Analyze Symbol"):
                with st.spinner(f"Analyzing {analysis_symbol}..."):
                    try:
                        analysis_result = analyze_single_symbol(analysis_symbol, analysis_interval)
                        if analysis_result:
                            st.session_state['single_analysis'] = analysis_result
                            st.success(f"✅ Analysis complete for {analysis_symbol}")
                        else:
                            st.warning(f"No significant signal found for {analysis_symbol}")
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
        with col2:
            if 'single_analysis' in st.session_state:
                analysis = st.session_state['single_analysis']
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Signal Score", f"{float(analysis.get('score', 0)):.1f}%")
                    st.metric("Signal Type", analysis.get('signal_type', 'N/A').title())
                    st.metric("BB Slope", analysis.get('indicators', {}).get('bb_slope', 'N/A'))
                with metrics_col2:
                    indicators = analysis.get('indicators', {})
                    st.metric("Market Price", f"${float(indicators.get('price', 0)):.4f}")
                    st.metric("Entry", f"${float(analysis.get('entry', 0)):.4f}")
                    st.metric("Stop Loss", f"${float(analysis.get('sl', 0)):.4f}")
                with metrics_col3:
                    st.metric("Take Profit", f"${float(analysis.get('tp', 0)):.4f}")
                    st.metric("Trail", f"${float(analysis.get('trail', 0)):.4f}")
                    st.metric("Liquidation", f"${float(analysis.get('liquidation', 0)):.4f}")
                    st.metric("Margin USDT", f"${float(analysis.get('margin_usdt', 0)):.2f}")
                chart = create_signal_chart(analysis)
                if chart:
                    st.plotly_chart(chart)
            else:
                st.info("Select a symbol and click 'Analyze Symbol' to see detailed analysis")
        
        trade_col1, trade_col2 = st.columns(2)
        engine = TradingEngine()
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
                        signal = st.session_state['single_analysis']
                        success = asyncio.run(execute_real_trade(engine, signal))
                        if success:
                            st.success(f"✅ Real trade executed for {signal.get('symbol', 'N/A')}")
                        else:
                            st.error("❌ Failed to execute real trade")
                except Exception as e:
                    st.error(f"Real trade error: {e}")

    with tab4:
        st.subheader("🤖 ML-Powered Signal Filtering")
        ml_filter = MLFilter()
        threshold = st.slider(
            "ML Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Minimum ML probability for signal to pass filter"
        )
        signals = db_manager.get_signals(limit=100)
        st.write(f"Fetched {len(signals)} signals from database")
        filtered_signals = ml_filter.filter_signals(signals, threshold=threshold)
        st.write(f"{len(filtered_signals)} signals passed the ML filter")
        
        filtered_data = []
        for sig in filtered_signals:
            ind = sig.indicators if hasattr(sig, "indicators") else sig.get("indicators", {})
            readable_ind = "\n".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in ind.items()])
            filtered_data.append({
                "Symbol": sig.symbol if hasattr(sig, "symbol") else sig.get("symbol", "N/A"),
                "Side": sig.side if hasattr(sig, "side") else sig.get("side", "N/A"),
                "Score": f"{float(sig.score if hasattr(sig, 'score') else sig.get('score', 0)):.1f}%",
                "Market Price": f"${float(ind.get('price', 0)):.4f}",
                "Entry": f"${float(sig.entry if hasattr(sig, 'entry') else sig.get('entry', 0)):.4f}",
                "Stop Loss": f"${float(sig.sl if hasattr(sig, 'sl') else sig.get('sl', 0)):.4f}",
                "Take Profit": f"${float(sig.tp if hasattr(sig, 'tp') else sig.get('tp', 0)):.4f}",
                "Trail": f"${float(sig.trail if hasattr(sig, 'trail') else sig.get('trail', 0)):.4f}",
                "Liquidation": f"${float(sig.liquidation if hasattr(sig, 'liquidation') else sig.get('liquidation', 0)):.4f}",
                "Margin USDT": f"${float(sig.margin_usdt if hasattr(sig, 'margin_usdt') else sig.get('margin_usdt', 0)):.2f}",
                "Market": sig.market if hasattr(sig, "market") else sig.get("market", "N/A"),
                "BB Slope": ind.get("bb_slope", "N/A"),
                "Indicators": readable_ind
            })
        
        st.dataframe(pd.DataFrame(filtered_data), height=400)

        st.markdown("---")
        if st.button("Show Feature Importance"):
            importance = ml_filter.get_feature_importance()
            if importance:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Importance'])
                st.bar_chart(importance_df)
            else:
                st.info("No trained ML model available")

        st.markdown("---")
        if st.button("Train ML Model from Trades"):
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

if __name__ == "__main__":
    main()