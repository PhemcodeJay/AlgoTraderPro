import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime
import asyncio
from ml import MLFilter
from sqlalchemy import update

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import TradeModel, db_manager
from indicators import get_candles
from signal_generator import generate_signals, get_usdt_symbols, analyze_single_symbol
from notifications import send_all_notifications
from engine import TradingEngine
from exceptions import APIException

# Configure logging
from logging_config import get_logger
logger = get_logger(__name__)

# Apply black background theme
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1a1a1a;
    }
    .stButton>button {
        color: #000000;
        background-color: #00ff88;
        border-color: #00ff88;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>textarea, .stSlider>div>div>div>input {
        background-color: #1a1a1a;
        color: #ffffff;
        border-color: #333333;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00ff88 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #000000;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #1a1a1a;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff88;
        color: #000000;
    }
    .stDivider {
        border-color: #333333;
    }
    .stExpanderHeader {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stExpanderContent {
        background-color: #0a0a0a;
    }
    .stPlotlyChart, .stVegaLiteChart {
        background-color: #1a1a1a;
        border-radius: 5px;
        padding: 10px;
    }
    .stMarkdown a {
        color: #00ff88;
    }
    .stMetric {
        background-color: #1a1a1a;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #333333;
    }
    .stDataFrame {
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Signals - AlgoTraderPro", page_icon="🎯", layout="wide")

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
        # Validate signal data
        symbol = signal.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            logger.error("Symbol must be a non-empty string")
            return False

        side = signal.get("side", "Buy")
        if side not in ["Buy", "Sell"]:
            logger.error(f"Invalid side for {symbol}: {side}")
            return False

        # Ensure signal has all required fields
        required_fields = ["symbol", "side", "entry", "qty", "leverage"]
        for field in required_fields:
            if field not in signal or signal[field] is None:
                logger.error(f"Missing or invalid field {field} in signal for {symbol}")
                return False

        # Prepare trade data
        trade_data = {
            "symbol": signal["symbol"],
            "side": signal["side"],
            "qty": float(signal.get("qty", 0.01)),
            "entry": float(signal["entry"]),
            "order_id": f"signal_{signal['symbol']}_{int(datetime.now().timestamp())}",
            "virtual": False,
            "status": "open",
            "strategy": signal.get("strategy", "Signal"),
            "leverage": int(signal.get("leverage", 10)),
            "sl": float(signal.get("sl", 0)) if signal.get("sl") else None,
            "tp": float(signal.get("tp", 0)) if signal.get("tp") else None,
            "trail": float(signal.get("trail", 0)) if signal.get("trail") else None,
            "liquidation": float(signal.get("liquidation", 0)) if signal.get("liquidation") else None,
            "margin_usdt": float(signal.get("margin_usdt", 0)) if signal.get("margin_usdt") else None,
            "margin_mode": "CROSS"
        }

        # Add to database
        success = db_manager.add_trade(trade_data)
        if not success:
            logger.error(f"Failed to add trade to database for {symbol}")
            st.error("❌ Failed to add trade to database")
            return False

        # Execute real trade
        try:
            success = await engine.execute_real_trade([trade_data])
            if success:
                # Sync trades after execution
                await asyncio.sleep(2)
                engine.sync_real_trades()
                engine.sync_real_balance()
                logger.info(f"Real trade executed successfully for {symbol}")
                return True
            else:
                logger.error(f"Failed to execute real trade for {symbol}")
                # Rollback database entry
                session = db_manager._get_session()
                session.execute(
                    update(TradeModel)
                    .where(TradeModel.order_id == trade_data["order_id"])
                    .values(status="failed")
                )
                session.commit()
                return False
        except APIException as e:
            if e.error_code == "100028":
                logger.warning(f"Unified account error for {symbol}: {e}. Retrying with cross margin mode.")
                trade_data["margin_mode"] = "CROSS"
                success = await engine.execute_real_trade([trade_data])
                if success:
                    await asyncio.sleep(2)
                    engine.sync_real_trades()
                    engine.sync_real_balance()
                    logger.info(f"Real trade executed successfully for {symbol} on retry")
                    return True
                else:
                    logger.error(f"Retry failed to execute real trade for {symbol}")
                    # Rollback database entry
                    session = db_manager._get_session()
                    session.execute(
                        update(TradeModel)
                        .where(TradeModel.order_id == trade_data["order_id"])
                        .values(status="failed")
                    )
                    session.commit()
                    return False
            else:
                logger.error(f"Error executing real trade for {symbol}: {e}")
                # Rollback database entry
                session = db_manager._get_session()
                session.execute(
                    update(TradeModel)
                    .where(TradeModel.order_id == trade_data["order_id"])
                    .values(status="failed")
                )
                session.commit()
                return False

    except Exception as e:
        logger.error(f"Error executing real trade for {symbol}: {e}", exc_info=True)
        st.error(f"Error executing real trade: {e}")
        # Rollback database entry if it was added
        if 'trade_data' in locals():
            session = db_manager._get_session()
            session.execute(
                update(TradeModel)
                .where(TradeModel.order_id == trade_data["order_id"])
                .values(status="failed")
            )
            session.commit()
        return False

def main():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">🎯 Trading Signals</h1>
        <p style="color: #888; margin: 0;">AI-Powered Technical Analysis & Signal Generation</p>
    </div>
    """, unsafe_allow_html=True)

    # Instructions
    with st.expander("ℹ️ How to Use Signals", expanded=False):
        st.markdown("""
        ### 📊 Signal Generation Process

        1. **Click "🔍 Generate New Signals"** to scan all configured trading pairs
        2. System analyzes technical indicators (RSI, MACD, Bollinger Bands, Volume, etc.)
        3. Machine learning scores each opportunity (0-100 scale)
        4. Top signals are displayed with recommended entry, TP, and SL levels

        ### 🎯 Understanding Signal Scores

        - **80-100**: Extremely strong signal - highest probability setup
        - **60-79**: Strong signal - favorable conditions
        - **50-59**: Moderate signal - acceptable but watch closely
        - **Below 50**: Weak signal - proceed with caution

        ### 💼 Executing Trades

        **Manual Execution:**
        - Click "Execute Trade" next to any signal
        - Trade is placed immediately at market price
        - TP and SL are automatically set based on signal levels

        **Automated Execution:**
        - Go to Trades → Automation tab
        - Set countdown timer for periodic signal generation
        - System auto-executes top signals when countdown completes

        ### 🔍 Signal Details

        Each signal shows:
        - **Symbol**: Trading pair (e.g., BTCUSDT)
        - **Type**: Buy (Long) or Sell (Short)
        - **Score**: ML-calculated signal strength
        - **Entry**: Recommended entry price
        - **TP**: Take profit target
        - **SL**: Stop loss level
        - **Timestamp**: When signal was generated

        ### ⚠️ Best Practices

        - Review multiple signals before trading
        - Higher scores don't guarantee success - manage risk
        - Use appropriate position sizing
        - Monitor market conditions before executing
        """)

    st.divider()

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

    if 'real_trade_status' not in st.session_state:
        st.session_state.real_trade_status = None

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

    # Initialize engine
    engine = st.session_state.get("engine")
    if not engine:
        st.error("Trading engine not initialized. Please restart the application.")
        return

    # Get current trading mode from session state
    trading_mode = st.session_state.get("trading_mode", "virtual")

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
                        "Score": f"{float(str(s.get('score', '0')).replace('%','') or 0):.1f}%",
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
                # Pagination for signals
                if 'main_signal_page' not in st.session_state:
                    st.session_state.main_signal_page = 0
                
                items_per_page = 9
                total_pages = (len(signals_data) - 1) // items_per_page + 1
                start_idx = st.session_state.main_signal_page * items_per_page
                end_idx = start_idx + items_per_page
                page_signals = signals_data[start_idx:end_idx]

                # Display in 3-column grid
                cols = st.columns(3)
                for idx, sig in enumerate(page_signals):
                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div style="border: 1px solid #262730; border-radius: 10px; padding: 15px; margin-bottom: 15px; background: #1E1E1E;">
                            <h4 style="margin: 0; color: #00ff88;">{sig['Symbol']}</h4>
                            <p style="margin: 5px 0;"><b>{sig['Side']}</b> | Score: <b>{sig['Score']}</b></p>
                            <p style="margin: 5px 0; font-size: 13px;">Entry: {sig['Entry']}</p>
                            <p style="margin: 5px 0; font-size: 13px;">SL: {sig['Stop Loss']} | TP: {sig['Take Profit']}</p>
                            <p style="margin: 5px 0; font-size: 12px; color: #888;">Market: {sig['Market']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Pagination controls
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⬅️ Previous", key="prev_main_sig", disabled=st.session_state.main_signal_page == 0):
                        st.session_state.main_signal_page -= 1
                        st.rerun()
                with col2:
                    st.markdown(f"<p style='text-align: center;'>Page {st.session_state.main_signal_page + 1} of {total_pages}</p>", unsafe_allow_html=True)
                with col3:
                    if st.button("Next ➡️", key="next_main_sig", disabled=st.session_state.main_signal_page >= total_pages - 1):
                        st.session_state.main_signal_page += 1
                        st.rerun()

                # Batch execution section
                st.markdown("---")
                st.subheader("📊 Batch Execute Multiple Signals")

                # Multi-select signals
                if 'selected_signal_indices' not in st.session_state:
                    st.session_state.selected_signal_indices = []

                selected_indices = st.multiselect(
                    "Select signals to execute in batch:",
                    range(len(signals)),
                    default=st.session_state.selected_signal_indices,
                    format_func=lambda idx: f"{signals[idx].get('Symbol', signals[idx].get('symbol', 'N/A'))} - {signals[idx].get('score', '0%')}"
                )
                st.session_state.selected_signal_indices = selected_indices

                if selected_indices:
                    st.info(f"✅ {len(selected_indices)} signal(s) selected for batch execution")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button(f"📈 Execute {len(selected_indices)} Virtual Trade(s)"):
                            try:
                                success_count = 0
                                for idx in selected_indices:
                                    success = engine.execute_virtual_trade(signals[idx])
                                    if success:
                                        success_count += 1
                                if success_count > 0:
                                    st.success(f"✅ {success_count}/{len(selected_indices)} virtual trade(s) executed!")
                                else:
                                    st.error("❌ Failed to execute virtual trades")
                            except Exception as e:
                                st.error(f"Error executing virtual trades: {e}")

                    with col2:
                        real_disabled = trading_mode != "real"
                        if st.button(f"💰 Execute {len(selected_indices)} Real Trade(s)", disabled=real_disabled):
                            if real_disabled:
                                st.info("Switch to real mode to execute real trades")
                            else:
                                try:
                                    if not engine.is_trading_enabled():
                                        st.error("Trading disabled or emergency stop active")
                                    else:
                                        with st.spinner(f"Executing {len(selected_indices)} real trade(s)..."):
                                            # Prepare batch signals for execution
                                            batch_signals = []
                                            for idx in selected_indices:
                                                sig = signals[idx].copy() if isinstance(signals[idx], dict) else signals[idx].__dict__.copy()

                                                # Normalize keys to lowercase and ensure all required fields
                                                normalized_sig = {
                                                    "symbol": sig.get("symbol") or sig.get("Symbol"),
                                                    "side": (sig.get("side") or sig.get("Side", "Buy")).title(),
                                                    "entry": float(sig.get("entry") or sig.get("Entry", 0)),
                                                    "qty": float(sig.get("qty", 0.01)),
                                                    "leverage": int(sig.get("leverage", 10)),
                                                    "margin_mode": "CROSS",
                                                    "sl": float(sig.get("sl", 0)) if sig.get("sl") else None,
                                                    "tp": float(sig.get("tp", 0)) if sig.get("tp") else None,
                                                    "trail": float(sig.get("trail", 0)) if sig.get("trail") else None,
                                                    "liquidation": float(sig.get("liquidation", 0)) if sig.get("liquidation") else None,
                                                    "margin_usdt": float(sig.get("margin_usdt", 0)) if sig.get("margin_usdt") else None,
                                                    "score": sig.get("score"),
                                                    "strategy": sig.get("strategy", "Signal")
                                                }

                                                # Validate required fields
                                                if normalized_sig["symbol"] and normalized_sig["entry"] > 0:
                                                    batch_signals.append(normalized_sig)
                                                else:
                                                    logger.warning(f"Skipping invalid signal: {normalized_sig}")

                                            if not batch_signals:
                                                st.error("No valid signals to execute")
                                            else:
                                                # Execute batch using engine's execute_real_trade
                                                async def execute_batch():
                                                    return await engine.execute_real_trade(batch_signals, trading_mode="real")

                                                loop = asyncio.new_event_loop()
                                                asyncio.set_event_loop(loop)
                                                success_count = loop.run_until_complete(execute_batch())
                                                loop.close()

                                                if success_count > 0:
                                                    st.success(f"✅ {success_count}/{len(batch_signals)} real trade(s) executed successfully!")
                                                    # Sync after batch execution
                                                    import time
                                                    time.sleep(2)
                                                    engine.sync_real_trades()
                                                    engine.sync_real_balance()
                                                    st.rerun()
                                                else:
                                                    st.error("❌ Failed to execute real trades")
                                except Exception as e:
                                    st.error(f"Error executing batch real trades: {e}")
                                    logger.error(f"Batch real trade error: {e}", exc_info=True)

                    with col3:
                        if st.button("🗑️ Clear Selection"):
                            st.session_state.selected_signal_indices = []
                            st.rerun()

                # Single signal analysis section
                st.markdown("---")
                st.subheader("🔍 Single Signal Analysis")

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

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📈 Execute Single Virtual Trade"):
                            try:
                                success = engine.execute_virtual_trade(
                                    selected_signal
                                )
                                if success:
                                    st.success("✅ Virtual trade executed!")
                                else:
                                    st.error("❌ Failed to execute virtual trade")
                            except Exception as e:
                                st.error(f"Error executing virtual trade: {e}")
                    with col2:
                        real_disabled = trading_mode != "real"
                        if st.button("💰 Execute Single Real Trade", disabled=real_disabled, key="execute_real_trade_tab1"):
                            if real_disabled:
                                st.info("Switch to real mode to execute real trades")
                            else:
                                try:
                                    if not engine.is_trading_enabled():
                                        st.error("Trading disabled or emergency stop active")
                                    else:
                                        with st.spinner("Executing real trade..."):
                                            # Normalize signal data
                                            sig = selected_signal.copy() if isinstance(selected_signal, dict) else selected_signal.__dict__.copy()

                                            normalized_sig = {
                                                "symbol": sig.get("symbol") or sig.get("Symbol"),
                                                "side": (sig.get("side") or sig.get("Side", "Buy")).title(),
                                                "entry": float(sig.get("entry") or sig.get("Entry", 0)),
                                                "qty": float(sig.get("qty", 0.01)),
                                                "leverage": int(sig.get("leverage", 10)),
                                                "margin_mode": "CROSS",
                                                "sl": float(sig.get("sl", 0)) if sig.get("sl") else None,
                                                "tp": float(sig.get("tp", 0)) if sig.get("tp") else None,
                                                "trail": float(sig.get("trail", 0)) if sig.get("trail") else None,
                                                "liquidation": float(sig.get("liquidation", 0)) if sig.get("liquidation") else None,
                                                "margin_usdt": float(sig.get("margin_usdt", 0)) if sig.get("margin_usdt") else None,
                                                "score": sig.get("score"),
                                                "strategy": sig.get("strategy", "Signal")
                                            }

                                            # Execute real trade
                                            async def execute_single():
                                                return await engine.execute_real_trade([normalized_sig], trading_mode="real")

                                            loop = asyncio.new_event_loop()
                                            asyncio.set_event_loop(loop)
                                            success_count = loop.run_until_complete(execute_single())
                                            loop.close()

                                            if success_count > 0:
                                                st.success(f"✅ Real trade executed for {normalized_sig.get('symbol', 'N/A')}")
                                                import time
                                                time.sleep(2)
                                                engine.sync_real_trades()
                                                engine.sync_real_balance()
                                                st.rerun()
                                            else:
                                                st.error("❌ Failed to execute real trade")
                                except Exception as e:
                                    st.error(f"Error executing real trade: {e}")
                                    logger.error(f"Real trade execution error: {e}", exc_info=True)
            else:
                st.warning("No valid signals available to display.")
        else:
            st.info("🎯 Click 'Generate New Signals' to start analyzing the markets!")

        # Check real trade status
        if st.session_state.real_trade_status:
            try:
                if st.session_state.real_trade_status.done():
                    success = st.session_state.real_trade_status.result()
                    if success:
                        st.success("✅ Real trade executed successfully")
                    else:
                        st.error("❌ Real trade execution failed")
                    st.session_state.real_trade_status = None
            except Exception as e:
                st.error(f"Error checking real trade status: {e}")
                st.session_state.real_trade_status = None

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
        with trade_col1:
            virtual_disabled = 'single_analysis' not in st.session_state
            if st.button("💻 Execute Virtual Trade", disabled=virtual_disabled):
                try:
                    success = engine.execute_virtual_trade(
                        st.session_state['single_analysis']
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
                        async def execute_single_real_trade():
                            sig = st.session_state['single_analysis'].copy()
                            sig["symbol"] = sig.get("symbol") or sig.get("Symbol")
                            sig["side"] = sig.get("side") or sig.get("Side")
                            sig["entry"] = sig.get("entry") or sig.get("Entry")
                            sig["qty"] = sig.get("qty", 0.01)
                            sig["leverage"] = sig.get("leverage", 10)
                            sig["margin_mode"] = "CROSS"

                            success_count = await engine.execute_real_trade([sig], trading_mode="real")
                            if success_count > 0:
                                await asyncio.sleep(2)
                                engine.sync_real_trades()
                                engine.sync_real_balance()
                                st.success(f"✅ Real trade executed for {sig.get('symbol', 'N/A')}")
                            else:
                                st.error("❌ Failed to execute real trade")
                            return success_count

                        with st.spinner("Executing real trade..."):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            success_count = loop.run_until_complete(execute_single_real_trade())
                            loop.close()
                            if success_count > 0:
                                st.rerun()
                except Exception as e:
                    st.error(f"Real trade error: {e}")
                    logger.error(f"Real trade execution error: {e}")

        # Check real trade status
        if st.session_state.real_trade_status:
            try:
                if st.session_state.real_trade_status.done():
                    success = st.session_state.real_trade_status.result()
                    if success:
                        st.success("✅ Real trade executed successfully")
                    else:
                        st.error("❌ Real trade execution failed")
                    st.session_state.real_trade_status = None
            except Exception as e:
                st.error(f"Error checking real trade status: {e}")
                st.session_state.real_trade_status = None

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
    # Initialize session state for theme if not already present
    if "theme" not in st.session_state:
        st.session_state.theme = "dark" # Default to dark theme

    # Initialize trading engine
    if "engine" not in st.session_state:
        try:
            st.session_state.engine = TradingEngine()
            logger.info("Trading engine initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            st.error("Error initializing the trading engine. Please check server logs.")
            st.stop()

    main()