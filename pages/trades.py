import streamlit as st
import pandas as pd
import asyncio
import sys
import os
from datetime import datetime, timezone

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_engine import TradingEngine
from db import db_manager, Trade
from automated_trader import AutomatedTrader
from utils import calculate_portfolio_metrics
from signal_generator import get_usdt_symbols, generate_signals
from settings import load_settings, save_settings

# Configure logging
from logging_config import get_logger
logger = get_logger(__name__)

st.set_page_config(
    page_title="Trades - AlgoTrader Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

def close_trade_safely(trade_id: str, virtual: bool = True):
    """Close a trade with proper error handling"""
    try:
        engine = st.session_state.get("engine")
        if not engine:
            st.error("Trading engine not initialized")
            logger.error("Trading engine not initialized")
            return False
        
        # Ensure database session is initialized
        if not db_manager.session:
            try:
                db_manager.init_session()
            except Exception as e:
                st.error("Failed to initialize database session")
                logger.error(f"Failed to initialize database session: {e}", exc_info=True)
                return False
        
        # Get trade from database
        open_trades = [t for t in db_manager.get_trades(limit=1000) if t.status == "open"]
        trade = next((t for t in open_trades if str(t.id) == str(trade_id) or str(t.order_id) == str(trade_id)), None)
        
        if not trade:
            st.error(f"Trade {trade_id} not found")
            logger.error(f"Trade {trade_id} not found")
            return False
        
        # Get current price for PnL calculation
        current_price = engine.client.get_current_price(trade.symbol)
        if not current_price:
            st.error(f"Failed to fetch current price for {trade.symbol}")
            logger.error(f"Failed to fetch current price for {trade.symbol}")
            return False
        
        # Calculate PnL
        trade_dict = trade.to_dict()
        trade_dict["exit_price"] = current_price
        trade_dict["closed_at"] = datetime.now(timezone.utc)
        pnl = engine.calculate_pnl(trade_dict)
        
        # Update trade in database
        try:
            trade.status = "closed"
            trade.exit_price = current_price
            trade.pnl = pnl
            trade.closed_at = datetime.now(timezone.utc)
            db_manager.session.add(trade)
            db_manager.session.commit()
            success = True
        except Exception as e:
            db_manager.session.rollback()
            st.error(f"Database error updating trade {trade.order_id}: {e}")
            logger.error(f"Database error updating trade {trade.order_id}: {e}", exc_info=True)
            return False
        
        if success:
            # Update virtual balance if it's a virtual trade
            if virtual:
                engine.update_virtual_balances(pnl, mode="virtual")
            
            # Update wallet cache
            st.session_state.wallet_cache.clear()
            st.success(f"✅ Trade closed successfully! PnL: ${pnl:.2f}")
            logger.info(f"Trade {trade.order_id} closed successfully, PnL: ${pnl:.2f}")
            return True
        else:
            st.error("❌ Failed to close trade in database")
            logger.error(f"Failed to close trade {trade.order_id} in database")
            return False
            
    except Exception as e:
        st.error(f"Error closing trade: {e}")
        logger.error(f"Error closing trade {trade_id}: {e}", exc_info=True)
        return False

def display_manual_trading():
    """Display manual trading controls"""
    st.subheader("📤 Manual Trading")
    symbols = get_usdt_symbols()
    symbol = st.selectbox("Select Symbol", symbols)
    side = st.selectbox("Side", ["BUY", "SELL"])
    qty = st.number_input("Quantity", min_value=0.0, step=0.01)
    entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01)
    take_profit = st.number_input("Take Profit", min_value=0.0, step=0.01)
    stop_loss = st.number_input("Stop Loss", min_value=0.0, step=0.01)
    
    if st.button("Execute Manual Trade"):
        try:
            engine = st.session_state.get("engine")
            if not engine:
                st.error("Trading engine not initialized")
                logger.error("Trading engine not initialized")
                return
            
            # Ensure database session
            if not db_manager.session:
                db_manager.init_session()
            
            trade_data = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_price": entry_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "virtual": st.session_state.get("trading_mode") == "virtual",
                "timestamp": datetime.now(timezone.utc),
                "status": "open",
                "order_id": str(uuid.uuid4()),
                "strategy": "Manual",
                "leverage": engine.settings.get("LEVERAGE", 10)
            }
            trade = Trade(**trade_data)
            db_manager.session.add(trade)
            db_manager.session.commit()
            st.success(f"✅ Manual trade placed for {symbol} ({side})")
            logger.info(f"Manual trade placed: {trade_data}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to execute manual trade: {e}")
            logger.error(f"Failed to execute manual trade: {e}", exc_info=True)

def display_automation_tab():
    """Display automation controls"""
    st.subheader("🤖 Automation Controls")
    settings = load_settings()
    auto_trading_enabled = settings.get("AUTO_TRADING_ENABLED", True)
    
    if st.checkbox("Enable Auto Trading", value=auto_trading_enabled):
        if not auto_trading_enabled:
            new_settings = settings.copy()
            new_settings["AUTO_TRADING_ENABLED"] = True
            if save_settings(new_settings):
                st.success("✅ Auto trading enabled")
                logger.info("Auto trading enabled")
                st.rerun()
            else:
                st.error("❌ Failed to save auto trading setting")
                logger.error("Failed to save auto trading setting")
    else:
        if auto_trading_enabled:
            new_settings = settings.copy()
            new_settings["AUTO_TRADING_ENABLED"] = False
            if save_settings(new_settings):
                st.success("✅ Auto trading disabled")
                logger.info("Auto trading disabled")
                st.rerun()
            else:
                st.error("❌ Failed to save auto trading setting")
                logger.error("Failed to save auto trading setting")
    
    if st.button("Run Signal Scan"):
        try:
            engine = st.session_state.get("engine")
            if not engine:
                st.error("Trading engine not initialized")
                logger.error("Trading engine not initialized")
                return
            trader = AutomatedTrader(engine, engine.client)
            # Simulate single scan by generating and processing signals
            signals = generate_signals()
            for signal in signals:
                trade_data = {
                    "symbol": signal.symbol,
                    "side": signal.side,
                    "qty": signal.margin_usdt / signal.entry if signal.entry and signal.margin_usdt else 0.01,
                    "entry_price": signal.entry,
                    "take_profit": signal.tp,
                    "stop_loss": signal.sl,
                    "virtual": st.session_state.get("trading_mode") == "virtual",
                    "timestamp": datetime.now(timezone.utc),
                    "status": "open",
                    "order_id": str(uuid.uuid4()),
                    "strategy": signal.strategy,
                    "leverage": signal.leverage,
                    "score": signal.score
                }
                if trade_data["qty"] > 0:
                    trade = Trade(**trade_data)
                    db_manager.session.add(trade)
                    db_manager.session.commit()
                    logger.info(f"Auto trade placed: {trade_data}")
            st.success(f"✅ Signal scan completed, processed {len(signals)} signals")
            logger.info(f"Signal scan completed, processed {len(signals)} signals")
            st.rerun()
        except Exception as e:
            st.error(f"Error running signal scan: {e}")
            logger.error(f"Error running signal scan: {e}", exc_info=True)

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

        st.markdown("### 💼 Trade Management")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔄 Open Trades",
            "📜 Trade History",
            "📤 Manual Trading",
            "🤖 Automation",
            "📊 Statistics"
        ])
        
        with tab1:
            st.subheader("🔄 Open Trades")
            open_trades = [t for t in db_manager.get_trades(limit=1000) if t.status == "open"]
            if open_trades:
                trade_data = []
                for trade in open_trades:
                    trade_data.append({
                        "Trade ID": trade.order_id,
                        "Symbol": trade.symbol or "N/A",
                        "Side": trade.side or "N/A",
                        "Entry Price": f"${trade.entry_price:.4f}" if trade.entry_price else "$0.0000",
                        "Quantity": f"{trade.qty:.4f}" if trade.qty else "0.0000",
                        "Status": trade.status.title() if trade.status else "N/A",
                        "Timestamp": trade.timestamp.strftime("%Y-%m-%d %H:%M:%S") if trade.timestamp else "N/A"
                    })
                df = pd.DataFrame(trade_data)
                st.dataframe(df)
                
                trade_id = st.selectbox("Select Trade to Close", [t["Trade ID"] for t in trade_data])
                if st.button("Close Selected Trade"):
                    if trade_id:
                        close_trade_safely(trade_id, virtual=(st.session_state.trading_mode == "virtual"))
                        st.rerun()
                    else:
                        st.warning("Please select a trade to close")
                        logger.warning("No trade selected for closing")
            else:
                st.info("No open trades")
        
        with tab2:
            st.subheader("📜 Trade History")
            closed_trades = [t for t in db_manager.get_trades(limit=1000) if t.status == "closed"]
            if closed_trades:
                trade_data = []
                for trade in closed_trades:
                    trade_data.append({
                        "Trade ID": trade.order_id,
                        "Symbol": trade.symbol or "N/A",
                        "Side": trade.side or "N/A",
                        "Entry Price": f"${trade.entry_price:.4f}" if trade.entry_price else "$0.0000",
                        "Exit Price": f"${trade.exit_price:.4f}" if trade.exit_price else "$0.0000",
                        "PnL": f"${trade.pnl:.2f}" if trade.pnl is not None else "N/A",
                        "Status": trade.status.title() if trade.status else "N/A",
                        "Closed At": trade.closed_at.strftime("%Y-%m-%d %H:%M:%S") if trade.closed_at else "N/A"
                    })
                df = pd.DataFrame(trade_data)
                st.dataframe(df)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Trade History",
                    csv,
                    "trade_history.csv",
                    "text/csv"
                )
            else:
                st.info("No trade history available. Start trading to see your history here!")
        
        with tab3:
            display_manual_trading()
        
        with tab4:
            display_automation_tab()
        
        with tab5:
            st.subheader("📊 Trading Statistics")
            
            all_trades = db_manager.get_trades(limit=1000)
            if all_trades:
                trade_dicts = [t.to_dict() for t in all_trades]
                metrics = calculate_portfolio_metrics(trade_dicts)
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Total Trades", metrics['total_trades'])
                with metric_col2:
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                with metric_col3:
                    st.metric("Total P&L", f"${metrics['total_pnl']:.2f}")
                with metric_col4:
                    st.metric("Avg P&L/Trade", f"${metrics['avg_pnl']:.2f}")
                
                st.markdown("### 🎯 Detailed Statistics")
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.metric("Profitable Trades", metrics['profitable_trades'])
                    st.metric("Best Trade", f"${metrics['best_trade']:.2f}")
                with detail_col2:
                    losing_trades = metrics['total_trades'] - metrics['profitable_trades']
                    st.metric("Losing Trades", losing_trades)
                    st.metric("Worst Trade", f"${metrics['worst_trade']:.2f}")
                
                st.markdown("### 📈 Performance by Symbol")
                symbol_performance = {}
                for trade in trade_dicts:
                    symbol = trade.get("symbol", "N/A")
                    pnl = trade.get("pnl", 0) or 0
                    if symbol not in symbol_performance:
                        symbol_performance[symbol] = {'trades': 0, 'total_pnl': 0}
                    symbol_performance[symbol]['trades'] += 1
                    symbol_performance[symbol]['total_pnl'] += pnl
                
                if symbol_performance:
                    symbol_data = []
                    for symbol, data in symbol_performance.items():
                        symbol_data.append({
                            "Symbol": symbol,
                            "Trades": data['trades'],
                            "Total PnL": f"${data['total_pnl']:.2f}",
                            "Avg PnL": f"${data['total_pnl'] / data['trades']:.2f}"
                        })
                    st.dataframe(pd.DataFrame(symbol_data))
            else:
                st.info("No trading statistics available. Complete some trades to see detailed analytics!")

    except Exception as e:
        st.error(f"Error loading trades page: {e}")
        logger.error(f"Trades page error: {e}", exc_info=True)
        
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