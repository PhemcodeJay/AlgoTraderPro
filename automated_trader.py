import asyncio
import logging
import time
import json
import streamlit as st
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from engine import TradingEngine
from bybit_client import BybitClient
from signal_generator import generate_signals, get_usdt_symbols
from db import db_manager, Trade, TradeModel
from settings import load_settings
from logging_config import get_trading_logger
from exceptions import APIException
from sqlalchemy import update

logger = get_trading_logger(__name__)

@st.cache_resource
def get_automated_trader(_engine: TradingEngine) -> 'AutomatedTrader':
    """Cached factory for AutomatedTrader instance."""
    client = st.session_state.get("bybit_client", None)
    if not client:
        logger.warning("Bybit client not initialized, creating new instance")
        client = BybitClient()
        st.session_state.bybit_client = client
    return AutomatedTrader(_engine, client)

class AutomatedTrader:
    def __init__(self, engine: TradingEngine, client: BybitClient):
        self.engine = engine
        self.client = client
        self.db = db_manager
        
        self.is_running = False
        self.task = None
        self.stop_event = asyncio.Event()
        
        # Trading parameters
        settings = load_settings()
        self.scan_interval = settings.get("SCAN_INTERVAL", 300)  # Default 5 min
        self.top_n_signals = settings.get("TOP_N_SIGNALS", 5)
        self.max_positions = self.engine.max_open_positions
        self.risk_per_trade = self.engine.max_risk_per_trade
        self.leverage = self.engine.settings.get("LEVERAGE", 15)
        
        # Statistics
        self.stats = {
            "signals_generated": 0,
            "trades_executed": 0,
            "profitable_trades": 0,
            "total_pnl": 0.0,
            "start_time": None,
            "last_scan": None
        }
        
        # Store in session state for emergency stop
        st.session_state.automated_trader = self
        
        logger.info("Automated trader initialized", extra={
            'scan_interval': self.scan_interval,
            'max_positions': self.max_positions,
            'risk_per_trade': self.risk_per_trade,
            'leverage': self.leverage
        })

    async def start(self, status_container=None) -> bool:
        """Start automated trading with enhanced error handling"""
        try:
            if self.is_running:
                logger.warning("Automated trader already running")
                if status_container:
                    status_container.warning("Automated trader already running")
                return False
            
            if not self.engine.is_trading_enabled():
                logger.error("Cannot start: Trading engine is disabled or in emergency stop")
                if status_container:
                    status_container.error("Trading engine is disabled or in emergency stop")
                return False
            
            # Verify database connection
            if not self.db.is_connected():
                logger.error("Database not connected")
                if status_container:
                    status_container.error("Failed to connect to database")
                return False
            
            # Verify Bybit client connection for real mode
            trading_mode = st.session_state.get("trading_mode", "virtual")
            if trading_mode == "real" and (not self.client or not self.client.is_connected()):
                logger.error("Bybit client not connected for real mode")
                if status_container:
                    status_container.error("Bybit client not connected. Check API keys in .env file.")
                return False

            self.is_running = True
            self.stop_event.clear()
            self.stats["start_time"] = datetime.now(timezone.utc)
            
            # Start the main trading loop
            self.task = asyncio.create_task(self._trading_loop(status_container))
            
            logger.info(f"Automated trading started in {trading_mode} mode")
            if status_container:
                status_container.success(f"🤖 Automated trading started in {trading_mode.title()} mode")
            return True
            
        except Exception as e:
            logger.error(f"Error starting automated trader: {e}", exc_info=True)
            self.is_running = False
            if status_container:
                status_container.error(f"Failed to start trading: {e}")
            return False

    async def stop(self) -> bool:
        """Stop automated trading"""
        try:
            if not self.is_running:
                logger.info("Automated trader already stopped")
                return True
            
            self.is_running = False
            self.stop_event.set()
            
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Automated trading stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping automated trader: {e}", exc_info=True)
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        try:
            trading_mode = st.session_state.get("trading_mode", "virtual")
            return {
                "is_running": self.is_running,
                "stats": self.stats.copy(),
                "current_positions": len(self.engine.get_open_virtual_trades()) if trading_mode == "virtual" else len(self.engine.get_open_real_trades()),
                "max_positions": self.max_positions,
                "scan_interval": self.scan_interval,
                "risk_per_trade": self.risk_per_trade,
                "leverage": self.leverage
            }
        except Exception as e:
            logger.error(f"Error getting trader status: {e}", exc_info=True)
            return {}

    async def reset_stats(self):
        """Reset trading statistics"""
        try:
            self.stats = {
                "signals_generated": 0,
                "trades_executed": 0,
                "profitable_trades": 0,
                "total_pnl": 0.0,
                "start_time": datetime.now(timezone.utc),
                "last_scan": None
            }
            logger.info("Trading statistics reset")
        except Exception as e:
            logger.error(f"Error resetting stats: {e}", exc_info=True)

    async def _trading_loop(self, status_container=None):
        """Main automated trading loop with live countdown and effective rescan trigger"""
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    trading_mode = st.session_state.get("trading_mode", "virtual")
                    
                    # Notify scan start
                    if status_container:
                        status_container.info(f"🤖 Scanning markets in {trading_mode.title()} mode... Last scan: {self.stats.get('last_scan', 'Never')}")
                    logger.info(f"Starting new market scan in {trading_mode.title()} mode")

                    # Sync balance for real trading
                    if trading_mode == "real":
                        balance = self.engine.sync_real_balance()
                        if not balance or balance["available"] <= 0:
                            logger.warning(f"Failed to sync real balance or insufficient funds: {balance}")
                            if status_container:
                                status_container.warning("Failed to sync real balance or insufficient funds, retrying in 60s")
                            await asyncio.sleep(60)
                            continue
                        logger.info(f"Synced real balance: ${balance['available']:.2f} available")
                        self.engine.get_open_real_trades()

                    # 1️⃣ Scan for signals and trade
                    await self._scan_and_trade(trading_mode)

                    # 2️⃣ Monitor existing positions
                    await self._monitor_positions(trading_mode)

                    # 3️⃣ Update timestamp
                    self.stats["last_scan"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                    # 4️⃣ Countdown until next scan
                    next_scan_time = time.time() + self.scan_interval
                    logger.info(f"Waiting {self.scan_interval//60} minutes until next scan")
                    while time.time() < next_scan_time and self.is_running and not self.stop_event.is_set():
                        remaining = int(next_scan_time - time.time())
                        hours, rem = divmod(remaining, 3600)
                        mins, secs = divmod(rem, 60)
                        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"

                        # Show countdown in terminal
                        print(f"\r🕒 Next scan in {time_str} (hh:mm:ss)", end="", flush=True)

                        # Update Streamlit (optional)
                        if status_container:
                            status_container.info(
                                f"🕒 Next scan in {time_str} | Last scan: {self.stats.get('last_scan', 'Never')}"
                            )

                        await asyncio.sleep(1)

                    logger.info("Countdown finished — initiating next market scan")

                except Exception as e:
                    logger.error(f"Error in trading loop: {e}", exc_info=True)
                    if status_container:
                        status_container.warning(f"Trading loop error: {e}")
                    await asyncio.sleep(60)  # Retry after 1 minute

        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.critical(f"Unexpected trading loop error: {e}", exc_info=True)
            if status_container:
                status_container.error(f"Critical trading loop error: {e}")

    async def _scan_and_trade(self, trading_mode: str):
        """Scan for signals and execute trades"""
        try:
            symbols = get_usdt_symbols(limit=100, _trading_mode=trading_mode)
            signals = generate_signals(symbols, interval="60", top_n=self.top_n_signals, _trading_mode=trading_mode)
            self.stats["signals_generated"] += len(signals)
            
            if not signals:
                logger.info("No signals generated")
                return
            
            # Get current positions count and balance
            current_positions = len(self.engine.get_open_virtual_trades()) if trading_mode == "virtual" else len(self.engine.get_open_real_trades())
            balance = self.engine.sync_real_balance() if trading_mode == "real" else {"available": 100.0}
            available_balance = balance.get("available", 0.0)
            logger.info(f"Current positions: {current_positions}, Available balance: ${available_balance:.2f}")
            
            # Filter signals to respect max positions and balance
            max_new_trades = self.max_positions - current_positions
            filtered_signals = []
            
            for signal in signals[:max_new_trades]:
                symbol = signal.get("symbol")
                if not symbol:
                    logger.warning("Skipping signal with missing symbol")
                    continue
                
                # Calculate position size
                position_size = self.engine.calculate_position_size(
                    symbol, 
                    signal.get("entry", 0), 
                    signal.get("sl", 0),
                    available_balance=available_balance
                )
                if position_size <= 0:
                    logger.warning(f"Skipping {symbol}: Invalid position size {position_size}")
                    continue
                
                # Risk check (relaxed to allow smaller trades)
                risk_amount = position_size * signal.get("entry", 0) / self.leverage
                if risk_amount > available_balance * self.risk_per_trade:
                    logger.warning(f"Risk too high for {symbol}: {risk_amount:.2f} > {available_balance * self.risk_per_trade:.2f}")
                    continue
                
                # Convert signal to dict if needed
                signal_dict = signal if isinstance(signal, dict) else signal.to_dict()
                filtered_signals.append(signal_dict)
                logger.info(f"Valid signal for {symbol}: size={position_size:.6f}, risk=${risk_amount:.2f}")
            
            if not filtered_signals:
                logger.info("No valid signals after filtering")
                return
            
            # Execute trades based on mode
            if trading_mode == "virtual":
                for signal_dict in filtered_signals:
                    symbol = signal_dict.get("symbol")
                    success = self.engine.execute_virtual_trade(signal_dict, trading_mode)
                    if success:
                        self.stats["trades_executed"] += 1
                        logger.info(f"Virtual trade executed for {symbol}")
                    else:
                        logger.error(f"Failed to execute virtual trade for {symbol}")
            else:
                if not self.client.is_connected():
                    logger.error("Bybit client not connected for real trades")
                    if st._is_running_with_streamlit:
                        st.error("Bybit client not connected for real trades")
                    return
                
                try:
                    success = await self.engine.execute_real_trades(filtered_signals, trading_mode)
                    if success:
                        self.stats["trades_executed"] += len(filtered_signals)
                        # Wait briefly for Bybit to process
                        await asyncio.sleep(2)
                        # Sync real trades to DB after execution
                        self.engine.get_open_real_trades()
                        self.engine.sync_real_balance()
                        logger.info(f"Batch executed {len(filtered_signals)} real trades and synced to DB")
                    else:
                        logger.error("Failed to execute batch of real trades")
                except APIException as e:
                    if e.error_code == "100028":
                        logger.warning(f"Unified account error: {e}. Retrying with cross margin mode.")
                        for signal_dict in filtered_signals:
                            signal_dict["margin_mode"] = "CROSS"
                        success = await self.engine.execute_real_trades(filtered_signals, trading_mode)
                        if success:
                            self.stats["trades_executed"] += len(filtered_signals)
                            await asyncio.sleep(2)
                            self.engine.get_open_real_trades()
                            self.engine.sync_real_balance()
                            logger.info(f"Batch executed {len(filtered_signals)} real trades on retry and synced to DB")
                        else:
                            logger.error("Failed to execute batch of real trades on retry")
                    else:
                        logger.error(f"Error executing real trades: {e}", exc_info=True)
                        if st._is_running_with_streamlit:
                            st.error(f"Error executing real trades: {e}")
            
        except Exception as e:
            logger.error(f"Error in scan and trade: {e}", exc_info=True)
            if st._is_running_with_streamlit:
                st.error(f"Error scanning and trading: {e}")

    async def _monitor_positions(self, trading_mode: str):
        """Monitor and manage existing positions"""
        try:
            open_trades = self.engine.get_open_virtual_trades() if trading_mode == "virtual" else self.engine.get_open_real_trades()
            
            for trade in open_trades:
                try:
                    await self._check_trade_exit(trade, trading_mode)
                except Exception as e:
                    logger.error(f"Error monitoring trade {trade.order_id}: {e}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}", exc_info=True)
            if st._is_running_with_streamlit:
                st.error(f"Error monitoring positions: {e}")

    async def _check_trade_exit(self, trade: Trade, trading_mode: str):
        """Check if trade should be closed"""
        try:
            symbol = trade.symbol
            current_price = self.client.get_current_price(symbol)
            
            if current_price <= 0:
                logger.warning(f"Invalid current price for {symbol}: {current_price}")
                return
            
            entry_price = trade.entry_price
            side = trade.side.upper()
            
            # Calculate current PnL (for logging/stats)
            if side in ["BUY", "LONG"]:
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            # For real trades, check if Bybit closed the position
            if trading_mode == "real":
                if not self.client.is_connected():
                    logger.warning(f"Bybit client not connected for checking {symbol}")
                    return
                try:
                    positions = await self.client.get_positions(symbol=symbol)
                    position = next((p for p in positions if p["size"] > 0 and p["side"].upper() == side), None)
                    if not position:
                        should_exit = True
                        exit_reason = "Bybit Closed (SL/TP)"
                        current_price = self.client.get_current_price(symbol)
                    else:
                        # Update leverage from position
                        leverage_from_api = position.get("leverage")
                        if leverage_from_api is not None and float(leverage_from_api) > 0:
                            trade.leverage = int(leverage_from_api)
                            logger.debug(f"Updated leverage for {symbol} to {trade.leverage} from Bybit API")
                        else:
                            logger.warning(f"Invalid or missing leverage for {symbol} in API response, retaining {trade.leverage}")
                except Exception as e:
                    logger.warning(f"Failed to check Bybit position for {symbol}: {e}")
                    # Fallback to manual SL/TP check
                    if trade.sl is None or trade.tp is None:
                        logger.warning(f"Missing SL/TP for real trade {trade.order_id}")
                        return
                    if side in ["BUY", "LONG"]:
                        should_exit_sl = current_price <= trade.sl
                        should_exit_tp = current_price >= trade.tp
                    else:
                        should_exit_sl = current_price >= trade.sl
                        should_exit_tp = current_price <= trade.tp
                    if should_exit_sl:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif should_exit_tp:
                        should_exit = True
                        exit_reason = "Take Profit"
            else:
                # Virtual trades: Use stored SL/TP
                if trade.sl is None or trade.tp is None:
                    logger.warning(f"Missing SL/TP for virtual trade {trade.order_id}")
                    return
                if side in ["BUY", "LONG"]:
                    should_exit_sl = current_price <= trade.sl
                    should_exit_tp = current_price >= trade.tp
                else:
                    should_exit_sl = current_price >= trade.sl
                    should_exit_tp = current_price <= trade.tp
                if should_exit_sl:
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif should_exit_tp:
                    should_exit = True
                    exit_reason = "Take Profit"
            
            # Time-based exit (24 hours)
            if not should_exit and trade.timestamp:
                hours_open = (datetime.now(timezone.utc) - trade.timestamp).total_seconds() / 3600
                if hours_open >= 24:
                    should_exit = True
                    exit_reason = "Time Limit"
            
            if should_exit:
                logger.info(f"Triggering exit for {symbol}: Reason={exit_reason}, PnL={pnl_pct:.2f}%")
                await self._close_trade(trade, current_price, exit_reason, trading_mode)
                    
        except Exception as e:
            logger.error(f"Error checking trade exit for {trade.symbol}: {e}", exc_info=True)
            if st._is_running_with_streamlit:
                st.error(f"Error checking trade exit for {trade.symbol}: {e}")

    async def _close_trade(self, trade: Trade, exit_price: float, reason: str, trading_mode: str):
        """Close a trade"""
        try:
            # Calculate PnL
            trade_dict = trade.to_dict()
            if trading_mode == "virtual":
                pnl = self.engine.calculate_virtual_pnl(trade_dict)
            else:
                if not self.client.is_connected():
                    logger.error("Bybit client not connected for real trade close")
                    if st._is_running_with_streamlit:
                        st.error("Bybit client not connected for real trade close")
                    return
                positions = await self.client.get_positions(symbol=trade.symbol)
                pnl = float(positions[0].get("unrealisedPnl", 0.0)) if positions else 0.0

            # For real mode, place closing order
            if trading_mode == "real":
                close_side = "Sell" if trade.side.upper() in ["BUY", "LONG"] else "Buy"
                try:
                    close_result = await self.client.place_order(
                        symbol=trade.symbol,
                        side=close_side,
                        qty=trade.qty,
                        leverage=trade.leverage,
                        mode="CROSS"
                    )
                    if "error" in close_result:
                        logger.error(f"Failed to close real position {trade.order_id}: {close_result['error']}")
                        if st._is_running_with_streamlit:
                            st.error(f"Failed to close real position {trade.order_id}: {close_result['error']}")
                        return
                    logger.info(f"Closed real position {trade.order_id}")
                    await asyncio.sleep(2)
                    self.engine.get_open_real_trades()
                    logger.info(f"Synced real trades to DB after closing {trade.order_id}")
                except APIException as e:
                    if e.error_code == "100028":
                        logger.warning(f"Unified account error closing {trade.order_id}: {e}. Retrying with cross margin mode.")
                        close_result = await self.client.place_order(
                            symbol=trade.symbol,
                            side=close_side,
                            qty=trade.qty,
                            leverage=trade.leverage,
                            mode="CROSS"
                        )
                        if "error" in close_result:
                            logger.error(f"Retry failed to close real position {trade.order_id}: {close_result['error']}")
                            if st._is_running_with_streamlit:
                                st.error(f"Retry failed to close real position {trade.order_id}: {close_result['error']}")
                            return
                        logger.info(f"Closed real position {trade.order_id} on retry")
                        await asyncio.sleep(2)
                        self.engine.get_open_real_trades()
                        logger.info(f"Synced real trades to DB after closing {trade.order_id}")
                    else:
                        logger.error(f"Error closing real position {trade.order_id}: {e}")
                        if st._is_running_with_streamlit:
                            st.error(f"Error closing real position {trade.order_id}: {e}")
                        return

            # Update trade in database
            if not self.db.session:
                logger.error("Database session not initialized")
                if st._is_running_with_streamlit:
                    st.error("Database session not initialized")
                return
            
            success = False
            try:
                self.db.session.execute(
                    update(TradeModel)
                    .where(TradeModel.id == int(trade.id))  # Use integer ID
                    .values(
                        status="closed",
                        exit_price=exit_price,
                        pnl=pnl,
                        closed_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                )
                self.db.session.commit()
                success = True
                logger.info(f"Updated trade ID {trade.id} to closed in database")
            except Exception as e:
                self.db.session.rollback()
                logger.error(f"Database error updating trade ID {trade.id}: {e}", exc_info=True)
                if st._is_running_with_streamlit:
                    st.error(f"Database error updating trade ID {trade.id}: {e}")
            
            if success:
                self.stats["total_pnl"] += pnl
                if pnl > 0:
                    self.stats["profitable_trades"] += 1
                
                if trading_mode == "virtual":
                    self.engine.update_virtual_balances(pnl, mode=trading_mode)
                else:
                    self.engine.sync_real_balance()
                
                logger.info(f"Trade closed: {trade.symbol} {reason} PnL: ${pnl:.2f}, Mode: {trading_mode}")
                if st._is_running_with_streamlit:
                    st.success(f"✅ Trade closed: {trade.symbol} ({reason}) PnL: ${pnl:.2f}")
                st.cache_data.clear()  # Clear cache for UI refresh
            else:
                logger.error(f"Failed to close trade ID {trade.id}")
                if st._is_running_with_streamlit:
                    st.error(f"Failed to close trade ID {trade.id}")
                
        except Exception as e:
            logger.error(f"Error closing trade ID {trade.id}: {e}", exc_info=True)
            if st._is_running_with_streamlit:
                st.error(f"Error closing trade ID {trade.id}: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            total_trades = self.stats["trades_executed"]
            profitable_trades = self.stats["profitable_trades"]
            
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = self.stats["total_pnl"] / total_trades if total_trades > 0 else 0
            
            runtime = None
            if self.stats["start_time"]:
                runtime = datetime.now(timezone.utc) - self.stats["start_time"]
            
            return {
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "win_rate": round(win_rate, 1),
                "total_pnl": round(self.stats["total_pnl"], 2),
                "avg_pnl": round(avg_pnl, 2),
                "runtime": str(runtime).split(".")[0] if runtime else "Not started",
                "signals_generated": self.stats["signals_generated"],
                "last_scan": self.stats["last_scan"]
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}", exc_info=True)
            return {}