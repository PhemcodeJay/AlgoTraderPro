import asyncio
import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from engine import TradingEngine
from bybit_client import BybitClient
from signal_generator import generate_signals, get_usdt_symbols
from db import db_manager, Trade, Signal
from settings import load_settings
from logging_config import get_trading_logger

logger = get_trading_logger(__name__)

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
        self.leverage = self.engine.settings.get("LEVERAGE", 10)
        
        # Statistics
        self.stats = {
            "signals_generated": 0,
            "trades_executed": 0,
            "profitable_trades": 0,
            "total_pnl": 0.0,
            "start_time": None,
            "last_scan": None
        }
        
        logger.info("Automated trader initialized", extra={
            'scan_interval': self.scan_interval,
            'max_positions': self.max_positions,
            'risk_per_trade': self.risk_per_trade,
            'leverage': self.leverage
        })

    async def start(self, status_container=None) -> bool:
        """Start automated trading"""
        try:
            if self.is_running:
                logger.warning("Automated trader already running")
                return False
            
            if not self.engine.is_trading_enabled():
                logger.warning("Cannot start: Trading engine is disabled or in emergency stop")
                return False
            
            self.is_running = True
            self.stop_event.clear()
            self.stats["start_time"] = datetime.now(timezone.utc)
            
            # Start the main trading loop
            self.task = asyncio.create_task(self._trading_loop(status_container))
            
            logger.info("Automated trading started")
            if status_container:
                status_container.success("🤖 Automated trading started")
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
            return {
                "is_running": self.is_running,
                "stats": self.stats.copy(),
                "current_positions": len(self.engine.get_open_virtual_trades()) + len(self.engine.get_open_real_trades()),
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
        """Main automated trading loop"""
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    # Update status
                    if status_container:
                        status_container.info(f"🤖 Scanning markets... Last scan: {self.stats.get('last_scan', 'Never')}")
                    
                    # Sync real balance if in real mode
                    trading_mode = db_manager.get_setting("trading_mode") or "virtual"
                    if trading_mode == "real":
                        self.engine.sync_real_balance()
                        self.engine.get_open_real_trades()
                    
                    # Scan for signals
                    await self._scan_and_trade(trading_mode)
                    
                    # Monitor existing positions
                    await self._monitor_positions(trading_mode)
                    
                    # Update last scan time
                    self.stats["last_scan"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Wait for next scan
                    await asyncio.sleep(self.scan_interval)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}", exc_info=True)
                    if status_container:
                        status_container.warning(f"Trading loop error: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
                    
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Unexpected trading loop error: {e}", exc_info=True)

    async def _scan_and_trade(self, trading_mode: str):
        """Scan for signals and execute trades"""
        try:
            symbols = get_usdt_symbols(limit=50)
            signals = generate_signals(symbols, interval="60", top_n=self.top_n_signals, trading_mode=trading_mode)
            self.stats["signals_generated"] += len(signals)
            
            # Get current positions count
            current_positions = len(self.engine.get_open_virtual_trades()) if trading_mode == "virtual" else len(self.engine.get_open_real_trades())
            
            for signal in signals:
                symbol = signal.get("symbol")
                if not symbol:
                    continue
                    
                if current_positions >= self.max_positions:
                    logger.info(f"Max positions reached ({self.max_positions}), skipping {symbol}")
                    break
                
                # Risk check
                position_size = self.engine.calculate_position_size(symbol, signal.get("entry", 0))
                risk_amount = position_size * self.risk_per_trade
                if risk_amount > self.engine.max_position_size:
                    logger.warning(f"Risk too high for {symbol}: {risk_amount} > {self.engine.max_position_size}")
                    continue
                
                # Convert signal to dict if needed
                signal_dict = signal if isinstance(signal, dict) else signal.to_dict()
                
                success = False
                if trading_mode == "virtual":
                    success = self.engine.execute_virtual_trade(signal_dict)
                else:
                    success = await self.engine.execute_real_trade(signal_dict)
                    if success:
                        # Wait briefly for Bybit to process
                        await asyncio.sleep(2)
                        # Sync real trades to DB after execution
                        self.engine.get_open_real_trades()
                        logger.info(f"Synced real trades to DB after executing trade for {symbol}")
                
                if success:
                    self.stats["trades_executed"] += 1
                    current_positions += 1
                    logger.info(f"Trade executed for {symbol} in {trading_mode} mode")
            
        except Exception as e:
            logger.error(f"Error in scan and trade: {e}", exc_info=True)

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

    async def _check_trade_exit(self, trade: Trade, trading_mode: str):
        """Check if trade should be closed"""
        try:
            from typing import List, Dict, Any
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
            
            # For real trades, check if Bybit closed the position (SL/TP triggered)
            if trading_mode == "real":
                try:
                    positions: List[Dict[str, Any]] = self.client.get_positions(symbol=symbol)
                    position = next((p for p in positions if p["size"] > 0 and p["side"].upper() == side), None)
                    if not position:
                        # Position closed by Bybit (e.g., SL/TP triggered)
                        should_exit = True
                        exit_reason = "Bybit Closed (SL/TP)"
                        current_price = self.client.get_current_price(symbol)  # Use latest price for close
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
            
            # Time-based exit (24 hours) for both modes
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
                
    async def _close_trade(self, trade: Trade, exit_price: float, reason: str, trading_mode: str):
        """Close a trade"""
        try:
            # Calculate PnL - Mode-specific
            trade_dict = trade.to_dict()
            if trading_mode == "virtual":
                pnl = self.engine.calculate_virtual_pnl(trade_dict)
            else:
                # For real, fetch actual PnL from Bybit position
                if not self.client.is_connected():
                    logger.error("Bybit client not connected for real trade close")
                    return
                positions = self.client.get_positions(symbol=trade.symbol)
                pnl = positions[0].get("unrealisedPnl", 0.0) if positions else 0.0

            # For real mode, place closing order (reverse side market order)
            if trading_mode == "real":
                close_side = "Sell" if trade.side.upper() in ["BUY", "LONG"] else "Buy"
                close_result = await self.client.place_order(
                    symbol=trade.symbol,
                    side=close_side,
                    qty=trade.qty,
                    stop_loss=0.0,  # Fixed: Use float default
                    take_profit=0.0,  # Fixed: Use float default
                    leverage=trade.leverage
                )
                if "error" in close_result:
                    logger.error(f"Failed to close real position {trade.order_id}: {close_result['error']}")
                    return
                logger.info(f"Closed real position {trade.order_id}")
                # Sync after close
                await asyncio.sleep(2)
                self.engine.get_open_real_trades()
                logger.info(f"Synced real trades to DB after closing {trade.order_id}")

            # Update trade in database
            from db import TradeModel
            from sqlalchemy import update
            
            if not self.db.session:
                logger.error("Database session not initialized")
                return
            
            success = False
            try:
                self.db.session.execute(
                    update(TradeModel)
                    .where(TradeModel.order_id == trade.order_id)
                    .values(
                        status="closed",
                        exit_price=exit_price,
                        pnl=pnl,
                        closed_at=datetime.now(timezone.utc)
                    )
                )
                self.db.session.commit()
                success = True
            except Exception as e:
                self.db.session.rollback()
                logger.error(f"Database error updating trade {trade.order_id}: {e}", exc_info=True)
            
            if success:
                # Update statistics
                self.stats["total_pnl"] += pnl
                if pnl > 0:
                    self.stats["profitable_trades"] += 1
                
                # Update balance - Mode-specific
                if trading_mode == "virtual":
                    self.engine.update_virtual_balances(pnl, mode=trading_mode)
                else:
                    self.engine.sync_real_balance()
                
                logger.info(f"Trade closed: {trade.symbol} {reason} PnL: ${pnl:.2f}, Mode: {trading_mode}")
            else:
                logger.error(f"Failed to close trade {trade.order_id}")
                
        except Exception as e:
            logger.error(f"Error closing trade {trade.order_id}: {e}", exc_info=True)

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