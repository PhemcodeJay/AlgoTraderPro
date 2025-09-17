import asyncio
from dataclasses import asdict
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy import update
from engine import TradingEngine
from bybit_client import BybitClient
from signal_generator import generate_signals, get_usdt_symbols
from db import db_manager, Trade, Signal, TradeModel
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
        self.scan_interval, self.top_n_signals = self.engine.get_settings()
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
                "current_positions": len(self._get_open_trades()),
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
                    trading_mode = self.db.get_setting("trading_mode") or "virtual"
                    if trading_mode == "real":
                        self.engine.sync_real_balance()
                    
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
            logger.error(f"Trading loop error: {e}", exc_info=True)
            if status_container:
                status_container.error(f"Trading loop failed: {e}")
        finally:
            self.is_running = False
            if status_container:
                status_container.info("🤖 Automated trading stopped")

    async def _scan_and_trade(self, trading_mode: str):
        """Scan markets and execute trades"""
        try:
            # Check position limits
            current_positions = len(self._get_open_trades())
            if current_positions >= self.max_positions:
                logger.info(f"Max positions reached: {current_positions}/{self.max_positions}")
                return
            
            # Get symbols to scan
            symbols = get_usdt_symbols()
            
            # Generate signals
            signals = generate_signals(
                symbols,
                interval="60",
                top_n=self.top_n_signals,
                trading_mode=trading_mode
            )
            self.stats["signals_generated"] += len(signals)
            logger.info(f"Generated {len(signals)} signals, processing up to {self.top_n_signals}")
            
            if not signals:
                logger.info("No signals generated")
                return
            
            # Execute top signals
            available_slots = self.max_positions - current_positions
            for signal in signals[:min(self.top_n_signals, available_slots)]:
                try:
                    await self._execute_signal(signal, trading_mode)
                except Exception as e:
                    logger.error(f"Error executing signal for {signal.get('symbol')}: {e}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Error in scan and trade: {e}", exc_info=True)

    async def _execute_signal(self, signal: Dict[str, Any], trading_mode: str):
        """Execute a signal"""
        try:
            symbol = signal.get("symbol")
            if not symbol:
                logger.warning("Signal missing symbol")
                return
            
            side = signal.get("side", "Buy")
            score = signal.get("score", 0)
            entry_price = signal.get("entry")
            if entry_price is None:
                entry_price = self.client.get_current_price(symbol)
            
            if entry_price <= 0:
                logger.warning(f"Invalid entry price for {symbol}: {entry_price}")
                return
            
            position_size = self.engine.calculate_position_size(symbol, entry_price)
            
            if position_size <= 0:
                logger.warning(f"Invalid position size for {symbol}: {position_size}")
                return
            
            # Create trade data
            trade = Trade(
                symbol=symbol,
                side=side,
                qty=position_size,
                entry_price=entry_price,
                order_id=f"auto_{symbol}_{int(time.time())}",
                virtual=(trading_mode == "virtual"),
                status="open",
                score=score,
                strategy="Automated",
                leverage=self.leverage,
                timestamp=datetime.now(timezone.utc)
            )
            
            trade_dict = asdict(trade)
            
            # Execute trade
            success = self.db.add_trade(trade_dict)
            if success:
                self.stats["trades_executed"] += 1
                logger.info(f"Automated trade executed: {symbol} {side} @ {entry_price}, Mode: {trading_mode}")
                
                # Save signal to DB
                signal_obj = Signal(
                    symbol=symbol,
                    interval=signal.get("interval", "60"),
                    signal_type=signal.get("signal_type", "Auto"),
                    score=score,
                    indicators=signal.get("indicators", {}),
                    strategy=signal.get("strategy", "Auto"),
                    side=side,
                    sl=signal.get("sl"),
                    tp=signal.get("tp"),
                    trail=signal.get("trail"),
                    liquidation=signal.get("liquidation"),
                    leverage=signal.get("leverage", self.leverage),
                    margin_usdt=signal.get("margin_usdt"),
                    entry=entry_price,
                    market=signal.get("market"),
                    created_at=signal.get("created_at", datetime.now(timezone.utc))
                )
                
                self.db.add_signal(signal_obj)
            else:
                logger.error(f"Failed to execute trade for {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}", exc_info=True)

    def _get_open_trades(self, trading_mode: Optional[str] = None) -> List[Trade]:
        """Get open trades, optionally filtered by mode"""
        trades = self.db.get_trades(limit=100)  # Adjust limit as needed
        open_trades = [t for t in trades if t.status == "open"]

        if trading_mode == "virtual":
            return [t for t in open_trades if t.virtual]
        elif trading_mode == "real":
            return [t for t in open_trades if not t.virtual]
        
        return open_trades

    async def _monitor_positions(self, trading_mode: str):
        """Monitor and manage existing positions"""
        try:
            open_trades = self._get_open_trades(trading_mode)
            
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
            symbol = trade.symbol
            current_price = self.client.get_current_price(symbol)
            
            if current_price <= 0:
                logger.warning(f"Invalid current price for {symbol}: {current_price}")
                return
            
            entry_price = trade.entry_price
            side = trade.side.upper()
            
            # Calculate current PnL
            if side in ["BUY", "LONG"]:
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            # Stop loss (5% loss)
            if pnl_pct <= -5:
                should_exit = True
                exit_reason = "Stop Loss"
            
            # Take profit (10% gain)
            elif pnl_pct >= 10:
                should_exit = True
                exit_reason = "Take Profit"
            
            # Time-based exit (24 hours)
            elif trade.timestamp:
                hours_open = (datetime.now(timezone.utc) - trade.timestamp).total_seconds() / 3600
                if hours_open >= 24:
                    should_exit = True
                    exit_reason = "Time Limit"
            
            if should_exit:
                await self._close_trade(trade, current_price, exit_reason, trading_mode)
                
        except Exception as e:
            logger.error(f"Error checking trade exit for {trade.symbol}: {e}", exc_info=True)

    async def _close_trade(self, trade: Trade, exit_price: float, reason: str, trading_mode: str):
        """Close a trade"""
        try:
            if not self.db.session:
                logger.error("Database session not initialized")
                return
            
            trade_dict = asdict(trade)
            pnl = self.engine.calculate_virtual_pnl(trade_dict)
            
            # Update trade in database
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
                
                # Update balance
                self.engine.update_virtual_balances(pnl, mode=trading_mode)
                
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