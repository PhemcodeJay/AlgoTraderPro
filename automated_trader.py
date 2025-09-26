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
        
        # Ensure BybitClient uses mainnet
        self.client.base_url = "https://api.bybit.com"
        
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
        
        logger.info("Automated trader initialized for Bybit mainnet with UNIFIED account and ISOLATED margin futures", extra={
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
            
            logger.info("Automated trading started on mainnet with ISOLATED margin futures")
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
                "leverage": self.leverage,
                "base_url": self.client.base_url,
                "margin_mode": "ISOLATED",
                "account_type": "UNIFIED"
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
            
            # Get symbols to scan (use valid futures symbols from engine)
            symbols = self.engine.get_usdt_symbols()
            
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
        """Execute a trading signal."""
        try:
            symbol = signal.get("symbol")
            if not symbol:
                logger.warning("Signal missing symbol")
                return

            # Validate symbol exists and is tradable (USDT perpetual futures)
            ticker_info = self.client._make_request("GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol})
            if not ticker_info or "result" not in ticker_info or not ticker_info["result"].get("list"):
                logger.warning(f"Symbol {symbol} not found or not tradable in futures")
                return

            side = signal.get("side", "Buy")
            score = signal.get("score", 0)
            entry_price = signal.get("entry") or self.client.get_current_price(symbol)

            if entry_price <= 0:
                logger.warning(f"Invalid entry price for {symbol}: {entry_price}")
                return

            # Stop loss / take profit defaults (TP 25%, SL 5% for 5:1 reward:risk ratio)
            sl_multiplier = 0.95 if side.lower() == "buy" else 1.05
            tp_multiplier = 1.25 if side.lower() == "buy" else 0.75
            stop_loss = float(signal.get("sl", entry_price * sl_multiplier))
            take_profit = float(signal.get("tp", entry_price * tp_multiplier))

            # Get available balance for UNIFIED account
            if trading_mode == "real":
                available_balance = 0.0
                try:
                    # Fetch unified wallet balance (covers futures)
                    result = self.client._make_request(
                        "GET",
                        "/v5/account/wallet-balance",
                        {"accountType": "UNIFIED"}
                    )

                    if result and "result" in result and result["result"].get("list"):
                        account_info = result["result"]["list"][0]
                        available_balance = float(account_info.get("totalAvailableBalance", 0.0))
                        logger.info(f"Fetched UNIFIED balance for {symbol}: Available={available_balance:.2f}")
                    else:
                        logger.warning(f"No UNIFIED wallet data returned")

                except Exception as e:
                    logger.error(f"Failed to fetch UNIFIED wallet balance for {symbol}: {e}")
                    available_balance = 0.0

            else:
                # Ensure session is valid for virtual mode
                if not self.db.session:
                    logger.warning("Database session is None, reinitializing")
                    self.db._get_session()
                
                wallet_balance = self.db.get_wallet_balance("virtual")
                available_balance = getattr(wallet_balance, "available", 0.0) if wallet_balance else 0.0

            if available_balance <= 0:
                logger.warning(f"No available balance for {symbol}: Available={available_balance:.2f}")
                return

            # Calculate position size based on available balance
            position_size = self.engine.calculate_position_size(
                symbol,
                entry_price,
                available_balance=available_balance
            )

            if position_size <= 0:
                logger.info(f"Skipped trade for {symbol}: insufficient balance to meet minimum order size")
                return

            # Initialize order ID
            order_id = f"auto_{symbol}_{int(time.time())}"

            if trading_mode == "real":
                # Ensure required margin <= available
                required_margin = (position_size * entry_price) / self.leverage
                if required_margin > available_balance:
                    logger.warning(f"Insufficient balance for {symbol}: Required={required_margin:.2f}, Available={available_balance:.2f}")
                    return

                # Validate against symbol rules
                symbol_info = self.engine.get_symbol_info(symbol)
                if symbol_info:
                    lot_size_filter = symbol_info.get("lotSizeFilter", {})
                    min_qty = float(lot_size_filter.get("minOrderQty", 0))
                    qty_step = float(lot_size_filter.get("qtyStep", 0))

                    if position_size < min_qty:
                        logger.info(f"Skipped trade for {symbol}: position size {position_size} < min {min_qty}")
                        return

                    if qty_step > 0:
                        position_size = (position_size // qty_step) * qty_step

                # Set isolated margin mode for futures
                try:
                    switch_result = self.client._make_request(
                        "POST",
                        "/v5/position/switch-isolated",
                        {
                            "category": "linear",
                            "symbol": symbol,
                            "tradeMode": 1,  # 1 = Isolated Margin
                            "buyLeverage": str(self.leverage),
                            "sellLeverage": str(self.leverage)
                        }
                    )
                    if switch_result.get("retCode") != 0:
                        logger.error(f"Failed to set isolated margin for {symbol}: {switch_result.get('retMsg')}")
                        return
                    logger.info(f"Set isolated margin mode for {symbol} futures")
                except Exception as e:
                    logger.error(f"Error setting isolated margin for {symbol}: {e}")
                    return

                # Place the order (futures with isolated margin)
                order_result = await self.client.place_order(
                    symbol=symbol,
                    side=side,
                    qty=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=self.leverage,
                    mode="ISOLATED"
                )

                if "error" in order_result:
                    logger.error(f"Failed to place order for {symbol}: {order_result['error']}")
                    return

                if "order_id" not in order_result:
                    logger.error(f"No order_id returned for {symbol}: {order_result}")
                    return

                order_id = order_result["order_id"]
                
                # Wait for order to fill and confirm position
                await asyncio.sleep(2)  # Give time for execution
                
                positions = self.client.get_positions(symbol)
                if not positions or positions[0]["size"] == 0:
                    logger.error(f"Failed to open position for {symbol} after order {order_id}")
                    return
                
                pos = positions[0]
                if pos["side"].upper() != side.upper():
                    logger.error(f"Position side mismatch for {symbol}: expected {side}, got {pos['side']}")
                    return
                
                position_size = pos["size"]  # Actual filled qty
                entry_price = pos["entry_price"]  # Actual avg entry price
                self.leverage = pos["leverage"]  # Update if needed

            # Save trade
            trade = Trade(
                symbol=symbol,
                side=side,
                qty=position_size,
                entry_price=entry_price,
                order_id=order_id,
                virtual=(trading_mode == "virtual"),
                status="open",
                score=score,
                strategy="Automated",
                leverage=self.leverage,
                timestamp=datetime.now(timezone.utc)
            )

            trade_dict = asdict(trade)
            if not self.db.session:
                logger.warning("Database session is None, reinitializing")
                self.db._get_session()
            
            if self.db.add_trade(trade_dict):
                self.stats["trades_executed"] += 1
                logger.info(f"Automated trade executed: {symbol} {side} @ {entry_price}, Mode: {trading_mode}, Futures ISOLATED",
                            extra={"order_id": trade_dict["order_id"], "qty": position_size})

                # Save signal to DB
                signal_obj = Signal(
                    symbol=symbol,
                    interval=signal.get("interval", "60"),
                    signal_type=signal.get("signal_type", "Auto"),
                    score=score,
                    indicators=signal.get("indicators", {}),
                    strategy=signal.get("strategy", "Auto"),
                    side=side,
                    sl=stop_loss,
                    tp=take_profit,
                    trail=signal.get("trail"),
                    liquidation=signal.get("liquidation"),
                    leverage=signal.get("leverage", self.leverage),
                    margin_usdt=signal.get("margin_usdt"),
                    entry=entry_price,
                    market=signal.get("market", "futures"),
                    created_at=signal.get("created_at", datetime.now(timezone.utc))
                )
                self.db.add_signal(signal_obj)
            else:
                logger.error(f"Failed to save trade for {symbol} in database")
                if self.db.session:
                    self.db.session.rollback()

        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}", exc_info=True)
            if self.db.session:
                self.db.session.rollback()

    def _get_open_trades(self, trading_mode: Optional[str] = None) -> List[Trade]:
        """Get open trades, optionally filtered by mode"""
        if not self.db.session:
            logger.warning("Database session is None, reinitializing")
            self.db._get_session()
        
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
            
            if trading_mode == "real":
                # Ensure database session is initialized
                session = self.db.session
                if session is None:
                    logger.warning("Database session is None, reinitializing")
                    self.db._get_session()
                    session = self.db.session
                if session is None:
                    raise ValueError("Failed to initialize database session")
                
                # Sync closed positions for real mode
                bybit_positions = self.client.get_positions()
                pos_symbols = {p["symbol"] for p in bybit_positions if p["size"] > 0}
                
                for trade in open_trades:
                    if trade.symbol not in pos_symbols:
                        # Position closed externally (e.g., by TP/SL)
                        try:
                            # Fetch recent closed PnL records starting from trade timestamp
                            start_time_ms = int(trade.timestamp.timestamp() * 1000)
                            pnl_params = {
                                "category": "linear",
                                "symbol": trade.symbol,
                                "startTime": start_time_ms,
                                "limit": 1
                            }
                            closed_pnl = self.client._make_request("GET", "/v5/position/closed-pnl", pnl_params)
                            
                            if closed_pnl and "list" in closed_pnl and closed_pnl["list"]:
                                pnl_info = closed_pnl["list"][0]
                                exit_price = float(pnl_info.get("avgExitPrice", 0))
                                pnl = float(pnl_info.get("closedPnl", 0))
                                closed_at = datetime.fromtimestamp(int(pnl_info["updatedTime"]) / 1000, timezone.utc)
                                
                                # Update trade in DB
                                session.execute(
                                    update(TradeModel)
                                    .where(TradeModel.order_id == trade.order_id)
                                    .values(
                                        status="closed",
                                        exit_price=exit_price,
                                        pnl=pnl,
                                        closed_at=closed_at
                                    )
                                )
                                session.commit()
                                
                                logger.info(f"Synced externally closed trade {trade.order_id}: PnL {pnl:.2f}")
                                
                                # Sync balance
                                self.engine.sync_real_balance()
                            else:
                                logger.warning(f"No closed PnL found for {trade.symbol} since {trade.timestamp}")
                                
                        except Exception as e:
                            logger.error(f"Error syncing closed trade {trade.order_id}: {e}", exc_info=True)
                            session.rollback()
            
            # Monitor open trades
            for trade in open_trades:
                try:
                    await self._check_trade_exit(trade, trading_mode)
                except Exception as e:
                    logger.error(f"Error monitoring trade {trade.order_id}: {e}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}", exc_info=True)
            if self.db.session:
                self.db.session.rollback()

    async def _check_trade_exit(self, trade: Trade, trading_mode: str):
        """Check if trade should be closed"""
        if trading_mode == "real":
            # For real trades, skip manual exit checks as TP/SL are handled by Bybit
            return
        
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
            
            # Exit conditions (TP 25%, SL 5%)
            should_exit = False
            exit_reason = ""
            
            # Stop loss (5% loss)
            if pnl_pct <= -5:
                should_exit = True
                exit_reason = "Stop Loss"
            
            # Take profit (25% gain)
            elif pnl_pct >= 25:
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
            session = self.db.session
            if session is None:
                logger.warning("Database session is None, reinitializing")
                self.db._get_session()
                session = self.db.session
            if session is None:
                raise ValueError("Failed to initialize database session")
            
            trade_dict = asdict(trade)
            pnl = self.engine.calculate_virtual_pnl(trade_dict)
            
            if trading_mode == "real":
                # First, check and cancel open order if exists
                open_orders = self.client.get_open_orders(trade.symbol)
                order_exists = any(o["order_id"] == trade.order_id for o in open_orders)
                if order_exists:
                    success = await self.client.cancel_order(trade.symbol, trade.order_id)
                    if not success:
                        logger.warning(f"Failed to cancel order {trade.order_id} for {trade.symbol}")
                
                # Then, check and close position if exists
                positions = self.client.get_positions(trade.symbol)
                if positions:
                    pos = positions[0]
                    close_side = "Sell" if pos["side"] == "Buy" else "Buy"
                    # Place a market order to close the position
                    close_result = await self.client.place_order(
                        symbol=trade.symbol,
                        side=close_side,
                        qty=pos["size"],
                        stop_loss=0,  # No SL for closing order
                        take_profit=0,  # No TP for closing order
                        leverage=pos["leverage"],
                        mode="ISOLATED"
                    )
                    if "order_id" not in close_result:
                        logger.error(f"Failed to close position for {trade.symbol}: {close_result}")
                        return
                    # Wait briefly to ensure order is processed
                    await asyncio.sleep(1)
                    # Recalculate pnl after close (approximate)
                    pnl = self.engine.calculate_virtual_pnl(trade_dict)
            
            # Update trade in database
            success = False
            try:
                session.execute(
                    update(TradeModel)
                    .where(TradeModel.order_id == trade.order_id)
                    .values(
                        status="closed",
                        exit_price=exit_price,
                        pnl=pnl,
                        closed_at=datetime.now(timezone.utc)
                    )
                )
                session.commit()
                success = True
            except Exception as e:
                logger.error(f"Database error updating trade {trade.order_id}: {e}", exc_info=True)
                session.rollback()
            
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
            if self.db.session:
                self.db.session.rollback()
            
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