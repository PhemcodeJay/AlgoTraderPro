import json
from datetime import datetime, timezone, timedelta
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from bybit_client import BybitClient
from db import DatabaseManager, Trade, TradeModel, WalletBalance
from settings import load_settings
from logging_config import get_trading_logger
from exceptions import TradingException, create_error_context
from utils import get_symbol_precision, round_to_precision

logger = get_trading_logger('engine')

class TradingEngine:
    def __init__(self):
        try:
            self.client = BybitClient()
            self.settings = load_settings()
            self.db = DatabaseManager()
            self._candle_cache = {}
            
            # Position safety limits
            self.max_position_size = self.settings.get("MAX_POSITION_SIZE", 10000.0)  # USDT
            self.max_open_positions = self.settings.get("MAX_OPEN_POSITIONS", 10)
            self.max_daily_loss = self.settings.get("MAX_DAILY_LOSS", 100.0)  # USDT
            self.max_risk_per_trade = self.settings.get("MAX_RISK_PER_TRADE", 0.05)  # 5%
            
            # Trading state management
            self._trading_enabled = True
            self._emergency_stop = False
            self._last_health_check = None
            self._consecutive_failures = 0
            self._daily_pnl = 0.0
            self._daily_reset_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Performance tracking
            self.trade_count = 0
            self.successful_trades = 0
            self.failed_trades = 0
            
            logger.info(
                "Trading engine initialized successfully",
                extra={
                    'max_position_size': self.max_position_size,
                    'max_open_positions': self.max_open_positions,
                    'max_daily_loss': self.max_daily_loss
                }
            )
            
        except Exception as e:
            error_context = create_error_context(
                module=__name__,
                function='__init__',
                extra_data={'settings': self.settings if hasattr(self, 'settings') else None}
            )
            logger.error(f"Failed to initialize trading engine: {str(e)}")
            raise TradingException(
                f"Trading engine initialization failed: {str(e)}",
                context=error_context,
                original_exception=e
            )
    
    def _reset_daily_stats(self):
        """Reset daily statistics if new day"""
        try:
            current_time = datetime.now(timezone.utc)
            if current_time >= self._daily_reset_time + timedelta(days=1):
                self._daily_pnl = 0.0
                self._daily_reset_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info("Daily trading statistics reset")
        except Exception as e:
            logger.warning(f"Failed to reset daily stats: {str(e)}")
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency stop conditions"""
        try:
            self._reset_daily_stats()
            
            # Check daily loss limit
            if self._daily_pnl <= -self.max_daily_loss:
                logger.critical(
                    f"Daily loss limit exceeded: {self._daily_pnl} <= -{self.max_daily_loss}",
                    extra={'daily_pnl': self._daily_pnl, 'limit': self.max_daily_loss}
                )
                self._emergency_stop = True
                return False
            
            # Check consecutive failures
            if self._consecutive_failures >= 10:
                logger.critical(
                    f"Too many consecutive failures: {self._consecutive_failures}",
                    extra={'consecutive_failures': self._consecutive_failures}
                )
                self._emergency_stop = True
                return False
            
            # Check API health
            api_health = self.client.get_connection_health()
            if api_health['status'] not in ['healthy', 'degraded']:
                logger.warning(
                    f"API health check failed: {api_health['status']}",
                    extra={'api_health': api_health}
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {str(e)}")
            return False
    
    def is_trading_enabled(self) -> bool:
        """Check if trading is currently enabled"""
        return self._trading_enabled and not self._emergency_stop and self._check_emergency_conditions()
    
    def enable_trading(self) -> bool:
        """Enable trading with safety checks"""
        try:
            if self._emergency_stop:
                logger.warning("Cannot enable trading: Emergency stop is active")
                return False
            
            if not self._check_emergency_conditions():
                logger.warning("Cannot enable trading: Emergency conditions detected")
                return False
            
            self._trading_enabled = True
            logger.info("Trading enabled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable trading: {str(e)}")
            return False
    
    def disable_trading(self, reason: str = "Manual disable") -> bool:
        """Disable trading"""
        try:
            self._trading_enabled = False
            logger.warning(f"Trading disabled: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to disable trading: {str(e)}")
            return False
    
    def emergency_stop(self, reason: str = "Emergency stop triggered") -> bool:
        """Trigger emergency stop"""
        try:
            self._emergency_stop = True
            self._trading_enabled = False
            
            logger.critical(
                f"EMERGENCY STOP ACTIVATED: {reason}",
                extra={'reason': reason, 'timestamp': datetime.now(timezone.utc).isoformat()}
            )
            
            # Close all open positions
            open_trades = self.get_open_real_trades()
            for trade in open_trades:
                close_side = "Sell" if trade.side.upper() in ["BUY", "LONG"] else "Buy"
                try:
                    close_result = asyncio.run(self.client.place_order(
                        symbol=trade.symbol,
                        side=close_side,
                        qty=trade.qty,
                        leverage=trade.leverage,
                        mode="CROSS"
                    ))
                    if "error" not in close_result:
                        logger.info(f"Closed real position {trade.order_id} during emergency stop")
                except Exception as e:
                    logger.error(f"Failed to close position {trade.order_id}: {e}")
            
            return True
            
        except Exception as e:
            logger.critical(f"Failed to activate emergency stop: {str(e)}")
            return False

    def get_settings(self) -> Tuple[int, int]:
        """Get current scan interval and top N signals"""
        return self.settings.get("SCAN_INTERVAL", 3600), self.settings.get("TOP_N_SIGNALS", 5)

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """Update trading settings"""
        try:
            self.settings.update(new_settings)
            with open("settings.json", "w") as f:
                json.dump(self.settings, f, indent=2)
            logger.info("Settings updated")
            return True
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False

    def get_cached_candles(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """Get cached candles or fetch new ones"""
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            now = datetime.now(timezone.utc)
            
            if cache_key in self._candle_cache:
                cached_time, cached_data = self._candle_cache[cache_key]
                if (now - cached_time).total_seconds() < 300:
                    return cached_data

            candles = self.client.get_klines(symbol, interval, limit)
            if candles:
                self._candle_cache[cache_key] = (now, candles)
                return candles
            return []
        except Exception as e:
            logger.error(f"Error getting candles for {symbol}: {e}")
            return []

    def get_usdt_symbols(self) -> List[str]:
        """Get list of USDT trading pairs"""
        return self.settings.get("SYMBOLS", [
            "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", 
            "XRPUSDT", "BNBUSDT", "AVAXUSDT"
        ])

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information (e.g., lot size, tick size)"""
        try:
            result = self.client._make_request("GET", "/v5/market/instruments-info", {
                "category": "linear",
                "symbol": symbol
            })
            if result and "list" in result and result["list"]:
                return result["list"][0]
            return {}
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {}
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        risk_percent: Optional[float] = None,
        leverage: Optional[int] = None,
        available_balance: Optional[float] = None
    ) -> float:
        """Calculate position size based on risk, available balance, leverage, and symbol rules."""
        try:
            import math

            risk_pct = risk_percent or self.settings.get("RISK_PCT", 0.01)
            lev = leverage or self.settings.get("LEVERAGE", 15)

            mode = "real" if self.db.get_setting("trading_mode") == "real" else "virtual"
            wallet_balance = self.db.get_wallet_balance(mode)
            if not wallet_balance:
                self.db.migrate_capital_json_to_db()
                wallet_balance = self.db.get_wallet_balance(mode)

            balance = available_balance if available_balance is not None else (wallet_balance.available if wallet_balance else 100.0)
            if balance <= 0:
                logger.warning(f"Cannot calculate position size for {symbol}: Available balance is {balance}")
                return 0.0

            risk_amount = max(balance * risk_pct, 2.0)
            position_value = risk_amount * lev
            position_size = position_value / entry_price

            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"No symbol info for {symbol}")
                return 0.0

            lot_size_filter = symbol_info.get("lotSizeFilter", {})
            min_qty = float(lot_size_filter.get("minOrderQty", 0))
            qty_step = float(lot_size_filter.get("qtyStep", 0))

            if min_qty > 0 and position_size < min_qty:
                min_position_value = min_qty * entry_price
                min_margin_required = min_position_value / lev
                if min_margin_required > balance:
                    fraction = balance * lev / entry_price
                    if fraction >= qty_step:
                        position_size = max(fraction, qty_step)
                        logger.info(f"Adjusted tiny balance to position size {position_size} for {symbol}")
                    else:
                        logger.warning(
                            f"Skipping {symbol}: required margin {min_margin_required:.2f}, available {balance:.2f}"
                        )
                        return 0.0
                else:
                    position_size = min_qty
                    logger.info(f"Adjusted position size up to minimum {min_qty} for {symbol}")

            if qty_step > 0:
                position_size = round(position_size / qty_step) * qty_step

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}", exc_info=True)
            return 0.0

    def calculate_virtual_pnl(self, trade: Dict) -> float:
        """Calculate profit/loss for a virtual trade"""
        try:
            symbol = trade.get("symbol")
            if not symbol or not isinstance(symbol, str):
                logger.error("Invalid or missing symbol in trade data")
                return 0.0
            entry_price = float(trade.get("entry_price", 0))
            current_price = self.client.get_current_price(symbol)
            qty = float(trade.get("qty", 0))
            side = trade.get("side", "BUY").upper()

            if current_price <= 0 or entry_price <= 0 or qty <= 0:
                logger.warning(f"Invalid trade data for PNL calculation: {trade}")
                return 0.0

            if side in ["BUY", "LONG"]:
                return (current_price - entry_price) * qty
            else:
                return (entry_price - current_price) * qty

        except Exception as e:
            logger.error(f"Error calculating virtual PNL: {e}", exc_info=True)
            return 0.0

    def update_virtual_balances(self, pnl: float, mode: str = "virtual") -> bool:
        """Update virtual wallet balance"""
        try:
            wallet_balance = self.db.get_wallet_balance(mode)
            if not wallet_balance:
                self.db.migrate_capital_json_to_db()
                wallet_balance = self.db.get_wallet_balance(mode)

            if not wallet_balance:
                logger.error(f"No wallet balance found for {mode} mode")
                return False

            new_available = wallet_balance.available + pnl
            new_capital = wallet_balance.capital + pnl
            new_used = max(wallet_balance.used - pnl, 0.0)

            updated_balance = WalletBalance(
                trading_mode=mode,
                capital=new_capital,
                available=new_available,
                used=new_used,
                start_balance=wallet_balance.start_balance,
                currency=wallet_balance.currency,
                updated_at=datetime.now(timezone.utc),
                id=wallet_balance.id
            )

            success = self.db.update_wallet_balance(updated_balance)
            if success:
                logger.info(f"Updated {mode} balance: Capital=${new_capital:.2f}, Available=${new_available:.2f}, Used=${new_used:.2f}")
            return success

        except Exception as e:
            logger.error(f"Error updating virtual balance: {e}", exc_info=True)
            return False

    def get_open_real_trades(self) -> List[Trade]:
        """Fetch and sync open real trades from Bybit to database"""
        try:
            if not self.client.is_connected():
                logger.error("Bybit client not connected")
                return []

            positions = self.client._make_request("GET", "/v5/position/list", {
                "category": "linear",
                "settleCoin": "USDT"
            })

            if not positions or "list" not in positions:
                logger.warning("No positions found in Bybit response")
                return []

            open_trades = []
            for pos in positions["list"]:
                if float(pos.get("size", 0)) <= 0:
                    continue

                symbol = pos.get("symbol")
                side = pos.get("side").upper()
                qty = float(pos.get("size"))
                entry_price = float(pos.get("avgPrice"))
                leverage = int(float(pos.get("leverage")))
                position_id = pos.get("positionIdx")  # Use positionIdx as order_id
                sl = float(pos.get("stopLoss") or 0)
                tp = float(pos.get("takeProfit") or 0)
                created_at = datetime.fromtimestamp(float(pos.get("createdTime")) / 1000, timezone.utc)

                # Check if trade exists in DB
                existing_trade = self.db.get_trade_by_order_id(position_id)
                if existing_trade:
                    # Update existing trade
                    from sqlalchemy import update
                    if not self.db.session:
                        logger.error("Database session not initialized")
                        return []
                    try:
                        self.db.session.execute(
                            update(TradeModel)
                            .where(TradeModel.order_id == position_id)
                            .values(
                                qty=qty,
                                entry_price=entry_price,
                                leverage=leverage,
                                sl=sl,
                                tp=tp,
                                status="open",
                                updated_at=datetime.now(timezone.utc)
                            )
                        )
                        self.db.session.commit()
                        logger.info(f"Updated trade {position_id} for {symbol} in DB")
                    except Exception as e:
                        self.db.session.rollback()
                        logger.error(f"Failed to update trade {position_id}: {e}")
                else:
                    # Create new trade
                    trade_data = {
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "entry_price": entry_price,
                        "order_id": position_id,
                        "virtual": False,
                        "status": "open",
                        "score": 0.0,
                        "strategy": "Auto",
                        "leverage": leverage,
                        "sl": sl,
                        "tp": tp,
                        "trail": 0.0,
                        "liquidation": float(pos.get("liqPrice", 0)),
                        "margin_usdt": float(pos.get("positionValue", 0)) / leverage,
                        "timestamp": created_at
                    }
                    success = self.db.add_trade(trade_data)
                    if success:
                        logger.info(f"Added new real trade {position_id} for {symbol} to DB")
                    else:
                        logger.error(f"Failed to add trade {position_id} for {symbol} to DB")

                trade = Trade(**trade_data)
                open_trades.append(trade)

            # Remove stale trades from DB
            db_trades = self.db.get_open_trades(virtual=False)
            db_order_ids = {trade.order_id for trade in db_trades}
            bybit_order_ids = {pos.get("positionIdx") for pos in positions["list"] if float(pos.get("size", 0)) > 0}
            for trade in db_trades:
                if trade.order_id not in bybit_order_ids:
                    from sqlalchemy import update
                    if not self.db.session:
                        logger.error("Database session not initialized")
                        return []
                    try:
                        current_price = self.client.get_current_price(trade.symbol)
                        if current_price <= 0:
                            current_price = trade.entry_price  # Fallback to entry price
                        self.db.session.execute(
                            update(TradeModel)
                            .where(TradeModel.order_id == trade.order_id)
                            .values(
                                status="closed",
                                exit_price=float(current_price),
                                closed_at=datetime.now(timezone.utc)
                            )
                        )
                        self.db.session.commit()
                        logger.info(f"Closed stale trade {trade.order_id} in DB")
                    except Exception as e:
                        self.db.session.rollback()
                        logger.error(f"Failed to close stale trade {trade.order_id}: {e}")

            return open_trades

        except Exception as e:
            logger.error(f"Error syncing open real trades: {e}", exc_info=True)
            return []

    def sync_real_balance(self) -> bool:
        """Sync real wallet balance from Bybit to database"""
        try:
            if not self.client.is_connected():
                logger.warning("Bybit client not connected. Attempting to reconnect...")
                try:
                    self.client = BybitClient()
                    if not self.client.is_connected():
                        logger.error("Reconnection failed. Check API credentials and network.")
                        return False
                    logger.info("Bybit client reconnected successfully")
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}", exc_info=True)
                    return False

            result = self.client._make_request(
                "GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}
            )

            if not result or "list" not in result or not result["list"]:
                logger.warning("No account data in Bybit response")
                return False

            wallet = result["list"][0]
            total_equity = float(wallet.get("totalEquity") or 0.0)
            coins = wallet.get("coin", [])
            usdt_coin = next((c for c in coins if c.get("coin") == "USDT"), None)

            if usdt_coin:
                total_available = float(usdt_coin.get("walletBalance") or 0.0)
            else:
                total_available = total_equity

            used = max(total_equity - total_available, 0.0)
            if used < 1e-6:
                used = 0.0

            if total_available == 0 and total_equity > 0:
                logger.warning(
                    "Available balance is 0 while equity > 0. "
                    "Funds may be locked in margin, open positions, or collateral disabled in Bybit."
                )

            existing_balance = self.db.get_wallet_balance("real")
            start_balance = (
                existing_balance.start_balance
                if existing_balance and existing_balance.start_balance > 0
                else total_equity
            )

            wallet_balance = WalletBalance(
                trading_mode="real",
                capital=total_equity,
                available=total_available,
                used=used,
                start_balance=start_balance,
                currency="USDT",
                updated_at=datetime.now(timezone.utc),
                id=existing_balance.id if existing_balance else None,
            )

            if self.db.update_wallet_balance(wallet_balance):
                logger.info(
                    f"✅ Real balance synced with Bybit: Capital=${total_equity:.2f}, "
                    f"Available=${total_available:.2f}, Used=${used:.2f}"
                )
                return True
            else:
                logger.error("Failed to update wallet balance in database")
                return False

        except Exception as e:
            logger.error(f"❌ Error syncing real balance: {e}", exc_info=True)
            return False

    def get_open_virtual_trades(self) -> List[Trade]:
        """Fetch open virtual trades from database"""
        try:
            return self.db.get_open_trades(virtual=True)
        except Exception as e:
            logger.error(f"Error fetching open virtual trades: {e}", exc_info=True)
            return []

    def execute_virtual_trade(self, signal: Dict, trading_mode: str = "virtual") -> bool:
        """Execute a virtual trade based on a signal"""
        try:
            symbol = signal.get("symbol")
            if not symbol or not isinstance(symbol, str):
                logger.error("Symbol is required and must be a string for executing trade")
                return False
                
            side = signal.get("side", "Buy").upper()
            entry_price = signal.get("entry") or self.client.get_current_price(symbol)
            
            if entry_price <= 0:
                logger.error(f"Invalid entry price for {symbol}")
                return False
            
            position_size = self.calculate_position_size(symbol, entry_price)
            
            trade_data = {
                "symbol": symbol,
                "side": side,
                "qty": position_size,
                "entry_price": entry_price,
                "order_id": f"virtual_{symbol}_{int(datetime.now().timestamp())}",
                "virtual": trading_mode == "virtual",
                "status": "open",
                "score": signal.get("score"),
                "strategy": signal.get("strategy", "Auto"),
                "leverage": signal.get("leverage", 15),
                "sl": signal.get("sl"),
                "tp": signal.get("tp"),
                "trail": signal.get("trail"),
                "liquidation": signal.get("liquidation"),
                "margin_usdt": signal.get("margin_usdt")
            }
            
            success = self.db.add_trade(trade_data)
            if success:
                logger.info(
                    f"Virtual trade executed: {symbol} {side} @ {entry_price}, "
                    f"SL: {trade_data['sl']}, TP: {trade_data['tp']}, "
                    f"Trail: {trade_data['trail']}, Liquidation: {trade_data['liquidation']}, "
                    f"Margin: {trade_data['margin_usdt']} USDT"
                )
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error executing virtual trade: {e}", exc_info=True)
            return False
            
    async def execute_real_trade(self, signals: List[Dict], trading_mode: str = "real") -> bool:
        """Execute multiple real trades on Bybit based on a list of signals"""
        try:
            if not self.is_trading_enabled():
                logger.error("Trading is disabled or emergency stop is active")
                return False

            if not signals or not isinstance(signals, list):
                logger.error("Signals must be a non-empty list of dictionaries")
                return False

            success_count = 0
            total_trades = len(signals)
            
            open_trades = self.get_open_real_trades()
            current_positions = len(open_trades)
            total_position_value = sum(trade.entry_price * trade.qty for trade in open_trades)

            for signal in signals:
                symbol = signal.get("symbol")
                if not symbol or not isinstance(symbol, str):
                    logger.error("Symbol is required and must be a string for executing trade")
                    continue

                side = signal.get("side", "Buy").upper()
                entry_price = signal.get("entry") or self.client.get_current_price(symbol)
                if entry_price <= 0:
                    logger.error(f"Invalid entry price for {symbol}")
                    continue

                # Get symbol-specific tick size
                tick_size = get_symbol_precision(symbol)
                
                # Use signal's SL/TP if available, otherwise calculate
                sl = signal.get("sl")
                tp = signal.get("tp")
                if sl is None or tp is None:
                    sl_percent = 0.1
                    tp_percent = 0.5
                    if side == "BUY":
                        sl = round_to_precision(entry_price * (1 - sl_percent), tick_size)
                        tp = round_to_precision(entry_price * (1 + tp_percent), tick_size)
                    else:
                        sl = round_to_precision(entry_price * (1 + sl_percent), tick_size)
                        tp = round_to_precision(entry_price * (1 - tp_percent), tick_size)

                position_size = self.calculate_position_size(symbol, entry_price)
                if position_size <= 0:
                    logger.error(f"Invalid position size for {symbol}: {position_size}")
                    continue

                if current_positions >= self.max_open_positions:
                    logger.warning(f"Max open positions ({self.max_open_positions}) reached. Skipping trade for {symbol}")
                    continue

                total_position_value += entry_price * position_size
                if total_position_value > self.max_position_size:
                    logger.warning(f"Max position size ({self.max_position_size} USDT) exceeded. Skipping trade for {symbol}")
                    continue

                # Place order with SL/TP
                order_response = await self.client.place_order(
                    symbol=symbol,
                    side=side,
                    qty=position_size,
                    leverage=signal.get("leverage", 15),
                    mode="CROSS",
                    stop_loss=sl,
                    take_profit=tp
                )

                if "error" in order_response or not order_response.get("order_id"):
                    logger.error(f"Failed to place order for {symbol}: {order_response.get('error', 'Unknown error')}")
                    self._consecutive_failures += 1
                    continue

                order_id = order_response.get("order_id")

                trade_data = {
                    "symbol": symbol,
                    "side": side,
                    "qty": position_size,
                    "entry_price": entry_price,
                    "order_id": order_id,
                    "virtual": False,
                    "status": "open",
                    "score": signal.get("score"),
                    "strategy": signal.get("strategy", "Auto"),
                    "leverage": signal.get("leverage", 15),
                    "sl": sl,
                    "tp": tp,
                    "trail": signal.get("trail"),
                    "liquidation": signal.get("liquidation"),
                    "margin_usdt": signal.get("margin_usdt"),
                    "timestamp": datetime.now(timezone.utc)
                }

                success = self.db.add_trade(trade_data)
                if success:
                    logger.info(
                        f"Real trade executed: {symbol} {side} @ {entry_price:.2f}, "
                        f"Qty: {position_size:.6f}, Order ID: {order_id}, "
                        f"SL: {sl:.2f}, TP: {tp:.2f}, "
                        f"Trail: {trade_data['trail']}, Liquidation: {trade_data['liquidation']}, "
                        f"Margin: {trade_data['margin_usdt']} USDT"
                    )
                    success_count += 1
                    self._consecutive_failures = 0
                    self.trade_count += 1
                    self.successful_trades += 1
                    current_positions += 1
                else:
                    logger.error(f"Failed to save trade to database for {symbol}, Order ID: {order_id}")
                    self._consecutive_failures += 1
                    self.failed_trades += 1

                # Brief pause to avoid rate limits
                await asyncio.sleep(0.5)

            # Sync trades after execution
            self.get_open_real_trades()
            logger.info(f"Synced real trades to DB after executing {success_count}/{total_trades} trades")

            self._daily_pnl += 0.0  # Updated in position monitoring

            logger.info(f"Executed {success_count}/{total_trades} real trades successfully")
            return success_count > 0

        except Exception as e:
            logger.error(f"Error executing real trades: {e}", exc_info=True)
            self._consecutive_failures += 1
            self.failed_trades += 1
            return False

    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, "client"):
                self.client.close()
            logger.info("Trading engine closed")
        except Exception as e:
            logger.error(f"Error closing trading engine: {e}")