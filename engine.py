import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from bybit_client import BybitClient
from db import db_manager, Trade, WalletBalance
from db import WalletBalance as DBWalletBalance
from settings import load_settings
from logging_config import get_trading_logger
from exceptions import (
    TradingException, create_error_context
)

logger = get_trading_logger('engine')
    
class TradingEngine:
    def __init__(self):
        try:
            self.client = BybitClient()
            self.settings = load_settings()
            self.db = db_manager
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
            
            # TODO: Close all open positions immediately
            # This would be implemented based on exchange capabilities
            
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
            # Save to file
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
            now = datetime.now()
            
            # Check cache (5 minute expiry)
            if cache_key in self._candle_cache:
                cached_time, cached_data = self._candle_cache[cache_key]
                if (now - cached_time).total_seconds() < 300:
                    return cached_data

            # Fetch new data
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
        """Get symbol information (e.g., lot size)"""
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

            # --- Determine risk & leverage ---
            risk_pct = risk_percent or self.settings.get("RISK_PCT", 0.01)
            lev = leverage or self.settings.get("LEVERAGE", 10)

            # --- Get wallet balance ---
            mode = "real" if self.db.get_setting("trading_mode") == "real" else "virtual"
            wallet_balance = self.db.get_wallet_balance(mode)
            if not wallet_balance:
                self.db.migrate_capital_json_to_db()
                wallet_balance = self.db.get_wallet_balance(mode)

            balance = available_balance if available_balance is not None else (wallet_balance.available if wallet_balance else 100.0)
            if balance <= 0:
                logger.warning(f"Cannot calculate position size for {symbol}: Available balance is {balance}")
                return 0.0

            # --- Risk-based position sizing ---
            risk_amount = max(balance * risk_pct, 1.0)  # enforce at least $1 risk
            position_value = risk_amount * lev
            position_size = position_value / entry_price

            # --- Symbol trading rules ---
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"No symbol info for {symbol}")
                return 0.0

            lot_size_filter = symbol_info.get("lotSizeFilter", {})
            min_qty = float(lot_size_filter.get("minOrderQty", 0))
            qty_step = float(lot_size_filter.get("qtyStep", 0))

            # --- Enforce minimum quantity safely ---
            if min_qty > 0 and position_size < min_qty:
                min_position_value = min_qty * entry_price
                min_margin_required = min_position_value / lev
                if min_margin_required > balance:
                    # Try to allocate at least a fraction of min_qty if balance is very small
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

            # --- Align to quantity step ---
            if qty_step > 0:
                steps = math.floor(position_size / qty_step)
                position_size = steps * qty_step
                # Ensure still >= min_qty
                if min_qty > 0 and position_size < min_qty:
                    position_size = min_qty

            # --- Return safe rounded value ---
            return round(position_size, 6) if position_size > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}", exc_info=True)
            return 0.0


    def calculate_virtual_pnl(self, trade: Dict) -> float:
        """Calculate unrealized PnL for virtual trades"""
        try:
            current_price = self.client.get_current_price(trade["symbol"])
            entry_price = float(trade.get("entry_price", 0))
            qty = float(trade.get("qty", 0))
            side = trade.get("side", "Buy").upper()
            
            if current_price <= 0 or entry_price <= 0:
                return 0.0
            
            if side in ["BUY", "LONG"]:
                pnl = (current_price - entry_price) * qty
            else:
                pnl = (entry_price - current_price) * qty
                
            return round(pnl, 2)
        except Exception as e:
            logger.error(f"Error calculating virtual PnL: {e}")
            return 0.0

    def get_open_virtual_trades(self) -> List[Trade]:
        """Get open virtual trades"""
        try:
            all_trades = self.db.get_trades()
            return [trade for trade in all_trades if trade.virtual and trade.status == "open"]
        except Exception as e:
            logger.error(f"Error getting open virtual trades: {e}")
            return []

    def get_open_real_trades(self) -> List[Trade]:
        """Get open real trades"""
        try:
            all_trades = self.db.get_trades()
            return [trade for trade in all_trades if not trade.virtual and trade.status == "open"]
        except Exception as e:
            logger.error(f"Error getting open real trades: {e}")
            return []

    def get_closed_virtual_trades(self) -> List[Trade]:
        """Get closed virtual trades"""
        try:
            all_trades = self.db.get_trades()
            return [trade for trade in all_trades if trade.virtual and trade.status == "closed"]
        except Exception as e:
            logger.error(f"Error getting closed virtual trades: {e}")
            return []

    def get_closed_real_trades(self) -> List[Trade]:
        """Get closed real trades"""
        try:
            all_trades = self.db.get_trades()
            return [trade for trade in all_trades if not trade.virtual and trade.status == "closed"]
        except Exception as e:
            logger.error(f"Error getting closed real trades: {e}")
            return []

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive trading statistics"""
        try:
            all_trades = self.db.get_trades()
            virtual_trades = [t for t in all_trades if t.virtual]
            real_trades = [t for t in all_trades if not t.virtual]
            
            def calc_stats(trades):
                if not trades:
                    return {
                        "total_trades": 0,
                        "win_rate": 0,
                        "total_pnl": 0,
                        "avg_pnl": 0,
                        "profitable_trades": 0
                    }
                
                pnls = [t.pnl or 0 for t in trades]
                profitable = len([p for p in pnls if p > 0])
                
                return {
                    "total_trades": len(trades),
                    "win_rate": (profitable / len(trades)) * 100,
                    "total_pnl": sum(pnls),
                    "avg_pnl": np.mean(pnls),
                    "profitable_trades": profitable
                }
            
            virtual_stats = calc_stats(virtual_trades)
            real_stats = calc_stats(real_trades)
            overall_stats = calc_stats(all_trades)
            
            return {
                **overall_stats,
                "virtual_total_trades": virtual_stats["total_trades"],
                "virtual_win_rate": virtual_stats["win_rate"],
                "virtual_total_pnl": virtual_stats["total_pnl"],
                "real_total_trades": real_stats["total_trades"],
                "real_win_rate": real_stats["win_rate"],
                "real_total_pnl": real_stats["total_pnl"]
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            return {}

    def update_virtual_balances(self, pnl: float, mode: str = "virtual"):
        """Update virtual balance after a trade."""
        try:
            # Fetch current balance from database
            wallet_balance = self.db.get_wallet_balance(mode)

            # Create default balance if not exists
            if not wallet_balance:
                default_balance = WalletBalance(
                    trading_mode=mode,
                    capital=1000.0 if mode == "virtual" else 0.0,
                    available=1000.0 if mode == "virtual" else 0.0,
                    used=0.0,
                    start_balance=1000.0 if mode == "virtual" else 0.0,
                    currency="USDT",
                    updated_at=datetime.utcnow(),
                )
                self.db.update_wallet_balance(default_balance)
                wallet_balance = self.db.get_wallet_balance(mode)

            # Null safety check
            if not wallet_balance:
                logger.error(f"Failed to get or create wallet balance for mode: {mode}")
                return

            # Compute new balance values
            new_available = max(0.0, wallet_balance.available + pnl)
            new_capital = wallet_balance.capital + pnl
            new_used = max(0.0, new_capital - new_available)

            # Build updated WalletBalance dataclass
            updated_balance = WalletBalance(
                trading_mode=mode,
                capital=new_capital,
                available=new_available,
                used=new_used,
                start_balance=wallet_balance.start_balance,
                currency=wallet_balance.currency,
                updated_at=datetime.utcnow(),
                id=wallet_balance.id,  # preserve primary key
            )

            # Upsert into database
            self.db.update_wallet_balance(updated_balance)

            logger.info(f"Updated {mode} balance: PnL {pnl:+.2f} -> available {new_available:.2f}")

        except Exception as e:
            logger.error(f"Error updating virtual balances: {e}")

    def sync_real_balance(self):
        """Sync real balance with Bybit account"""
        try:
            # Verify Bybit client initialization
            if not hasattr(self, 'client') or self.client is None:
                logger.error("Bybit client not initialized. Check API credentials in .env")
                return False

            # Ensure client connection
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

            # Fetch balance from Bybit
            result = self.client._make_request(
                "GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}
            )

            if not result or "list" not in result or not result["list"]:
                logger.warning("No account data in Bybit response")
                return False

            wallet = result["list"][0]

            # Total equity (net assets)
            total_equity = float(wallet.get("totalEquity") or 0.0)

            # Look for USDT balance
            coins = wallet.get("coin", [])
            usdt_coin = next((c for c in coins if c.get("coin") == "USDT"), None)

            if usdt_coin:
                # Use walletBalance (available to trade/withdraw)
                total_available = float(usdt_coin.get("walletBalance") or 0.0)
            else:
                total_available = total_equity  # fallback if USDT not found

            # Recalculate used accurately
            used = max(total_equity - total_available, 0.0)
            if used < 1e-6:  # suppress tiny floating point noise
                used = 0.0

            if total_available == 0 and total_equity > 0:
                logger.warning(
                    "Available balance is 0 while equity > 0. "
                    "Funds may be locked in margin, open positions, or collateral disabled in Bybit."
                )

            # Preserve start balance from DB
            existing_balance: Optional[DBWalletBalance] = self.db.get_wallet_balance("real")
            start_balance = (
                existing_balance.start_balance
                if existing_balance and existing_balance.start_balance > 0
                else total_equity
            )

            # Build wallet balance model
            wallet_balance = DBWalletBalance(
                trading_mode="real",
                capital=total_equity,
                available=total_available,
                used=used,
                start_balance=start_balance,
                currency="USDT",
                updated_at=datetime.now(timezone.utc),
                id=existing_balance.id if existing_balance else None,
            )

            # Save to DB
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

    def execute_virtual_trade(self, signal: Dict, trading_mode: str = "virtual") -> bool:
        """Execute a virtual trade based on a signal"""
        try:
            symbol = signal.get("symbol")
            if not symbol:
                logger.error("Symbol is required for executing trade")
                return False
                
            side = signal.get("side", "Buy").upper()  # Normalize to uppercase
            entry_price = signal.get("entry") or self.client.get_current_price(symbol)
            
            if entry_price <= 0:
                logger.error(f"Invalid entry price for {symbol}")
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(symbol, entry_price)
            
            # Create trade record
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
                "leverage": signal.get("leverage", 10),
                "sl": signal.get("sl"),  # Stop Loss from signal
                "tp": signal.get("tp"),  # Take Profit from signal
                "trail": signal.get("trail"),  # Trailing Stop from signal
                "liquidation": signal.get("liquidation"),  # Liquidation price from signal
                "margin_usdt": signal.get("margin_usdt")  # Margin from signal
            }
            
            # Save to database
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
            
    from datetime import datetime

async def execute_real_trade(self, signal: dict, trading_mode: str = "real") -> bool:
    """Execute a real trade on Bybit based on a signal, with 5% SL and 25% TP."""
    try:
        symbol = signal.get("symbol")
        if not symbol:
            logger.error("Symbol is required for executing trade")
            return False

        side = signal.get("side", "Buy").upper()  # Normalize to uppercase
        entry_price = signal.get("entry") or await self.client.get_current_price(symbol)
        if entry_price <= 0:
            logger.error(f"Invalid entry price for {symbol}")
            return False

        # Percentage-based SL and TP
        sl_percent = 5   # 5% stop loss
        tp_percent = 25  # 25% take profit

        if side == "BUY":
            stop_loss = entry_price * (1 - sl_percent / 100)
            take_profit = entry_price * (1 + tp_percent / 100)
        else:  # SELL
            stop_loss = entry_price * (1 + sl_percent / 100)
            take_profit = entry_price * (1 - tp_percent / 100)

        # Calculate position size
        position_size = self.calculate_position_size(symbol, entry_price)
        if position_size <= 0:
            logger.error(f"Invalid position size for {symbol}: {position_size}")
            return False

        # Place main order (without SL/TP)
        order_response = await self.client.place_order(
            symbol=symbol,
            side=side,
            qty=position_size,
            leverage=signal.get("leverage", 10)
        )

        if not order_response.get("success", False):
            logger.error(f"Failed to place order for {symbol}: {order_response.get('error', 'Unknown error')}")
            return False

        order_id = order_response.get("order_id", f"real_{symbol}_{int(datetime.now().timestamp())}")

        # Set conditional orders for SL and TP
        opposite_side = "Sell" if side == "BUY" else "Buy"

        # Stop Loss
        await self.client.create_conditional_order(
            symbol=symbol,
            side=opposite_side,
            order_type="Stop",
            qty=position_size,
            stop_px=stop_loss
        )

        # Take Profit
        await self.client.create_conditional_order(
            symbol=symbol,
            side=opposite_side,
            order_type="Limit",
            qty=position_size,
            price=take_profit
        )

        # Prepare trade data
        trade_data = {
            "symbol": symbol,
            "side": side,
            "qty": position_size,
            "entry_price": entry_price,
            "order_id": order_id,
            "virtual": trading_mode == "virtual",
            "status": "open",
            "score": signal.get("score"),
            "strategy": signal.get("strategy", "Auto"),
            "leverage": signal.get("leverage", 10),
            "sl": stop_loss,
            "tp": take_profit,
            "trail": signal.get("trail"),
            "liquidation": signal.get("liquidation"),
            "margin_usdt": signal.get("margin_usdt")
        }

        # Save trade to database asynchronously
        success = await self.db.add_trade(trade_data)
        if success:
            logger.info(
                f"Real trade executed: {symbol} {side} @ {entry_price}, "
                f"Qty: {position_size}, Order ID: {order_id}, "
                f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, "
                f"Trail: {trade_data['trail']}, Liquidation: {trade_data['liquidation']}, "
                f"Margin: {trade_data['margin_usdt']} USDT"
            )
            return True
        else:
            logger.error(f"Failed to save trade to database for {symbol}, Order ID: {order_id}")
            return False

    except Exception as e:
        logger.error(f"Error executing real trade for {symbol}: {e}", exc_info=True)
        return False


    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, "client"):
                self.client.close()
            logger.info("Trading engine closed")
        except Exception as e:
            logger.error(f"Error closing trading engine: {e}")