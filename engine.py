import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
from bybit_client import BybitClient
from db import db_manager, WalletBalance, Trade
from logging_config import get_trading_logger
from utils import sync_real_wallet_balance, calculate_portfolio_metrics, normalize_signal
from indicators import scan_multiple_symbols
from signal_generator import generate_signals
from automated_trader import AutomatedTrader
from exceptions import TradingException, create_error_context

logger = get_trading_logger('engine')

class TradingEngine:
    def __init__(self, client: Optional[BybitClient] = None):
        try:
            self.client = client if client else BybitClient()
            self.settings = self.load_settings()
            self.db = db_manager
            self._candle_cache = {}
            
            # Position safety limits
            self.max_position_size = self.settings.get("MAX_POSITION_SIZE", 10000.0)  # USDT
            self.max_open_positions = self.settings.get("MAX_OPEN_POSITIONS", 10)
            self.max_daily_loss = self.settings.get("MAX_DAILY_LOSS", 1000.0)  # USDT
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
            
            # Initialize AutomatedTrader
            self.trader = AutomatedTrader(self, self.client)
            
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
    
    def load_settings(self) -> Dict[str, Any]:
        """Load trading settings from config or DB"""
        try:
            return {
                "SCAN_INTERVAL": 300,  # 5 minutes
                "TOP_N_SIGNALS": 10,
                "LEVERAGE": 10,
                "MAX_OPEN_POSITIONS": 10,
                "MAX_RISK_PER_TRADE": 0.05,
                "MAX_POSITION_SIZE": 10000.0,
                "MAX_DAILY_LOSS": 1000.0,
                "MIN_SIGNAL_SCORE": 60.0
            }
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return {}

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
            open_trades = self.db.get_open_trades()
            if open_trades:
                async def _close_all():
                    coroutines = [
                        self.trader._close_trade(
                            trade,
                            self.client.get_current_price(trade.symbol),
                            reason,
                            "real" if not trade.virtual else "virtual"
                        )
                        for trade in open_trades
                    ]
                    return await asyncio.gather(*coroutines, return_exceptions=True)

                asyncio.run(_close_all())

            return True

        except Exception as e:
            logger.critical(f"Failed to activate emergency stop: {str(e)}")
            return False

    def get_settings(self) -> Tuple[int, int, float]:
        """Get current scan interval, top N signals, and minimum signal score"""
        return (
            self.settings.get("SCAN_INTERVAL", 300),
            self.settings.get("TOP_N_SIGNALS", 10),
            self.settings.get("MIN_SIGNAL_SCORE", 60.0)
        )

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
        try:
            return self.client.get_available_symbols("linear")
        except Exception as e:
            logger.error(f"Error fetching USDT symbols: {e}")
            return [
                "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT",
                "XRPUSDT", "BNBUSDT", "AVAXUSDT"
            ]

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

            risk_pct = risk_percent or self.settings.get("MAX_RISK_PER_TRADE", 0.05)
            lev = leverage or self.settings.get("LEVERAGE", 10)

            mode = "real" if self.db.get_setting("trading_mode") == "real" else "virtual"
            wallet_balance = self.db.get_wallet_balance(mode)
            if not wallet_balance:
                logger.warning(f"No wallet balance found for {mode} mode, using default")
                balance = 1000.0
            else:
                balance = available_balance if available_balance is not None else wallet_balance.available

            symbol_info = self.get_symbol_info(symbol)
            min_qty = float(symbol_info.get("lotSizeFilter", {}).get("minOrderQty", 0.001))
            qty_step = float(symbol_info.get("lotSizeFilter", {}).get("qtyStep", 0.001))

            risk_amount = balance * risk_pct
            candles = self.get_cached_candles(symbol, "60", 14)
            highs = [float(c["high"]) for c in candles]
            lows = [float(c["low"]) for c in candles]
            atr = np.mean([high - low for high, low in zip(highs, lows)]) if highs and lows else 0.0

            stop_loss_distance = atr * 2 / entry_price if atr > 0 else 0.05
            position_size = risk_amount / stop_loss_distance
            position_size = min(position_size, self.max_position_size)
            position_size = max(min_qty, math.floor(position_size / qty_step) * qty_step)

            logger.info(f"Calculated position size for {symbol}: {position_size}")
            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def get_closed_real_trades(self) -> List[Dict]:
        """Get closed real trades"""
        try:
            all_trades = self.db.get_trades()
            closed_real_trades = [trade.to_dict() for trade in all_trades if not trade.virtual and trade.status == "closed"]
            return closed_real_trades
        except Exception as e:
            logger.error(f"Error getting closed real trades: {e}")
            return []

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive trading statistics"""
        try:
            all_trades = self.db.get_trades()
            virtual_trades = [t.to_dict() for t in all_trades if t.virtual]
            real_trades = [t.to_dict() for t in all_trades if not t.virtual]
            
            virtual_stats = calculate_portfolio_metrics(virtual_trades)
            real_stats = calculate_portfolio_metrics(real_trades)
            overall_stats = calculate_portfolio_metrics([t.to_dict() for t in all_trades])
            
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
            wallet_balance = self.db.get_wallet_balance(mode)
            if not wallet_balance:
                default_balance = WalletBalance(
                    trading_mode=mode,
                    capital=1000.0 if mode == "virtual" else 0.0,
                    available=1000.0 if mode == "virtual" else 0.0,
                    used=0.0,
                    start_balance=1000.0 if mode == "virtual" else 0.0,
                    currency="USDT",
                    updated_at=datetime.now(timezone.utc),
                )
                self.db.update_wallet_balance(default_balance)
                wallet_balance = self.db.get_wallet_balance(mode)

            if not wallet_balance:
                logger.error(f"Failed to get or create wallet balance for mode: {mode}")
                return

            new_available = max(0.0, wallet_balance.available + pnl)
            new_capital = wallet_balance.capital + pnl
            new_used = max(0.0, new_capital - new_available)

            updated_balance = WalletBalance(
                trading_mode=mode,
                capital=new_capital,
                available=new_available,
                used=new_used,
                start_balance=wallet_balance.start_balance,
                currency=wallet_balance.currency,
                updated_at=datetime.now(timezone.utc),
                id=wallet_balance.id,
            )

            self.db.update_wallet_balance(updated_balance)
            logger.info(f"Updated {mode} balance: PnL {pnl:+.2f} -> available {new_available:.2f}")

        except Exception as e:
            logger.error(f"Error updating virtual balances: {e}")

    def sync_real_balance(self) -> bool:
        """Sync real balance with Bybit account"""
        try:
            success = sync_real_wallet_balance(self.client)
            if success:
                logger.info("Real balance synced successfully")
                return True
            else:
                logger.error("Failed to sync real balance")
                return False

        except Exception as e:
            logger.error(f"Error syncing real balance: {e}", exc_info=True)
            return False

    def execute_virtual_trade(self, signal: Dict, trading_mode: str = "virtual") -> bool:
        """Execute a virtual trade based on a signal"""
        try:
            signal = normalize_signal(signal)
            symbol = signal.get("symbol")
            if not symbol:
                logger.error("Symbol is required for executing trade")
                return False
                
            side = signal.get("side", "Buy").upper()
            entry_price = signal.get("entry") or self.client.get_current_price(symbol)
            
            if entry_price <= 0:
                logger.error(f"Invalid entry price for {symbol}")
                return False
            
            position_size = self.calculate_position_size(symbol, entry_price)
            if position_size <= 0:
                logger.error(f"Invalid position size for {symbol}")
                return False
            
            trade_data = {
                "symbol": signal.get("symbol"),
                "side": side,
                "qty": position_size,
                "entry_price": entry_price,
                "order_id": f"virtual_{symbol}_{int(datetime.now(timezone.utc).timestamp())}",
                "virtual": trading_mode == "virtual",
                "status": "open",
                "score": signal.get("score"),
                "strategy": signal.get("strategy", "Auto"),
                "leverage": signal.get("leverage", 10),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            success = self.trader._execute_signal(trade_data, trading_mode)
            if success:
                self.db.add_trade(trade_data)
                logger.info(f"Trade executed: {symbol} {side} @ {entry_price}", extra=trade_data)
                if trading_mode == "virtual":
                    margin_used = (entry_price * position_size) / trade_data["leverage"]
                    self.update_virtual_balances(-margin_used, trading_mode)
                self.trade_count += 1
                self.successful_trades += 1
                return True
            else:
                logger.warning(f"Trade execution failed for {symbol}")
                self._consecutive_failures += 1
                self.failed_trades += 1
                return False
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            self._consecutive_failures += 1
            self.failed_trades += 1
            return False

    def run_trading_cycle(self) -> None:
        """Run a single trading cycle: scan symbols, generate signals, execute trades"""
        try:
            if not self.is_trading_enabled():
                logger.warning("Trading disabled or emergency stop active")
                return

            self.sync_real_balance()
            symbols = self.get_usdt_symbols()
            if not symbols:
                logger.warning("No tradable symbols available")
                return

            scan_interval, top_n_signals, min_signal_score = self.get_settings()
            signals = generate_signals(symbols, interval="60")
            signals = [signal for signal in signals if signal.get("score", 0) >= min_signal_score][:top_n_signals]

            if not signals:
                logger.info("No valid signals generated")
                return

            open_trades = self.db.get_open_trades()
            if len(open_trades) >= self.max_open_positions:
                logger.info(f"Max open positions ({self.max_open_positions}) reached")
                return

            for signal in signals:
                symbol = signal.get("symbol")
                trading_mode = "real" if self.db.get_setting("trading_mode") == "real" else "virtual"
                if self.execute_virtual_trade(signal, trading_mode):
                    self.trade_count += 1
                    self.successful_trades += 1
                else:
                    self.failed_trades += 1

            trades = [trade.to_dict() for trade in self.db.get_trades(limit=100)]
            metrics = calculate_portfolio_metrics(trades)
            self._daily_pnl = metrics["total_pnl"]
            logger.info("Portfolio metrics updated", extra=metrics)

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self._consecutive_failures += 1
            if self._consecutive_failures >= 10:
                self.emergency_stop("Too many consecutive failures in trading cycle")

    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, "client"):
                self.client.close()
            logger.info("Trading engine closed")
        except Exception as e:
            logger.error(f"Error closing trading engine: {e}")