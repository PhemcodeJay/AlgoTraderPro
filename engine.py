import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import uuid
import numpy as np
from bybit_client import BybitClient
from db import db_manager, Trade, WalletBalance
from db import WalletBalance as DBWalletBalance, TradeModel
from settings import load_settings
from logging_config import get_trading_logger
from exceptions import (
    TradingException, create_error_context
)
from sqlalchemy import select

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
            
            # Ensure DB session is initialized
            if self.db.session is None:
                logger.error("Failed to initialize database session in TradingEngine init")
                raise TradingException("Database session initialization failed")
            
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
            logger.critical(f"Emergency stop: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to trigger emergency stop: {str(e)}")
            return False

    def calculate_position_size(self, symbol: str, entry_price: float) -> float:
        """Calculate position size based on risk"""
        try:
            if entry_price <= 0:
                logger.error(f"Invalid entry price for {symbol}: {entry_price}")
                return 0.0
            
            # Get wallet balance
            trading_mode = self.db.get_setting("trading_mode") or "virtual"
            wallet = self.db.get_wallet_balance(trading_mode)
            if not wallet:
                logger.error(f"No wallet balance found for {trading_mode} mode")
                return 0.0
            
            available_balance = wallet.available
            risk_amount = available_balance * self.max_risk_per_trade
            
            # Ensure position size doesn't exceed max_position_size
            position_size = min(risk_amount / entry_price, self.max_position_size / entry_price)
            position_size = max(position_size, 0.0)  # Ensure non-negative
            
            logger.info(f"Calculated position size for {symbol}: {position_size} units")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def get_open_virtual_trades(self) -> List[Trade]:
        """Get open virtual trades from database"""
        try:
            if not self.db.session:
                logger.error("Database session not initialized")
                return []
            
            # Query open virtual trades directly
            trades = self.db.session.execute(
                select(TradeModel).where(TradeModel.virtual == True, TradeModel.status == "open")
            ).scalars().all()
            # Manually construct Trade instances
            return [Trade(
                symbol=t.symbol,
                side=t.side,
                qty=t.qty,
                entry_price=t.entry_price,
                order_id=t.order_id,
                virtual=t.virtual,
                status=t.status,
                score=t.score,
                strategy=t.strategy,
                leverage=t.leverage,
                sl=t.sl,
                tp=t.tp,
                pnl=t.pnl,
                timestamp=t.timestamp,
                closed_at=t.closed_at
            ) for t in trades]
            
        except Exception as e:
            logger.error(f"Error getting open virtual trades: {e}")
            return []

    def get_open_real_trades(self) -> List[Trade]:
        """Get open real trades from database"""
        try:
            if not self.db.session:
                logger.error("Database session not initialized")
                return []
            
            # Query open real trades directly
            trades = self.db.session.execute(
                select(TradeModel).where(TradeModel.virtual == False, TradeModel.status == "open")
            ).scalars().all()
            # Manually construct Trade instances
            return [Trade(
                symbol=t.symbol,
                side=t.side,
                qty=t.qty,
                entry_price=t.entry_price,
                order_id=t.order_id,
                virtual=t.virtual,
                status=t.status,
                score=t.score,
                strategy=t.strategy,
                leverage=t.leverage,
                sl=t.sl,
                tp=t.tp,
                pnl=t.pnl,
                timestamp=t.timestamp,
                closed_at=t.closed_at
            ) for t in trades]
            
        except Exception as e:
            logger.error(f"Error getting open real trades: {e}")
            return []

    def calculate_virtual_pnl(self, trade: Dict) -> float:
        """Calculate PnL for a virtual trade"""
        try:
            symbol = trade.get("symbol")
            if not symbol:
                logger.error("Symbol is missing in trade data for PnL calculation")
                return 0.0
                
            side = trade.get("side", "Buy").upper()
            qty = float(trade.get("qty", 0))
            entry_price = float(trade.get("entry_price", 0))
            exit_price = float(trade.get("exit_price", self.client.get_current_price(symbol)))
            
            if qty <= 0 or entry_price <= 0 or exit_price <= 0:
                logger.warning(f"Invalid trade data for PnL calculation: {trade}")
                return 0.0
            
            if side in ["BUY", "LONG"]:
                pnl = (exit_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_price) * qty
            
            return pnl * trade.get("leverage", 10)
            
        except Exception as e:
            logger.error(f"Error calculating virtual PnL: {e}")
            return 0.0

    def update_virtual_balances(self, pnl: float, mode: str = "virtual") -> bool:
        """Update virtual wallet balance"""
        try:
            if not self.db.session:
                logger.error("Database session not initialized")
                return False
            
            wallet = self.db.get_wallet_balance(mode)
            if not wallet:
                logger.error(f"No wallet found for {mode} mode")
                return False
            
            new_available = wallet.available + pnl
            new_capital = wallet.capital + pnl
            new_used = new_capital - new_available
            
            updated_wallet = DBWalletBalance(
                id=wallet.id,
                trading_mode=mode,
                capital=new_capital,
                available=new_available,
                used=new_used,
                start_balance=wallet.start_balance,
                currency=wallet.currency,
                updated_at=datetime.now(timezone.utc)
            )
            
            success = self.db.update_wallet_balance(updated_wallet)
            if success:
                logger.info(f"Updated {mode} wallet balance: capital={new_capital:.2f}, available={new_available:.2f}")
            return success
            
        except Exception as e:
            logger.error(f"Error updating virtual balances: {e}")
            return False

    def sync_real_balance(self) -> bool:
        """Sync real wallet balance with Bybit"""
        try:
            if not self.db.session:
                logger.error("Database session not initialized")
                return False
            
            if not self.client.is_connected():
                logger.error("Bybit client not connected")
                return False
            
            balances = self.client.get_account_balance()
            usdt_balance = balances.get("USDT", {})
            
            def safe_get(balance, key, default=0.0):
                if isinstance(balance, dict):
                    return balance.get(key, default)
                return getattr(balance, key, default)

            capital = float(safe_get(usdt_balance, "total"))
            available = float(safe_get(usdt_balance, "available"))
            used = float(safe_get(usdt_balance, "used"))

                        
            wallet = self.db.get_wallet_balance("real")
            if not wallet:
                wallet = DBWalletBalance(
                    trading_mode="real",
                    capital=0.0,
                    available=0.0,
                    used=0.0,
                    start_balance=0.0,
                    currency="USDT",
                    updated_at=datetime.now(timezone.utc)
                )
            
            updated_wallet = DBWalletBalance(
                id=wallet.id,
                trading_mode="real",
                capital=capital,
                available=available,
                used=used,
                start_balance=wallet.start_balance or capital,
                currency="USDT",
                updated_at=datetime.now(timezone.utc)
            )
            
            success = self.db.update_wallet_balance(updated_wallet)
            if success:
                logger.info(f"Real balance synced: capital={updated_wallet.capital:.2f}, available={updated_wallet.available:.2f}")
            return success
            
        except Exception as e:
            logger.error(f"Error syncing real balance: {e}", exc_info=True)
            return False

    def sync_real_trades(self):
        try:
            if not self.db.session:
                logger.error("Database session not initialized")
                return False

            positions = self.client.get_positions()
            orders = self.client.get_open_orders()
            
            # Clear existing real trades in DB
            real_trades = self.db.session.query(TradeModel).filter(TradeModel.virtual == False).all()
            for t in real_trades:
                self.db.session.delete(t)
            self.db.session.commit()
            
            # Add positions to DB as trades
            for pos in positions:
                trade_data = {
                    "symbol": pos["symbol"],
                    "side": pos["side"],
                    "qty": pos["size"],
                    "entry_price": pos["entry_price"],
                    "order_id": f"pos_{pos['symbol']}_{str(uuid.uuid4())[:8]}",
                    "virtual": False,
                    "status": "open",
                    "pnl": pos["unrealized_pnl"],
                    "leverage": pos["leverage"],
                    "timestamp": datetime.now(timezone.utc)
                }
                self.db.add_trade(trade_data)
            
            # Add open orders as pending trades
            for order in orders:
                trade_data = {
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "qty": order["qty"],
                    "entry_price": order["price"],
                    "order_id": order["order_id"],
                    "virtual": False,
                    "status": "pending",
                    "pnl": 0.0,
                    "leverage": self.settings.get("LEVERAGE", 10),
                    "timestamp": order["timestamp"]
                }
                self.db.add_trade(trade_data)
            
            logger.info("Real trades synced from Bybit")
            return True
        except Exception as e:
            logger.error(f"Error syncing real trades: {e}")
            if self.db.session:
                self.db.session.rollback()
            return False
            
    def execute_virtual_trade(self, signal: Dict, trading_mode: str = "virtual") -> bool:
        """Execute a virtual trade based on a signal"""
        try:
            symbol = signal.get("symbol")
            if not symbol:
                logger.error("Symbol is required for executing trade")
                return False
                
            side = signal.get("side", "Buy")
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
                "virtual": True,
                "status": "open",
                "score": signal.get("score"),
                "strategy": signal.get("strategy", "Auto"),
                "leverage": signal.get("leverage", 10),
                "sl": float(signal.get("sl", 0.0)) if signal.get("sl") is not None else 0.0,
                "tp": float(signal.get("tp", 0.0)) if signal.get("tp") is not None else 0.0
            }
            
            # Save to database
            success = self.db.add_trade(trade_data)
            if success:
                logger.info(f"Virtual trade executed: {symbol} {side} @ {entry_price}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error executing virtual trade: {e}")
            return False

    async def execute_real_trade(self, signal: Dict) -> bool:
        """Execute a real trade based on a signal using Bybit API"""
        try:
            symbol = signal.get("symbol")
            if not symbol:
                logger.error("Symbol is required for real trade")
                return False

            side = signal.get("side", "Buy").title()
            entry_price = signal.get("entry") or self.client.get_current_price(symbol)
            if entry_price <= 0:
                logger.error(f"Invalid entry price for {symbol}")
                return False

            qty = self.calculate_position_size(symbol, entry_price)
            if qty <= 0:
                logger.error(f"Invalid qty for {symbol}")
                return False

            # Use stop_loss and take_profit from signal
            stop_loss = float(signal.get("sl", 0.0)) if signal.get("sl") is not None else 0.0
            take_profit = float(signal.get("tp", 0.0)) if signal.get("tp") is not None else 0.0

            # Place order on Bybit
            result = await self.client.place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=signal.get("leverage", 10)
            )
            if "error" in result:
                logger.error(f"Failed to place real order: {result['error']}")
                return False

            # Create trade record from result
            trade_data = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_price": result.get("price", entry_price),
                "order_id": result.get("order_id"),
                "virtual": False,
                "status": "open",
                "score": signal.get("score"),
                "strategy": signal.get("strategy", "Auto"),
                "leverage": signal.get("leverage", 10),
                "sl": stop_loss,
                "tp": take_profit
            }

            # Save to database
            success = self.db.add_trade(trade_data)
            if success:
                logger.info(f"Real trade executed: {symbol} {side} @ {entry_price}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error executing real trade: {e}")
            return False

    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, "client"):
                self.client.close()
            logger.info("Trading engine closed")
        except Exception as e:
            logger.error(f"Error closing trading engine: {e}")