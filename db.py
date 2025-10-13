import os
import json
import time
from datetime import datetime, timezone
from dateutil import parser 
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy import create_engine, Integer, String, Float, DateTime, Boolean, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError, DatabaseError
import uuid
from logging_config import get_logger
from exceptions import (
    DatabaseException, DatabaseConnectionException, DatabaseTransactionException,
    DatabaseIntegrityException, DatabaseErrorRecoveryStrategy, create_error_context
)

logger = get_logger(__name__, structured_format=True)

Base = declarative_base()

@dataclass
class Signal:
    symbol: str
    interval: str
    signal_type: str
    score: float
    indicators: Dict
    strategy: str = "Auto"
    side: str = "BUY"
    sl: Optional[float] = None
    tp: Optional[float] = None
    trail: Optional[float] = None
    liquidation: Optional[float] = None
    leverage: int = 15
    margin_usdt: Optional[float] = None
    entry: Optional[float] = None
    market: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: Union[str, None] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        self.side = self.side.upper()
        if self.created_at and self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        return data

@dataclass
class Trade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    order_id: str
    virtual: bool = True
    status: str = "open"
    exit_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    pnl: Optional[float] = None
    score: Optional[float] = None
    strategy: str = "Auto"
    leverage: int = 15
    trail: Optional[float] = None
    liquidation: Optional[float] = None
    margin_usdt: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    id: Union[str, None] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.timestamp and self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        if self.closed_at and self.closed_at.tzinfo is None:
            self.closed_at = self.closed_at.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.timestamp:
            data["timestamp"] = self.timestamp.isoformat()
        if self.closed_at:
            data["closed_at"] = self.closed_at.isoformat()
        return data

@dataclass
class WalletBalance:
    trading_mode: str
    capital: float
    available: float
    used: float
    start_balance: float
    currency: str = "USDT"
    updated_at: Optional[datetime] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data

class SignalModel(Base):
    __tablename__ = 'signals'
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    indicators: Mapped[str] = mapped_column(Text, nullable=False)
    strategy: Mapped[str] = mapped_column(String(20), default="Auto")
    side: Mapped[str] = mapped_column(String(10), default="Buy")
    sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trail: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    liquidation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    leverage: Mapped[int] = mapped_column(Integer, default=15)
    margin_usdt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_signal(self) -> Signal:
        return Signal(
            id=str(self.id) if self.id else None,
            symbol=self.symbol,
            interval=self.interval,
            signal_type=self.signal_type,
            score=self.score,
            indicators=json.loads(self.indicators),
            strategy=self.strategy,
            side=self.side,
            sl=self.sl,
            tp=self.tp,
            trail=self.trail,
            liquidation=self.liquidation,
            leverage=self.leverage,
            margin_usdt=self.margin_usdt,
            entry=self.entry,
            market=self.market,
            created_at=self.created_at,
        )

class TradeModel(Base):
    __tablename__ = 'trades'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    order_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    virtual: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(20), default="open")
    sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trail: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    liquidation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    margin_usdt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    strategy: Mapped[str] = mapped_column(String(20), default="Auto")
    leverage: Mapped[int] = mapped_column(Integer, default=15)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_trade(self) -> Trade:
        return Trade(
            id=str(self.id) if self.id is not None else None,
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            entry_price=self.entry_price,
            order_id=self.order_id,
            virtual=self.virtual,
            status=self.status,
            exit_price=self.exit_price,
            sl=self.sl,
            tp=self.tp,
            pnl=self.pnl,
            trail=self.trail,
            liquidation=self.liquidation,
            margin_usdt=self.margin_usdt,
            score=self.score,
            strategy=self.strategy,
            leverage=self.leverage,
            timestamp=self.timestamp,
            closed_at=self.closed_at,
        )

class WalletBalanceModel(Base):
    __tablename__ = 'wallet_balances'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trading_mode: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    capital: Mapped[float] = mapped_column(Float, nullable=False)
    available: Mapped[float] = mapped_column(Float, nullable=False)
    used: Mapped[float] = mapped_column(Float, nullable=False)
    start_balance: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="USDT")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_wallet_balance(self) -> WalletBalance:
        return WalletBalance(
            trading_mode=self.trading_mode,
            capital=self.capital,
            available=self.available,
            used=self.used,
            start_balance=self.start_balance,
            currency=self.currency,
            updated_at=self.updated_at,
            id=self.id
        )

class SettingsModel(Base):
    __tablename__ = 'settings'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.session = None
        self.Session = None
        self.recovery_strategy = DatabaseErrorRecoveryStrategy(max_retries=3, delay=1.0)
        self._initialize_db()

    def _initialize_db(self):
        try:
            db_url = os.getenv("DATABASE_URL", "sqlite:///trader.db")
            self.engine = create_engine(db_url, echo=False)
            self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
            self.session = self.Session()
            Base.metadata.create_all(self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise DatabaseConnectionException(
                f"Database initialization failed: {str(e)}",
                context=create_error_context(module=__name__, function='_initialize_db')
            )

    def is_connected(self) -> bool:
        try:
            if not self.session:
                return False
            self.session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False

    def _execute_with_retry(self, operation: Callable, operation_name: str):
        return self.recovery_strategy.execute_with_retry(operation, operation_name)

    def _safe_transaction(self, operation: Callable, operation_type: str = "UNKNOWN", table: str = "unknown"):
        def transaction_wrapper():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            try:
                result = operation()
                self.session.commit()
                return result
            except IntegrityError as e:
                self.session.rollback()
                error_context = create_error_context(
                    module=__name__,
                    function=operation.__name__,
                    extra_data={'table': table, 'operation_type': operation_type}
                )
                raise DatabaseIntegrityException(
                    f"Integrity error in {operation_type} on {table}: {str(e)}",
                    context=error_context,
                    original_exception=e
                )
            except SQLAlchemyError as e:
                self.session.rollback()
                error_context = create_error_context(
                    module=__name__,
                    function=operation.__name__,
                    extra_data={'table': table, 'operation_type': operation_type}
                )
                raise DatabaseTransactionException(
                    f"Transaction error in {operation_type} on {table}: {str(e)}",
                    context=error_context,
                    original_exception=e
                )
            except Exception as e:
                self.session.rollback()
                logger.error(f"Unexpected error in {operation_type} on {table}: {str(e)}")
                raise

        return self._execute_with_retry(transaction_wrapper, f"{operation_type}_{table}")

    def add_signal(self, signal: Signal) -> bool:
        try:
            def _add_signal():
                if not self.session:
                    raise DatabaseConnectionException("Database session not initialized")
                signal_model = SignalModel(
                    id=uuid.UUID(signal.id) if signal.id else uuid.uuid4(),
                    symbol=signal.symbol,
                    interval=signal.interval,
                    signal_type=signal.signal_type,
                    score=signal.score,
                    indicators=json.dumps(signal.indicators),
                    strategy=signal.strategy,
                    side=signal.side,
                    sl=signal.sl,
                    tp=signal.tp,
                    trail=signal.trail,
                    liquidation=signal.liquidation,
                    leverage=signal.leverage,
                    margin_usdt=signal.margin_usdt,
                    entry=signal.entry,
                    market=signal.market,
                    created_at=signal.created_at
                )
                self.session.add(signal_model)
                return True
            return self._safe_transaction(operation=_add_signal, operation_type="INSERT", table="signals")
        except DatabaseException:
            logger.error(f"Database exception adding signal for {signal.symbol}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding signal for {signal.symbol}: {str(e)}")
            return False

    def add_trade(self, trade_data: Dict) -> bool:
        try:
            def _add_trade():
                if not self.session:
                    raise DatabaseConnectionException("Database session not initialized")
                trade_model = TradeModel(
                    symbol=trade_data.get("symbol"),
                    side=trade_data.get("side"),
                    qty=trade_data.get("qty"),
                    entry_price=trade_data.get("entry_price"),
                    order_id=trade_data.get("order_id"),
                    virtual=trade_data.get("virtual", True),
                    status=trade_data.get("status", "open"),
                    score=trade_data.get("score"),
                    strategy=trade_data.get("strategy", "Auto"),
                    leverage=trade_data.get("leverage", 15),
                    sl=trade_data.get("sl"),
                    tp=trade_data.get("tp"),
                    trail=trade_data.get("trail"),
                    liquidation=trade_data.get("liquidation"),
                    margin_usdt=trade_data.get("margin_usdt"),
                    timestamp=trade_data.get("timestamp", datetime.now(timezone.utc))
                )
                self.session.add(trade_model)
                return True
            return self._safe_transaction(operation=_add_trade, operation_type="INSERT", table="trades")
        except DatabaseException:
            logger.error(f"Database exception adding trade for {trade_data.get('symbol')}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding trade for {trade_data.get('symbol')}: {str(e)}")
            return False

    def get_trade_by_order_id(self, order_id: str) -> Optional[Trade]:
        """Retrieve a trade by order_id"""
        try:
            def _get_trade():
                if not self.session:
                    raise DatabaseConnectionException("Database session not initialized")
                trade = self.session.query(TradeModel).filter(TradeModel.order_id == order_id).first()
                return trade.to_trade() if trade else None
            return self._execute_with_retry(_get_trade, f"get_trade_by_order_id_{order_id}")
        except Exception as e:
            logger.error(f"Error getting trade by order_id {order_id}: {e}")
            return None

    def get_open_trades(self, virtual: bool) -> List[Trade]:
        """Retrieve open trades filtered by virtual status"""
        try:
            def _get_open_trades():
                if not self.session:
                    raise DatabaseConnectionException("Database session not initialized")
                trades = self.session.query(TradeModel).filter(
                    TradeModel.status == "open",
                    TradeModel.virtual == virtual
                ).all()
                return [t.to_trade() for t in trades]
            return self._execute_with_retry(_get_open_trades, f"get_open_trades_virtual_{virtual}")
        except Exception as e:
            logger.error(f"Error getting open trades (virtual={virtual}): {e}")
            return []

    def update_wallet_balance(self, wallet_balance: WalletBalance) -> bool:
        try:
            def _update_wallet_balance():
                if not self.session:
                    raise DatabaseConnectionException("Database session not initialized")
                existing = self.session.query(WalletBalanceModel).filter(
                    WalletBalanceModel.trading_mode == wallet_balance.trading_mode
                ).first()
                if existing:
                    existing.capital = wallet_balance.capital
                    existing.available = wallet_balance.available
                    existing.used = wallet_balance.used
                    existing.start_balance = wallet_balance.start_balance
                    existing.currency = wallet_balance.currency
                    existing.updated_at = wallet_balance.updated_at or datetime.now(timezone.utc)
                else:
                    new_balance = WalletBalanceModel(
                        trading_mode=wallet_balance.trading_mode,
                        capital=wallet_balance.capital,
                        available=wallet_balance.available,
                        used=wallet_balance.used,
                        start_balance=wallet_balance.start_balance,
                        currency=wallet_balance.currency,
                        updated_at=wallet_balance.updated_at or datetime.now(timezone.utc)
                    )
                    self.session.add(new_balance)
                return True
            return self._safe_transaction(operation=_update_wallet_balance, operation_type="UPSERT", table="wallet_balances")
        except DatabaseException:
            logger.error(f"Database exception updating wallet balance for {wallet_balance.trading_mode}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating wallet balance for {wallet_balance.trading_mode}: {e}")
            return False

    def get_signals(self, limit: int = 100) -> List[Signal]:
        try:
            def _get_signals():
                if not self.session:
                    raise DatabaseConnectionException("Database session not initialized")
                signals = self.session.query(SignalModel).order_by(
                    SignalModel.created_at.desc()
                ).limit(limit).all()
                return [s.to_signal() for s in signals]
            return self._execute_with_retry(_get_signals, "get_signals")
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return []

    def get_trades(self, limit: int = 100) -> List[Trade]:
        try:
            def _get_trades():
                if not self.session:
                    raise DatabaseConnectionException("Database session not initialized")
                trades = self.session.query(TradeModel).order_by(
                    TradeModel.timestamp.desc()
                ).limit(limit).all()
                return [t.to_trade() for t in trades]
            return self._execute_with_retry(_get_trades, "get_trades")
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    def get_wallet_balance(self, trading_mode: str) -> Optional[WalletBalance]:
        try:
            def _get_wallet_balance():
                if not self.session:
                    raise DatabaseConnectionException("Database session not initialized")
                balance = self.session.query(WalletBalanceModel).filter(
                    WalletBalanceModel.trading_mode == trading_mode
                ).first()
                return balance.to_wallet_balance() if balance else None
            return self._execute_with_retry(_get_wallet_balance, f"get_wallet_balance_{trading_mode}")
        except Exception as e:
            logger.error(f"Error getting wallet balance for {trading_mode}: {e}")
            return None

    def migrate_capital_json_to_db(self, capital_file_path: str = "capital.json") -> bool:
        try:
            if not os.path.exists(capital_file_path):
                logger.warning(f"Capital file {capital_file_path} not found, initializing default balances")
                default_virtual = WalletBalance(
                    trading_mode="virtual",
                    capital=100.0,
                    available=100.0,
                    used=0.0,
                    start_balance=100.0,
                    currency="USDT",
                    updated_at=datetime.now(timezone.utc),
                )
                default_real = WalletBalance(
                    trading_mode="real",
                    capital=0.0,
                    available=0.0,
                    used=0.0,
                    start_balance=0.0,
                    currency="USDT",
                    updated_at=datetime.now(timezone.utc),
                )
                self.update_wallet_balance(default_virtual)
                self.update_wallet_balance(default_real)
                return True

            with open(capital_file_path, "r") as f:
                capital_data = json.load(f)

            if "virtual" in capital_data:
                v = capital_data["virtual"]
                virtual_balance = WalletBalance(
                    trading_mode="virtual",
                    capital=float(v.get("capital", 100.0)),
                    available=float(v.get("available", 100.0)),
                    used=float(v.get("used", 0.0)),
                    start_balance=float(v.get("start_balance", 100.0)),
                    currency=v.get("currency", "USDT"),
                    updated_at=datetime.now(timezone.utc),
                )
                self.update_wallet_balance(virtual_balance)
                logger.info("Virtual balance migrated to database")

            if "real" in capital_data:
                r = capital_data["real"]
                virtual_balance = WalletBalance(
                    trading_mode="real",
                    capital=float(r.get("capital", 0.0)),
                    available=float(r.get("available", 0.0)),
                    used=float(r.get("used", 0.0)),
                    start_balance=float(r.get("start_balance", 0.0)),
                    currency=r.get("currency", "USDT"),
                    updated_at=datetime.now(timezone.utc),
                )
                self.update_wallet_balance(virtual_balance)
                logger.info("Real balance migrated to database")

            logger.info("Capital.json data successfully migrated to database")
            return True

        except Exception as e:
            logger.error(f"Error migrating capital.json to database: {e}")
            return False

    def get_all_wallet_balances(self) -> Dict[str, WalletBalance]:
        if not self.session:
            logger.error("Database session not initialized")
            return {}
        try:
            models = self.session.query(WalletBalanceModel).all()
            return {model.trading_mode: model.to_wallet_balance() for model in models}
        except Exception as e:
            logger.error(f"Error getting all wallet balances: {e}")
            return {}

    def save_setting(self, key: str, value: str) -> bool:
        def _save_setting_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            setting = self.session.query(SettingsModel).filter(SettingsModel.key == key).first()
            if setting:
                setting.value = value
                setting.updated_at = datetime.now(timezone.utc)
            else:
                setting = SettingsModel(key=key, value=value, updated_at=datetime.now(timezone.utc))
                self.session.add(setting)
            return True

        try:
            return self._safe_transaction(operation=_save_setting_operation, operation_type="UPSERT", table="settings")
        except DatabaseException:
            logger.error(f"Database exception saving setting {key}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving setting {key}: {str(e)}")
            return False

    def get_setting(self, key: str) -> Optional[str]:
        def _get_setting_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            setting = self.session.query(SettingsModel).filter(SettingsModel.key == key).first()
            return setting.value if setting else None

        try:
            return self._execute_with_retry(_get_setting_operation, f"get_setting_{key}")
        except DatabaseException:
            logger.error(f"Failed to get setting {key}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting setting {key}: {str(e)}")
            return None

    def close(self):
        try:
            if self.session:
                self.session.close()
            if self.engine:
                self.engine.dispose()
            logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Failed to close database connection: {str(e)}")

    def get_connection_stats(self) -> Dict[str, Any]:
        stats = {
            'pool_status': 'unknown',
            'checked_out': 0,
            'overflow': 0,
            'pool_size': 0
        }
        
        try:
            if self.engine and hasattr(self.engine, 'pool'):
                pool = self.engine.pool
                stats.update({
                    'pool_status': 'active',
                    'checked_out': getattr(pool, 'checkedout', 0),
                    'overflow': getattr(pool, 'overflow', 0),
                    'pool_size': getattr(pool, 'size', 0)
                })
        except Exception as e:
            stats['error'] = str(e)
            
        return stats

db_manager = DatabaseManager()