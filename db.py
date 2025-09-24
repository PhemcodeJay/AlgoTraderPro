import os
import json
import time
from datetime import datetime, timezone
from dateutil import parser 
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from sqlalchemy import create_engine, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
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
    leverage: int = 10
    margin_usdt: Optional[float] = None
    entry: Optional[float] = None
    market: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: Union[str, None] = None

    def __post_init__(self):
        # Auto-generate UUID if no id provided
        if not self.id:
            self.id = str(uuid.uuid4())
        # Normalize side to uppercase
        self.side = self.side.upper()
        # Ensure created_at is UTC-aware
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
    pnl: Optional[float] = None
    score: Optional[float] = None
    strategy: str = "Manual"
    leverage: int = 10
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    id: Union[str, None] = None

    def __post_init__(self):
        # Auto-generate UUID if no id provided
        if not self.id:
            self.id = str(uuid.uuid4())
        # Normalize datetimes to UTC-aware
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
    trading_mode: str  # 'virtual' or 'real'
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

# SQLAlchemy Models
class SignalModel(Base):
    __tablename__ = 'signals'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    indicators: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string
    strategy: Mapped[str] = mapped_column(String(20), default="Auto")
    side: Mapped[str] = mapped_column(String(10), default="Buy")
    sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trail: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    liquidation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    leverage: Mapped[int] = mapped_column(Integer, default=10)
    margin_usdt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_signal(self) -> Signal:
        return Signal(
            id=str(self.id) if self.id is not None else None,  # ✅ Cast int → str
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
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    strategy: Mapped[str] = mapped_column(String(20), default="Manual")
    leverage: Mapped[int] = mapped_column(Integer, default=10)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_trade(self) -> Trade:
        return Trade(
            id=str(self.id) if self.id is not None else None,  # ✅ Cast int → str
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            entry_price=self.entry_price,
            order_id=self.order_id,
            virtual=self.virtual,
            status=self.status,
            exit_price=self.exit_price,
            pnl=self.pnl,
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
            id=self.id,
            trading_mode=self.trading_mode,
            capital=self.capital,
            available=self.available,
            used=self.used,
            start_balance=self.start_balance,
            currency=self.currency,
            updated_at=self.updated_at
        )

class SettingsModel(Base):
    __tablename__ = 'settings'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    value: Mapped[str] = mapped_column(String(255), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.session = None
        self._initialize_db()
        self.recovery_strategy = DatabaseErrorRecoveryStrategy()

    def _initialize_db(self):
        try:
            # === PostgreSQL Configuration ===
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", 5432)
            db_name = os.getenv("DB_NAME", "Algotrader")
            db_user = os.getenv("DB_USER", "postgres")
            db_password = os.getenv("DB_PASSWORD", "1234")

            # Construct PostgreSQL URL
            postgres_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

            # Try connecting to PostgreSQL
            self.engine = create_engine(postgres_url, echo=False)
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            logger.info("PostgreSQL database initialized successfully")

        except OperationalError as pg_err:
            logger.warning(f"PostgreSQL connection failed: {pg_err}. Falling back to SQLite.")
            try:
                # Fallback to SQLite
                sqlite_url = os.getenv("SQLITE_URL", "sqlite:///algotrader.db")
                self.engine = create_engine(sqlite_url, echo=False)
                Base.metadata.create_all(self.engine)
                Session = sessionmaker(bind=self.engine)
                self.session = Session()
                logger.info("SQLite database initialized successfully")
            except Exception as sqlite_err:
                error_context = create_error_context(
                    module=__name__,
                    function='_initialize_db'
                )
                logger.error(f"Failed to initialize SQLite database: {sqlite_err}")
                raise DatabaseConnectionException(
                    f"Database initialization failed: {sqlite_err}",
                    context=error_context,
                    original_exception=sqlite_err
                )

        except Exception as e:
            error_context = create_error_context(
                module=__name__,
                function='_initialize_db'
            )
            logger.error(f"Unexpected error during database initialization: {e}")
            raise DatabaseConnectionException(
                f"Database initialization failed: {e}",
                context=error_context,
                original_exception=e
            )
    
    def _get_session(self):
        if self.session is None:
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
        return self.session

    def _execute_with_retry(self, operation: Callable, operation_name: str):
        attempt = 0
        while attempt < self.recovery_strategy.max_retries:
            if not self.session:
                error_context = create_error_context(
                    module=__name__,
                    function=operation_name,
                    extra_data={'attempt': attempt}
                )
                raise DatabaseConnectionException(
                    f"Database session not initialized for {operation_name}",
                    context=error_context
                )
            try:
                result = operation()
                self.session.commit()
                return result
            except (SQLAlchemyError, OperationalError) as e:
                attempt += 1
                self.session.rollback()
                if not self.recovery_strategy.should_retry(e, attempt):
                    error_context = create_error_context(
                        module=__name__,
                        function=operation_name,
                        extra_data={'attempt': attempt}
                    )
                    raise DatabaseException(
                        f"Failed to execute {operation_name} after {attempt} attempts: {e}",
                        context=error_context,
                        original_exception=e
                    )
                time.sleep(self.recovery_strategy.get_delay(attempt))
            except Exception as e:
                self.session.rollback()
                error_context = create_error_context(
                    module=__name__,
                    function=operation_name,
                    extra_data={'attempt': attempt}
                )
                raise DatabaseException(
                    f"Unexpected error in {operation_name}: {e}",
                    context=error_context,
                    original_exception=e
                )
        raise DatabaseException(
            f"Max retries reached for {operation_name}",
            context=create_error_context(module=__name__, function=operation_name)
        )

    def _safe_transaction(self, operation: Callable, operation_type: str, table: str):
        def wrapper(*args, **kwargs):
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            try:
                result = operation(*args, **kwargs)
                self.session.commit()
                return result
            except IntegrityError as e:
                self.session.rollback()
                error_context = create_error_context(
                    module=__name__,
                    function=operation.__name__,
                    extra_data={'table': table}
                )
                raise DatabaseIntegrityException(
                    f"Integrity error in {operation_type} on {table}: {e}",
                    constraint=str(e),
                    context=error_context,
                    original_exception=e
                )
            except SQLAlchemyError as e:
                self.session.rollback()
                error_context = create_error_context(
                    module=__name__,
                    function=operation.__name__,
                    extra_data={'table': table}
                )
                raise DatabaseTransactionException(
                    f"Transaction error in {operation_type} on {table}: {e}",
                    operation=operation_type,
                    table=table,
                    context=error_context,
                    original_exception=e
                )
            except Exception as e:
                self.session.rollback()
                error_context = create_error_context(
                    module=__name__,
                    function=operation.__name__,
                    extra_data={'table': table}
                )
                raise DatabaseException(
                    f"Unexpected error in {operation_type} on {table}: {e}",
                    context=error_context,
                    original_exception=e
                )
        return wrapper

    def add_signal(self, signal: Signal) -> bool:
        def _add_signal():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            signal_model = SignalModel(
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
                created_at=signal.created_at or datetime.now(timezone.utc)
            )
            self.session.add(signal_model)
            return True

        try:
            result = self._safe_transaction(_add_signal, operation_type="INSERT", table="signals")()
            logger.info(f"Signal added for {signal.symbol}")
            return result
        except DatabaseException:
            logger.error(f"Failed to add signal for {signal.symbol}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding signal for {signal.symbol}: {e}")
            return False
    
    def add_trade(self, trade: Dict) -> bool:
        if not self.session:
            raise DatabaseConnectionException("Database session not initialized")

        # 🔑 Force conversion BEFORE wrapper
        if isinstance(trade.get("timestamp"), str):
            trade["timestamp"] = parser.isoparse(trade["timestamp"])
        elif trade.get("timestamp") is None:
            trade["timestamp"] = datetime.now(timezone.utc)

        if isinstance(trade.get("closed_at"), str):
            trade["closed_at"] = parser.isoparse(trade["closed_at"])

        def _add_trade():
            trade_model = TradeModel(
                symbol=trade['symbol'],
                side=trade['side'],
                qty=trade['qty'],
                entry_price=trade['entry_price'],
                order_id=trade['order_id'],
                virtual=trade.get('virtual', True),
                status=trade.get('status', 'open'),
                exit_price=trade.get('exit_price'),
                pnl=trade.get('pnl'),
                score=trade.get('score'),
                strategy=trade.get('strategy', 'Manual'),
                leverage=trade.get('leverage', 10),
                timestamp=trade['timestamp'],     # ✅ already datetime
                closed_at=trade.get('closed_at')  # ✅ datetime or None
            )    
            self._get_session().add(trade_model)
            return True

        try:
            result = self._safe_transaction(_add_trade, operation_type="INSERT", table="trades")()
            logger.info(f"Trade added for {trade['symbol']}")
            return result
        except DatabaseException:
            logger.error(f"Failed to add trade for {trade['symbol']}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding trade for {trade['symbol']}: {e}")
            return False
    

    def update_wallet_balance(self, wallet_balance: WalletBalance) -> bool:
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

        try:
            result = self._safe_transaction(_update_wallet_balance, operation_type="UPSERT", table="wallet_balances")()
            logger.info(f"Wallet balance updated for {wallet_balance.trading_mode}")
            return result
        except DatabaseException:
            logger.error(f"Failed to update wallet balance for {wallet_balance.trading_mode}")
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

    def get_open_trades(self) -> List[Trade]:
        """Retrieve all open trades from the database."""
        try:
            def _get_open_trades():
                if not self.session:
                    raise DatabaseConnectionException("Database session not initialized")
                trades = self.session.query(TradeModel).filter(
                    TradeModel.status == "open"
                ).order_by(TradeModel.timestamp.desc()).all()
                return [t.to_trade() for t in trades]
            return self._execute_with_retry(_get_open_trades, "get_open_trades")
        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
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

            # Migrate virtual balance
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

            # Migrate real balance  
            if "real" in capital_data:
                r = capital_data["real"]
                real_balance = WalletBalance(
                    trading_mode="real",
                    capital=float(r.get("capital", 0.0)),
                    available=float(r.get("available", 0.0)),
                    used=float(r.get("used", 0.0)),
                    start_balance=float(r.get("start_balance", 0.0)),
                    currency=r.get("currency", "USDT"),
                    updated_at=datetime.now(timezone.utc),
                )
                self.update_wallet_balance(real_balance)
                logger.info("Real balance migrated to database")

            logger.info("Capital.json data successfully migrated to database")
            return True

        except Exception as e:
            logger.error(f"Error migrating capital.json to database: {e}")
            return False

    def get_all_wallet_balances(self) -> Dict[str, WalletBalance]:
        """Get all wallet balances as a dictionary"""
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
        """Save a setting to the database"""
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
            result = self._safe_transaction(_save_setting_operation, operation_type="UPSERT", table="settings")()
            logger.info(f"Saved setting {key} with value {value}")
            return result
        except DatabaseException:
            logger.error(f"Database exception saving setting {key}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving setting {key}: {str(e)}")
            return False

    def get_setting(self, key: str) -> Optional[str]:
        """Retrieve a setting from the database"""
        def _get_setting_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            setting = self.session.query(SettingsModel).filter(SettingsModel.key == key).first()
            return setting.value if setting else None

        try:
            result = self._execute_with_retry(_get_setting_operation, f"get_setting_{key}")
            logger.info(f"Retrieved setting {key}: {result}")
            return result
        except DatabaseException:
            logger.error(f"Failed to get setting {key}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting setting {key}: {str(e)}")
            return None

    def close(self):
        """Close database connection"""
        try:
            if self.session:
                self.session.close()
            if self.engine:
                self.engine.dispose()
            logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Failed to close database connection: {str(e)}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get database connection statistics"""
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

# Global database manager instance
db_manager = DatabaseManager()