import os
import json
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy import create_engine, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
import uuid
import requests
import socket
import hashlib
from logging_config import get_logger
from exceptions import (
    DatabaseConnectionException,
    DatabaseIntegrityException,
    DatabaseTransactionException,
    DatabaseException,
    DatabaseErrorRecoveryStrategy,
    create_error_context
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
    leverage: int = 10
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
    leverage: Mapped[int] = mapped_column(Integer, default=10)
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
    leverage: Mapped[int] = mapped_column(Integer, default=10)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_trade(self) -> Trade:
        return Trade(
            id=str(self.id) if self.id else None,
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
            score=self.score,
            strategy=self.strategy,
            leverage=self.leverage,
            trail=self.trail,
            liquidation=self.liquidation,
            margin_usdt=self.margin_usdt,
            timestamp=self.timestamp,
            closed_at=self.closed_at
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
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

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
    __table_args__ = (Index('idx_settings_key', 'key'),)

    key = mapped_column(String(255), primary_key=True)
    value = mapped_column(Text)
    updated_at = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class LicenseModel(Base):
    __tablename__ = 'licenses'
    __table_args__ = (Index('idx_license_key', 'license_key'),)

    id = mapped_column(Integer, primary_key=True)
    license_key = mapped_column(String(255), unique=True, nullable=False)
    user_email = mapped_column(String(255))
    tier = mapped_column(String(50), default='basic')
    duration_days = mapped_column(Integer, default=30)
    machine_hash = mapped_column(String(255), nullable=True)
    created_at = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    expiration_date = mapped_column(DateTime, nullable=False)
    is_active = mapped_column(Boolean, default=True)
    notes = mapped_column(Text)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "license_key": self.license_key,
            "user_email": self.user_email,
            "tier": self.tier,
            "duration_days": self.duration_days,
            "machine_hash": self.machine_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "is_active": self.is_active,
            "notes": self.notes
        }

class LicenseLogModel(Base):
    __tablename__ = 'license_logs'
    __table_args__ = (Index('idx_license_log_key', 'license_key'), Index('idx_license_log_time', 'event_time'))

    id = mapped_column(Integer, primary_key=True)
    license_key = mapped_column(String(255))
    event_type = mapped_column(String(50))
    event_time = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    hostname = mapped_column(String(255))
    mac_address = mapped_column(String(255))
    ip_address = mapped_column(String(128))
    geo = mapped_column(JSONB)
    message = mapped_column(Text)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "license_key": self.license_key,
            "event_type": self.event_type,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "hostname": self.hostname,
            "mac_address": self.mac_address,
            "ip_address": self.ip_address,
            "geo": self.geo,
            "message": self.message
        }

class DatabaseManager:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL not set in environment")
        
        self.engine = create_engine(self.db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session: Optional[Session] = None
        self.max_retries = 3
        self.recovery_strategy = DatabaseErrorRecoveryStrategy()
        
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created/verified successfully")
            self.session = self.SessionLocal()
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise DatabaseConnectionException(f"Database initialization failed: {str(e)}")

    def _safe_transaction(self, operation: Callable, operation_type: str = "UNKNOWN", table: str = "UNKNOWN") -> Callable:
        def wrapper(*args, **kwargs):
            if not self.session:
                context = create_error_context(module=__name__, function='_safe_transaction', line_number=None, extra_data={'operation_type': operation_type, 'table': table})
                raise DatabaseConnectionException(
                    message="Database session not initialized",
                    context=context
                )
            context = create_error_context(
                module=__name__,
                function=operation.__name__,
                line_number=None,
                extra_data={'operation_type': operation_type, 'table': table, 'args': str(args), 'kwargs': str(kwargs)}
            )
            try:
                with self.session.begin():
                    result = operation(*args, **kwargs)
                self.session.commit()
                return result
            except IntegrityError as e:
                self.session.rollback()
                logger.error(f"Integrity error in {operation_type} on {table}: {str(e)}", extra=asdict(context))
                raise DatabaseIntegrityException(
                    message=f"Integrity violation: {str(e)}",
                    context=context,
                    original_exception=e
                )
            except OperationalError as e:
                self.session.rollback()
                logger.error(f"Operational error in {operation_type} on {table}: {str(e)}", extra=asdict(context))
                raise DatabaseConnectionException(
                    message=f"Database connection error: {str(e)}",
                    context=context,
                    original_exception=e
                )
            except SQLAlchemyError as e:
                self.session.rollback()
                logger.error(f"SQLAlchemy error in {operation_type} on {table}: {str(e)}", extra=asdict(context))
                raise DatabaseTransactionException(
                    message=f"Transaction failed: {str(e)}",
                    operation=operation_type,
                    table=table,
                    context=context,
                    original_exception=e
                )
            except Exception as e:
                self.session.rollback()
                logger.error(f"Unexpected error in {operation_type} on {table}: {str(e)}", extra=asdict(context))
                raise DatabaseException(
                    message=f"Unexpected database error: {str(e)}",
                    context=context,
                    original_exception=e
                )
        
        return wrapper

    def _execute_with_retry(self, operation: Callable, operation_name: str) -> Any:
        for attempt in range(self.max_retries):
            try:
                return operation()
            except (DatabaseConnectionException, OperationalError) as e:
                if attempt == self.max_retries - 1:
                    raise
                delay = self.recovery_strategy.get_delay(attempt)
                logger.warning(f"Retry attempt {attempt + 1}/{self.max_retries} for {operation_name}: {str(e)}. Retrying in {delay}s")
                time.sleep(delay)
                if self.session:
                    self.session.close()
                self.session = self.SessionLocal()

    # Signal methods
    def save_signal(self, signal: Signal) -> bool:
        def _save_signal_operation():
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

        try:
            result = self._safe_transaction(_save_signal_operation, operation_type="INSERT", table="signals")()
            logger.info(f"Saved signal for {signal.symbol} with ID {signal.id}")
            return result
        except DatabaseException:
            logger.error(f"Database exception saving signal for {signal.symbol}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving signal: {str(e)}")
            return False

    def get_signals(self, limit: int = 100, offset: int = 0) -> List[Signal]:
        def _get_signals_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            models = self.session.query(SignalModel).order_by(SignalModel.created_at.desc()).limit(limit).offset(offset).all()
            return [model.to_signal() for model in models]

        try:
            return self._execute_with_retry(_get_signals_operation, "get_signals")
        except DatabaseException:
            logger.error("Failed to get signals")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting signals: {str(e)}")
            return []

    def get_signal_by_id(self, signal_id: str) -> Optional[Signal]:
        def _get_signal_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            model = self.session.query(SignalModel).filter(SignalModel.id == uuid.UUID(signal_id)).first()
            return model.to_signal() if model else None

        try:
            return self._execute_with_retry(_get_signal_operation, f"get_signal_{signal_id}")
        except DatabaseException:
            logger.error(f"Failed to get signal {signal_id}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting signal {signal_id}: {str(e)}")
            return None

    # Trade methods
    def save_trade(self, trade: Trade) -> bool:
        def _save_trade_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            trade_model = TradeModel(
                symbol=trade.symbol,
                side=trade.side,
                qty=trade.qty,
                entry_price=trade.entry_price,
                order_id=trade.order_id,
                virtual=trade.virtual,
                status=trade.status,
                exit_price=trade.exit_price,
                sl=trade.sl,
                tp=trade.tp,
                pnl=trade.pnl,
                score=trade.score,
                strategy=trade.strategy,
                leverage=trade.leverage,
                trail=trade.trail,
                liquidation=trade.liquidation,
                margin_usdt=trade.margin_usdt,
                timestamp=trade.timestamp,
                closed_at=trade.closed_at
            )
            self.session.add(trade_model)
            return True

        try:
            result = self._safe_transaction(_save_trade_operation, operation_type="INSERT", table="trades")()
            logger.info(f"Saved trade for {trade.symbol} with order ID {trade.order_id}")
            return result
        except DatabaseException:
            logger.error(f"Database exception saving trade for {trade.symbol}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving trade: {str(e)}")
            return False

    def update_trade(self, trade: Trade) -> bool:
        def _update_trade_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            existing_trade = self.session.query(TradeModel).filter(TradeModel.order_id == trade.order_id).first()
            if existing_trade:
                existing_trade.status = trade.status
                existing_trade.exit_price = trade.exit_price
                existing_trade.pnl = trade.pnl
                existing_trade.closed_at = trade.closed_at
                existing_trade.sl = trade.sl
                existing_trade.tp = trade.tp
                existing_trade.trail = trade.trail
                return True
            return False

        try:
            result = self._safe_transaction(_update_trade_operation, operation_type="UPDATE", table="trades")()
            if result:
                logger.info(f"Updated trade {trade.order_id}")
            else:
                logger.warning(f"Trade {trade.order_id} not found for update")
            return result
        except DatabaseException:
            logger.error(f"Database exception updating trade {trade.order_id}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating trade: {str(e)}")
            return False

    def get_trades(self, virtual: Optional[bool] = None, limit: int = 100, offset: int = 0) -> List[Trade]:
        def _get_trades_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            query = self.session.query(TradeModel).order_by(TradeModel.timestamp.desc())
            if virtual is not None:
                query = query.filter(TradeModel.virtual == virtual)
            models = query.limit(limit).offset(offset).all()
            return [model.to_trade() for model in models]

        try:
            return self._execute_with_retry(_get_trades_operation, "get_trades")
        except DatabaseException:
            logger.error("Failed to get trades")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting trades: {str(e)}")
            return []

    def get_trade_by_order_id(self, order_id: str) -> Optional[Trade]:
        def _get_trade_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            model = self.session.query(TradeModel).filter(TradeModel.order_id == order_id).first()
            return model.to_trade() if model else None

        try:
            return self._execute_with_retry(_get_trade_operation, f"get_trade_{order_id}")
        except DatabaseException:
            logger.error(f"Failed to get trade {order_id}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting trade {order_id}: {str(e)}")
            return None

    # Wallet balance methods
    def update_wallet_balance(self, balance: WalletBalance) -> bool:
        def _update_wallet_balance_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            existing_balance = self.session.query(WalletBalanceModel).filter(
                WalletBalanceModel.trading_mode == balance.trading_mode
            ).first()
            
            if existing_balance:
                existing_balance.capital = balance.capital
                existing_balance.available = balance.available
                existing_balance.used = balance.used
                existing_balance.start_balance = balance.start_balance
                existing_balance.currency = balance.currency
                existing_balance.updated_at = datetime.now(timezone.utc)
            else:
                new_balance = WalletBalanceModel(
                    trading_mode=balance.trading_mode,
                    capital=balance.capital,
                    available=balance.available,
                    used=balance.used,
                    start_balance=balance.start_balance,
                    currency=balance.currency,
                    updated_at=datetime.now(timezone.utc)
                )
                self.session.add(new_balance)
            return True

        try:
            result = self._safe_transaction(_update_wallet_balance_operation, operation_type="UPSERT", table="wallet_balances")()
            logger.info(f"Updated wallet balance for {balance.trading_mode}")
            return result
        except DatabaseException:
            logger.error(f"Database exception updating wallet balance for {balance.trading_mode}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating wallet balance: {str(e)}")
            return False

    def get_wallet_balance(self, trading_mode: str) -> Optional[WalletBalance]:
        def _get_wallet_balance():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            model = self.session.query(WalletBalanceModel).filter(
                WalletBalanceModel.trading_mode == trading_mode
            ).first()
            return model.to_wallet_balance() if model else None

        try:
            return self._execute_with_retry(_get_wallet_balance, f"get_wallet_balance_{trading_mode}")
        except DatabaseException:
            logger.error(f"Failed to get wallet balance for {trading_mode}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting wallet balance {trading_mode}: {str(e)}")
            return None

    # License helper functions
    def get_mac(self):
        mac = uuid.getnode()
        return ':'.join(('%012X' % mac)[i:i+2] for i in range(0, 12, 2))

    def machine_hash(self, hostname: str, mac: str) -> str:
        return hashlib.sha256(f"{hostname}|{mac}".encode("utf-8")).hexdigest()

    def get_ip_and_geo(self):
        if hasattr(self, '_geo_cache') and self._geo_cache['timestamp'] > datetime.now(timezone.utc) - timedelta(minutes=60):
            return self._geo_cache['ip'], self._geo_cache['geo']
        ip, geo = None, {}
        try:
            r = requests.get("https://api.ipify.org?format=json", timeout=5)
            ip = r.json().get("ip")
            ipinfo_token = os.getenv("IPINFO_TOKEN", "")
            if ipinfo_token:
                rq = requests.get(f"https://ipinfo.io/{ip}/json?token={ipinfo_token}", timeout=5)
                geo = rq.json()
            else:
                rq = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
                d = rq.json()
                geo = {"ip": ip, "city": d.get("city"), "region": d.get("regionName"),
                       "country": d.get("country"), "lat": d.get("lat"), "lon": d.get("lon")}
        except:
            geo = {}
        self._geo_cache = {'ip': ip, 'geo': geo, 'timestamp': datetime.now(timezone.utc)}
        return ip, geo

    def log_event(self, license_key, event_type, hostname=None, mac=None, ip=None, geo=None, message=None):
        def _log_event_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            log = LicenseLogModel(
                license_key=license_key,
                event_type=event_type,
                hostname=hostname,
                mac_address=mac,
                ip_address=ip,
                geo=geo,
                message=message,
                event_time=datetime.now(timezone.utc)
            )
            self.session.add(log)
            return True

        try:
            result = self._safe_transaction(_log_event_operation, operation_type="INSERT", table="license_logs")()
            logger.info(f"Logged event {event_type} for license {license_key}")
            return result
        except DatabaseException:
            logger.error(f"Database exception logging event for {license_key}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error logging event: {str(e)}")
            return False

    # License operations
    def get_license_info(self, key):
        def _get_license_info_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            model = self.session.query(LicenseModel).filter(LicenseModel.license_key == key).first()
            return model.to_dict() if model else None

        try:
            return self._execute_with_retry(_get_license_info_operation, f"get_license_info_{key}")
        except DatabaseException:
            logger.error(f"Failed to get license info for {key}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting license info {key}: {str(e)}")
            return None

    def validate_license(self, key: str, hostname: Optional[str] = None, mac: Optional[str] = None):
        def _validate_license_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            info = self.session.query(LicenseModel).filter(LicenseModel.license_key == key).first()
            hostname_val: str = hostname or socket.gethostname() or ""
            mac_val: str = mac or self.get_mac() or ""
            ip, geo = self.get_ip_and_geo()
            valid = False
            message = ""

            if not info:
                message = "License not found."
                self.log_event(key, "validate_fail", hostname_val, mac_val, ip, geo, message)
                return valid, message, None

            if not info.is_active:
                message = "License is inactive."
                self.log_event(key, "validate_fail_inactive", hostname_val, mac_val, ip, geo, message)
                return valid, message, info.to_dict()

            if datetime.now(timezone.utc) > info.expiration_date:
                message = "License expired."
                self.log_event(key, "validate_fail_expired", hostname_val, mac_val, ip, geo, message)
                return valid, message, info.to_dict()

            if info.machine_hash:
                actual_hash = self.machine_hash(hostname_val, mac_val)
                if actual_hash != info.machine_hash:
                    message = "License bound to a different machine."
                    self.log_event(key, "validate_fail_machine", hostname_val, mac_val, ip, geo, message)
                    return valid, message, info.to_dict()

            valid = True
            message = "License valid."
            self.log_event(key, "validate_success", hostname_val, mac_val, ip, geo, message)
            return valid, message, info.to_dict()

        try:
            return self._execute_with_retry(_validate_license_operation, f"validate_license_{key}")
        except DatabaseException:
            logger.error(f"Failed to validate license {key}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error validating license {key}: {str(e)}")
            return False, str(e), None

    def list_logs(self, limit=500):
        def _list_logs_operation():
            if not self.session:
                raise DatabaseConnectionException("Database session not initialized")
            models = self.session.query(LicenseLogModel).order_by(LicenseLogModel.event_time.desc()).limit(limit).all()
            return [model.to_dict() for model in models]

        try:
            return self._execute_with_retry(_list_logs_operation, "list_logs")
        except DatabaseException:
            logger.error("Failed to list logs")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing logs: {str(e)}")
            return []

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
            logger.error(f"Error getting connection stats: {str(e)}")
            
        return stats

# Global database manager instance
db_manager = DatabaseManager()