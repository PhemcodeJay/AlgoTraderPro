"""
Custom exception classes for AlgoTrader Pro
Provides structured error handling with proper exception types
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import traceback
from datetime import datetime
import time
from logging_config import get_logger

logger = get_logger(__name__, structured_format=True)

@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: datetime
    module: str
    function: str
    line_number: Optional[int] = None
    extra_data: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None

class AlgoTraderBaseException(Exception):
    """Base exception class for all AlgoTrader Pro exceptions"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[ErrorContext] = None, 
                 original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext(
            timestamp=datetime.now(),
            module="unknown",
            function="unknown"
        )
        self.original_exception = original_exception
        
        # Capture stack trace if not provided
        if not self.context.stack_trace:
            self.context.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging"""
        return {
            'exception_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'timestamp': self.context.timestamp.isoformat(),
            'module': self.context.module,
            'function': self.context.function,
            'line_number': self.context.line_number,
            'extra_data': self.context.extra_data,
            'original_exception': str(self.original_exception) if self.original_exception else None
        }

# API-related exceptions
class APIException(AlgoTraderBaseException):
    """Base class for API-related exceptions"""
    pass

class APIConnectionException(APIException):
    """Raised when API connection fails"""
    
    def __init__(self, message: str, endpoint: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.endpoint = endpoint
        self.status_code = status_code

class APIRateLimitException(APIException):
    """Raised when API rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after

class APIAuthenticationException(APIException):
    """Raised when API authentication fails"""
    pass

class APITimeoutException(APIException):
    """Raised when API request times out"""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration

class APIDataException(APIException):
    """Raised when API returns invalid or unexpected data"""
    
    def __init__(self, message: str, response_data: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.response_data = response_data

# Database-related exceptions
class DatabaseException(AlgoTraderBaseException):
    """Base class for database-related exceptions"""
    pass

class DatabaseConnectionException(DatabaseException):
    """Raised when database connection fails"""
    pass

class DatabaseTransactionException(DatabaseException):
    """Raised when database transaction fails"""
    
    def __init__(self, message: str, operation: str, table: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.table = table

class DatabaseIntegrityException(DatabaseException):
    """Raised when database integrity constraint is violated"""
    
    def __init__(self, message: str, constraint: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.constraint = constraint

class DatabaseMigrationException(DatabaseException):
    """Raised when database migration fails"""
    pass

# Trading-related exceptions
class TradingException(AlgoTraderBaseException):
    """Base class for trading-related exceptions"""
    pass

class InsufficientBalanceException(TradingException):
    """Raised when account has insufficient balance for trade"""
    
    def __init__(self, message: str, required_amount: float, available_amount: float, 
                 currency: str = "USDT", **kwargs):
        super().__init__(message, **kwargs)
        self.required_amount = required_amount
        self.available_amount = available_amount
        self.currency = currency

class InvalidPositionSizeException(TradingException):
    """Raised when position size is invalid"""
    
    def __init__(self, message: str, position_size: float, min_size: Optional[float] = None,
                 max_size: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.position_size = position_size
        self.min_size = min_size
        self.max_size = max_size

class RiskManagementException(TradingException):
    """Raised when trade violates risk management rules"""
    
    def __init__(self, message: str, risk_metric: str, current_value: float,
                 limit_value: float, **kwargs):
        super().__init__(message, **kwargs)
        self.risk_metric = risk_metric
        self.current_value = current_value
        self.limit_value = limit_value

class OrderExecutionException(TradingException):
    """Raised when order execution fails"""
    
    def __init__(self, message: str, order_id: Optional[str] = None, 
                 symbol: Optional[str] = None, side: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.order_id = order_id
        self.symbol = symbol
        self.side = side

class PositionManagementException(TradingException):
    """Raised when position management operation fails"""
    
    def __init__(self, message: str, position_id: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.position_id = position_id
        self.operation = operation

# WebSocket-related exceptions
class WebSocketException(AlgoTraderBaseException):
    """Base class for WebSocket-related exceptions"""
    pass

class WebSocketConnectionException(WebSocketException):
    """Raised when WebSocket connection fails"""
    
    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.url = url

class WebSocketAuthenticationException(WebSocketException):
    """Raised when WebSocket authentication fails"""
    pass

class WebSocketDataException(WebSocketException):
    """Raised when WebSocket receives invalid data"""
    
    def __init__(self, message: str, data: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.data = data

# Signal generation exceptions
class SignalException(AlgoTraderBaseException):
    """Base class for signal generation exceptions"""
    pass

class IndicatorCalculationException(SignalException):
    """Raised when technical indicator calculation fails"""
    
    def __init__(self, message: str, indicator: str, symbol: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.indicator = indicator
        self.symbol = symbol

class InsufficientDataException(SignalException):
    """Raised when insufficient market data is available"""
    
    def __init__(self, message: str, symbol: str, required_periods: int,
                 available_periods: int, **kwargs):
        super().__init__(message, **kwargs)
        self.symbol = symbol
        self.required_periods = required_periods
        self.available_periods = available_periods

class SignalValidationException(SignalException):
    """Raised when signal validation fails"""
    
    def __init__(self, message: str, signal_data: Optional[Dict] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.signal_data = signal_data

# Configuration exceptions
class ConfigurationException(AlgoTraderBaseException):
    """Base class for configuration-related exceptions"""
    pass

class InvalidConfigurationException(ConfigurationException):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_value: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_value = config_value

class MissingConfigurationException(ConfigurationException):
    """Raised when required configuration is missing"""
    
    def __init__(self, message: str, missing_keys: Optional[list] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.missing_keys = missing_keys or []

# Machine Learning exceptions
class MLException(AlgoTraderBaseException):
    """Base class for machine learning exceptions"""
    pass

class ModelLoadException(MLException):
    """Raised when ML model fails to load"""
    
    def __init__(self, message: str, model_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model_path = model_path

class ModelPredictionException(MLException):
    """Raised when ML model prediction fails"""
    
    def __init__(self, message: str, features: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.features = features

class FeaturePreparationException(MLException):
    """Raised when feature preparation fails"""
    
    def __init__(self, message: str, feature_names: Optional[list] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.feature_names = feature_names or []

# Notification exceptions
class NotificationException(AlgoTraderBaseException):
    """Base class for notification exceptions"""
    pass

class DiscordNotificationException(NotificationException):
    """Raised when Discord notification fails"""
    pass

class TelegramNotificationException(NotificationException):
    """Raised when Telegram notification fails"""
    pass

class PDFGenerationException(NotificationException):
    """Raised when PDF report generation fails"""
    
    def __init__(self, message: str, report_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.report_type = report_type

# Utility functions for error handling
def create_error_context(module: str, function: str, 
                        line_number: Optional[int] = None,
                        extra_data: Optional[Dict[str, Any]] = None) -> ErrorContext:
    """Create error context for exceptions"""
    return ErrorContext(
        timestamp=datetime.now(),
        module=module,
        function=function,
        line_number=line_number,
        extra_data=extra_data,
        stack_trace=traceback.format_exc()
    )

def handle_exception(exception: Exception, logger, 
                    default_message: str = "An unexpected error occurred") -> AlgoTraderBaseException:
    """Convert generic exceptions to AlgoTrader exceptions"""
    
    if isinstance(exception, AlgoTraderBaseException):
        return exception
    
    # Create context from current frame
    import inspect
    current_frame = inspect.currentframe()
    frame = current_frame.f_back if current_frame else None
    
    if frame:
        context = create_error_context(
            module=frame.f_globals.get('__name__', 'unknown') if frame.f_globals else 'unknown',
            function=frame.f_code.co_name if frame.f_code else 'unknown',
            line_number=frame.f_lineno if hasattr(frame, 'f_lineno') else None
        )
    else:
        context = create_error_context(
            module='unknown',
            function='unknown',
            line_number=None
        )
    
    # Log the original exception
    logger.error(f"Converting generic exception: {type(exception).__name__}: {str(exception)}")
    
    # Return wrapped exception
    return AlgoTraderBaseException(
        message=f"{default_message}: {str(exception)}",
        context=context,
        original_exception=exception
    )

def safe_execute(func, logger, default_return=None, 
                exception_type=AlgoTraderBaseException):
    """Safely execute a function with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except exception_type as e:
            logger.error(f"Expected exception in {func.__name__}: {e.to_dict()}")
            raise
        except Exception as e:
            wrapped_exception = handle_exception(e, logger, 
                f"Error in {func.__name__}")
            logger.error(f"Unexpected exception in {func.__name__}: {wrapped_exception.to_dict()}")
            
            if default_return is not None:
                return default_return
            raise wrapped_exception
    
    return wrapper

# Error recovery strategies
class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried"""
        return attempt < self.max_retries
    
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry"""
        return self.delay * (2 ** attempt)  # Exponential backoff

class APIErrorRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for API errors"""
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        if not super().should_retry(exception, attempt):
            return False
        
        # Don't retry authentication errors
        if isinstance(exception, APIAuthenticationException):
            return False
        
        # Retry rate limit errors with longer delay
        if isinstance(exception, APIRateLimitException):
            return attempt < 2
        
        # Retry connection and timeout errors
        return isinstance(exception, (APIConnectionException, APITimeoutException))

class DatabaseErrorRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for database errors"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        super().__init__(max_retries, delay)
        self.logger = get_logger(__name__, structured_format=True)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        if not super().should_retry(exception, attempt):
            return False
        
        # Don't retry integrity constraint violations
        if isinstance(exception, DatabaseIntegrityException):
            return False
        
        # Retry connection and transaction errors
        return isinstance(exception, (DatabaseConnectionException, DatabaseTransactionException))
    
    def execute_with_retry(self, operation: Callable, operation_name: str) -> Any:
        """Execute operation with retry logic"""
        attempt = 0
        last_exception = None

        while attempt < self.max_retries:
            try:
                return operation()
            except (DatabaseConnectionException, DatabaseTransactionException) as e:
                if not self.should_retry(e, attempt):
                    raise
                last_exception = e
                delay = self.get_delay(attempt)
                self.logger.warning(
                    f"Retry {attempt + 1}/{self.max_retries} for {operation_name} "
                    f"after {delay:.2f}s due to {str(e)}"
                )
                time.sleep(delay)
                attempt += 1
            except Exception as e:
                raise handle_exception(e, self.logger, f"Unexpected error in {operation_name}")

        raise DatabaseException(
            f"Failed to execute {operation_name} after {self.max_retries} retries: {str(last_exception)}",
            context=create_error_context(
                module=__name__,
                function="execute_with_retry",
                extra_data={"operation_name": operation_name, "attempts": attempt}
            ),
            original_exception=last_exception
        )
