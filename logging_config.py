"""
Centralized logging configuration for AlgoTrader Pro
Provides structured logging with proper log levels, rotation, and formatting
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output"""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = getattr(record, 'extra_data')
            
        # Add trading-specific context if available
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = getattr(record, 'symbol')
        if hasattr(record, 'trade_id'):
            log_entry['trade_id'] = getattr(record, 'trade_id')
        if hasattr(record, 'order_id'):
            log_entry['order_id'] = getattr(record, 'order_id')
        if hasattr(record, 'trading_mode'):
            log_entry['trading_mode'] = getattr(record, 'trading_mode')
            
        return json.dumps(log_entry, ensure_ascii=False)

class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        if self.datefmt:
            record.asctime = self.formatTime(record, self.datefmt)
        else:
            record.asctime = self.formatTime(record)
        
        # Format the message with color
        formatted = f"{color}[{record.levelname}]{reset} "
        formatted += f"{record.asctime} - {record.name} - "
        formatted += f"{record.funcName}:{record.lineno} - "
        formatted += f"{record.getMessage()}"
        
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
            
        return formatted

class LoggerConfig:
    """Centralized logger configuration and management"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.loggers: Dict[str, logging.Logger] = {}
        
        # Create logs directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        self._setup_root_logger()
        
    def _setup_root_logger(self):
        """Setup the root logger with basic configuration"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def get_logger(self, name: str, 
                   console_output: bool = True,
                   file_output: bool = True,
                   structured_format: bool = False) -> logging.Logger:
        """
        Get or create a logger with specified configuration
        
        Args:
            name: Logger name (usually module name)
            console_output: Whether to output to console
            file_output: Whether to output to file
            structured_format: Whether to use structured JSON format for files
        """
        if name in self.loggers:
            return self.loggers[name]
            
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_formatter = ColoredConsoleFormatter(
                '%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if file_output:
            log_file = self.log_dir / f"{name.replace('.', '_')}.log"
            
            # Rotating file handler (10MB max, keep 5 files)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            
            if structured_format:
                file_formatter = StructuredFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Store logger reference
        self.loggers[name] = logger
        return logger
    
    def get_trading_logger(self, component: str) -> logging.Logger:
        """Get a specialized logger for trading components"""
        return self.get_logger(
            f"trading.{component}",
            console_output=True,
            file_output=True,
            structured_format=True
        )
    
    def log_trade_event(self, logger: logging.Logger, level: str, message: str,
                       symbol: Optional[str] = None,
                       trade_id: Optional[str] = None,
                       order_id: Optional[str] = None,
                       trading_mode: Optional[str] = None,
                       extra_data: Optional[Dict[str, Any]] = None):
        """Log trading-specific events with context"""
        
        # Create log record
        record = logger.makeRecord(
            logger.name,
            getattr(logging, level.upper()),
            __file__,
            0,
            message,
            (),
            None
        )
        
        # Add trading context
        if symbol:
            record.symbol = symbol
        if trade_id:
            record.trade_id = trade_id
        if order_id:
            record.order_id = order_id
        if trading_mode:
            record.trading_mode = trading_mode
        if extra_data:
            record.extra_data = extra_data
            
        logger.handle(record)
    
    def log_api_request(self, logger: logging.Logger, method: str, endpoint: str,
                       status_code: Optional[int] = None,
                       response_time: Optional[float] = None,
                       error: Optional[str] = None):
        """Log API request with timing and status information"""
        
        extra_data = {
            'api_method': method,
            'api_endpoint': endpoint,
            'status_code': status_code,
            'response_time_ms': response_time
        }
        
        if error:
            self.log_trade_event(
                logger, 'ERROR',
                f"API request failed: {method} {endpoint} - {error}",
                extra_data=extra_data
            )
        else:
            self.log_trade_event(
                logger, 'INFO',
                f"API request: {method} {endpoint} - {status_code} ({response_time:.2f}ms)",
                extra_data=extra_data
            )
    
    def log_database_operation(self, logger: logging.Logger, operation: str,
                             table: str, success: bool,
                             record_count: Optional[int] = None,
                             error: Optional[str] = None):
        """Log database operations with context"""
        
        extra_data = {
            'db_operation': operation,
            'db_table': table,
            'record_count': record_count,
            'success': success
        }
        
        if success:
            message = f"DB {operation} on {table}"
            if record_count is not None:
                message += f" ({record_count} records)"
            self.log_trade_event(logger, 'INFO', message, extra_data=extra_data)
        else:
            self.log_trade_event(
                logger, 'ERROR',
                f"DB {operation} failed on {table}: {error}",
                extra_data=extra_data
            )
    
    def configure_uvicorn_logging(self):
        """Configure uvicorn/ASGI logging if needed"""
        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_logger.setLevel(self.log_level)
        
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.setLevel(self.log_level)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics and configuration info"""
        stats = {
            'log_level': logging.getLevelName(self.log_level),
            'log_directory': str(self.log_dir),
            'active_loggers': list(self.loggers.keys()),
            'log_files': []
        }
        
        # Get log file information
        for log_file in self.log_dir.glob("*.log"):
            try:
                file_stat = log_file.stat()
                stats['log_files'].append({
                    'name': log_file.name,
                    'size_mb': round(file_stat.st_size / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
            except Exception as e:
                stats['log_files'].append({
                    'name': log_file.name,
                    'error': str(e)
                })
        
        return stats

# Global logger configuration instance
_logger_config = None

def initialize_logging(log_dir: str = "logs", log_level: str = "INFO") -> LoggerConfig:
    """Initialize the global logging configuration"""
    global _logger_config
    _logger_config = LoggerConfig(log_dir, log_level)
    return _logger_config

def get_logger(name: str, **kwargs) -> logging.Logger:
    """Get a logger instance using global configuration"""
    global _logger_config
    if _logger_config is None:
        _logger_config = initialize_logging()
    return _logger_config.get_logger(name, **kwargs)

def get_trading_logger(component: str) -> logging.Logger:
    """Get a trading-specific logger"""
    global _logger_config
    if _logger_config is None:
        _logger_config = initialize_logging()
    return _logger_config.get_trading_logger(component)

def log_trade_event(logger: logging.Logger, level: str, message: str, **kwargs):
    """Log a trading event with context"""
    global _logger_config
    if _logger_config is None:
        _logger_config = initialize_logging()
    _logger_config.log_trade_event(logger, level, message, **kwargs)

def log_api_request(logger: logging.Logger, method: str, endpoint: str, **kwargs):
    """Log an API request with timing information"""
    global _logger_config
    if _logger_config is None:
        _logger_config = initialize_logging()
    _logger_config.log_api_request(logger, method, endpoint, **kwargs)

def log_database_operation(logger: logging.Logger, operation: str, table: str, **kwargs):
    """Log a database operation"""
    global _logger_config
    if _logger_config is None:
        _logger_config = initialize_logging()
    _logger_config.log_database_operation(logger, operation, table, **kwargs)

def get_log_stats() -> Dict[str, Any]:
    """Get logging statistics"""
    global _logger_config
    if _logger_config is None:
        _logger_config = initialize_logging()
    return _logger_config.get_log_stats()

# Initialize logging on import
initialize_logging()