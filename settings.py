import json
import logging
import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse

# Configure logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

def load_settings() -> Dict[str, Any]:
    default_settings = {
        "SCAN_INTERVAL": float(os.getenv("DEFAULT_SCAN_INTERVAL", 3600)),
        "TOP_N_SIGNALS": int(os.getenv("DEFAULT_TOP_N_SIGNALS", 5)),
        "MAX_LOSS_PCT": -15.0,
        "TP_PERCENT": 25.0,
        "SL_PERCENT": 5.0,
        "MAX_DRAWDOWN_PCT": -15.0,
        "LEVERAGE": float(os.getenv("LEVERAGE", 10)),
        "RISK_PCT": float(os.getenv("RISK_PCT", 0.01)),
        "VIRTUAL_BALANCE": 100.0,
        "ENTRY_BUFFER_PCT": float(os.getenv("ENTRY_BUFFER_PCT", 0.002)),
        "SYMBOLS": ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT"],
        "USE_WEBSOCKET": True,
        "MAX_POSITIONS": 5,
        "MIN_SIGNAL_SCORE": 60.0,
        "AUTO_TRADING_ENABLED": True,
        "NOTIFICATION_ENABLED": True,
        "RSI_OVERSOLD": 30,
        "RSI_OVERBOUGHT": 70,
        "MIN_VOLUME": 1000000,
        "MIN_ATR_PCT": 0.5,
        "MAX_SPREAD_PCT": 0.1,
        "LICENSE_KEY": None,
        "USE_LOCAL_VALIDATION": True,
        "RENDER_SERVER_URL": "http://localhost:8000"
    }

    try:
        if not os.path.exists("settings.json"):
            logger.warning("settings.json not found, creating with default settings")
            with open("settings.json", "w") as f:
                json.dump(default_settings, f, indent=2)
            return default_settings

        with open("settings.json", "r") as f:
            settings = json.load(f)

        # Merge with defaults for any missing keys
        for key, value in default_settings.items():
            if key not in settings:
                logger.warning(f"Missing {key} in settings.json, using default: {value}")
                settings[key] = value
            else:
                try:
                    # Validate numeric settings
                    if key in ["SCAN_INTERVAL", "LEVERAGE", "RISK_PCT", "VIRTUAL_BALANCE", "ENTRY_BUFFER_PCT", "MIN_SIGNAL_SCORE", "MIN_VOLUME", "MIN_ATR_PCT", "MAX_SPREAD_PCT"]:
                        settings[key] = float(settings[key])
                        if settings[key] <= 0:
                            logger.warning(f"Invalid {key} value {settings[key]}, using default: {value}")
                            settings[key] = value
                    if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"]:
                        settings[key] = float(settings[key])
                        if settings[key] >= 0:
                            logger.warning(f"Invalid {key} value {settings[key]}, using default: {value}")
                            settings[key] = value
                    if key in ["TOP_N_SIGNALS", "MAX_POSITIONS", "RSI_OVERSOLD", "RSI_OVERBOUGHT"]:
                        settings[key] = int(settings[key])
                        if settings[key] <= 0:
                            logger.warning(f"Invalid {key} value {settings[key]}, using default: {value}")
                            settings[key] = value
                    if key in ["USE_WEBSOCKET", "AUTO_TRADING_ENABLED", "NOTIFICATION_ENABLED", "USE_LOCAL_VALIDATION"]:
                        settings[key] = bool(settings[key])
                    if key == "SYMBOLS":
                        if not isinstance(settings[key], list) or not all(isinstance(s, str) for s in settings[key]):
                            logger.warning(f"Invalid SYMBOLS value {settings[key]}, using default: {value}")
                            settings[key] = value
                    if key == "LICENSE_KEY" and settings[key] is not None:
                        settings[key] = str(settings[key])
                    if key == "RENDER_SERVER_URL":
                        settings[key] = str(settings[key])
                        if not isinstance(settings[key], str):
                            logger.warning(f"Invalid RENDER_SERVER_URL {settings[key]}, must be a string, using default: {value}")
                            settings[key] = value
                        else:
                            try:
                                parsed = urlparse(settings[key])
                                if not all([parsed.scheme, parsed.netloc]):
                                    logger.warning(f"Invalid RENDER_SERVER_URL {settings[key]}, must be a valid URL, using default: {value}")
                                    settings[key] = value
                            except Exception:
                                logger.warning(f"Invalid RENDER_SERVER_URL {settings[key]}, using default: {value}")
                                settings[key] = value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid {key} value {settings[key]} in settings.json, using default: {value}")
                    settings[key] = value

        logger.info("Settings loaded successfully")
        return settings

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding settings.json: {e}, using default settings")
        return default_settings
    except Exception as e:
        logger.error(f"Error loading settings.json: {e}, using default settings")
        return default_settings

def save_settings(settings: Dict[str, Any]) -> bool:
    try:
        # Validate settings before saving
        for key, value in settings.items():
            if key in ["SCAN_INTERVAL", "LEVERAGE", "RISK_PCT", "VIRTUAL_BALANCE", "ENTRY_BUFFER_PCT", "MIN_SIGNAL_SCORE", "MIN_VOLUME", "MIN_ATR_PCT", "MAX_SPREAD_PCT"]:
                if float(value) <= 0:
                    logger.error(f"Invalid {key}: {value} must be positive")
                    return False
            if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"]:
                if float(value) >= 0:
                    logger.error(f"Invalid {key}: {value} must be negative")
                    return False
            if key in ["TOP_N_SIGNALS", "MAX_POSITIONS", "RSI_OVERSOLD", "RSI_OVERBOUGHT"]:
                if int(value) <= 0:
                    logger.error(f"Invalid {key}: {value} must be positive")
                    return False
            if key == "SYMBOLS":
                if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
                    logger.error(f"Invalid SYMBOLS: {value} must be a list of strings")
                    return False
            if key == "LICENSE_KEY" and value is not None:
                settings[key] = str(value)
            if key == "RENDER_SERVER_URL":
                if not isinstance(value, str):
                    logger.error(f"Invalid RENDER_SERVER_URL: {value} must be a string")
                    return False
                try:
                    parsed = urlparse(value)
                    if not all([parsed.scheme, parsed.netloc]):
                        logger.error(f"Invalid RENDER_SERVER_URL: {value} must be a valid URL")
                        return False
                except Exception:
                    logger.error(f"Invalid RENDER_SERVER_URL: {value} must be a valid URL")
                    return False
        
        with open("settings.json", "w") as f:
            json.dump(settings, f, indent=2)
        logger.info("Settings saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False

def validate_env() -> bool:
    required_vars = [
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
        "DATABASE_URL"
    ]
    optional_vars = [
        "DISCORD_WEBHOOK_URL",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "RENDER_SERVER_URL",
        "DEFAULT_SCAN_INTERVAL",
        "DEFAULT_TOP_N_SIGNALS",
        "LEVERAGE",
        "RISK_PCT",
        "ENTRY_BUFFER_PCT"
    ]
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    if missing_required:
        logger.error(f"Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    if missing_optional:
        logger.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")
    
    return True