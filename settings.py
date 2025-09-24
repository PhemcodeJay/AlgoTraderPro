import json
import logging
import os
from typing import Dict, Any

# Configure logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

def load_settings() -> Dict[str, Any]:
    default_settings = {
        "SCAN_INTERVAL": float(os.getenv("DEFAULT_SCAN_INTERVAL", 3600.0)),
        "TOP_N_SIGNALS": int(os.getenv("DEFAULT_TOP_N_SIGNALS", 10)),
        "MAX_LOSS_PCT": -15.0,
        "TP_PERCENT": 0.25,
        "SL_PERCENT": 0.05,
        "MAX_DRAWDOWN_PCT": -20.0,
        "LEVERAGE": float(os.getenv("LEVERAGE", 10.0)),
        "RISK_PCT": float(os.getenv("RISK_PCT", 0.02)),
        "VIRTUAL_BALANCE": 100.0,
        "ENTRY_BUFFER_PCT": float(os.getenv("ENTRY_BUFFER_PCT", 0.002)),
        "SYMBOLS": ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"],
        "USE_WEBSOCKET": True,
        "MAX_POSITIONS": 10,
        "MIN_SIGNAL_SCORE": 40.0,
        "AUTO_TRADING_ENABLED": True,
        "NOTIFICATION_ENABLED": True,
        "RSI_OVERSOLD": 30,
        "RSI_OVERBOUGHT": 70,
        "MIN_VOLUME": 1000000,
        "MIN_ATR_PCT": 0.5,
        "MAX_SPREAD_PCT": 0.1
    }

    try:
        if not os.path.exists("settings.json"):
            logger.warning("settings file not found, creating with default settings")
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
                    if key in ["SYMBOLS", "USE_WEBSOCKET", "AUTO_TRADING_ENABLED", "NOTIFICATION_ENABLED"]:
                        continue  # Skip validation for non-numeric fields
                    settings[key] = float(settings[key]) if isinstance(value, float) else int(settings[key])
                    if key in ["LEVERAGE", "RISK_PCT", "VIRTUAL_BALANCE", "ENTRY_BUFFER_PCT", "MIN_SIGNAL_SCORE", "MIN_VOLUME", "MIN_ATR_PCT", "MAX_SPREAD_PCT"]:
                        if settings[key] <= 0:
                            logger.warning(f"Invalid {key} value {settings[key]}, using default: {value}")
                            settings[key] = value
                    if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"] and settings[key] > 0:
                        logger.warning(f"Invalid {key} value {settings[key]}, using default: {value}")
                        settings[key] = value
                    if key == "TOP_N_SIGNALS" and settings[key] <= 0:
                        logger.warning(f"Invalid TOP_N_SIGNALS value {settings[key]}, using default: {value}")
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
            if key in ["LEVERAGE", "RISK_PCT", "VIRTUAL_BALANCE", "ENTRY_BUFFER_PCT", "MIN_SIGNAL_SCORE", "MIN_VOLUME", "MIN_ATR_PCT", "MAX_SPREAD_PCT"]:
                if float(value) <= 0:
                    logger.error(f"Invalid {key}: {value} must be positive")
                    return False
            if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"] and float(value) > 0:
                logger.error(f"Invalid {key}: {value} must be negative")
                return False
            if key == "TOP_N_SIGNALS" and int(value) <= 0:
                logger.error(f"Invalid TOP_N_SIGNALS: {value} must be positive")
                return False
        
        with open("settings.json", "w") as f:
            json.dump(settings, f, indent=2)
        logger.info("Settings saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False

def validate_env() -> bool:
    required_vars = ["BYBIT_API_KEY", "BYBIT_API_SECRET"]
    optional_vars = ["DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "WHATSAPP_TO", "DATABASE_URL"]
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    if missing_required:
        logger.error(f"Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    if missing_optional:
        logger.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")
    
    return True