import json
import logging
import os
from typing import Dict, Any

# Configure logging using centralized system
# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

def load_settings() -> Dict[str, Any]:
    default_settings = {
        "SCAN_INTERVAL": int(os.getenv("DEFAULT_SCAN_INTERVAL", 3600)),
        "TOP_N_SIGNALS": int(os.getenv("DEFAULT_TOP_N_SIGNALS", 5)),
        "MAX_LOSS_PCT": -15.0,
        "TP_PERCENT": 0.25,
        "SL_PERCENT": 0.05,
        "MAX_DRAWDOWN_PCT": -15.0,
        "LEVERAGE": float(os.getenv("LEVERAGE", 10)),
        "RISK_PCT": float(os.getenv("RISK_PCT", 0.01)),
        "VIRTUAL_BALANCE": 100.0,
        "ENTRY_BUFFER_PCT": float(os.getenv("ENTRY_BUFFER_PCT", 0.002)),
        "SYMBOLS": ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT"],
        "USE_WEBSOCKET": True,
        "MAX_POSITIONS": 5,
        "MIN_SIGNAL_SCORE": 60
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
                    # Validate numeric settings
                    if isinstance(value, (int, float)):
                        settings[key] = float(settings[key])
                        if key in ["LEVERAGE", "RISK_PCT", "VIRTUAL_BALANCE", "ENTRY_BUFFER_PCT"]:
                            if settings[key] <= 0:
                                logger.warning(f"Invalid {key} value {settings[key]}, using default: {value}")
                                settings[key] = value
                        if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"] and settings[key] > 0:
                            logger.warning(f"Invalid {key} value {settings[key]}, using default: {value}")
                            settings[key] = value
                    
                    if key == "TOP_N_SIGNALS":
                        settings[key] = int(settings[key])
                        if settings[key] <= 0:
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
            if key in ["LEVERAGE", "RISK_PCT", "VIRTUAL_BALANCE", "ENTRY_BUFFER_PCT"] and float(value) <= 0:
                logger.error(f"Invalid {key}: {value} must be positive")
                return False
            if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"] and float(value) > 0:
                logger.error(f"Invalid {key}: {value} must be negative")
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
        "BYBIT_API_KEY", "BYBIT_API_SECRET"
    ]
    optional_vars = [
        "DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DATABASE_URL"
    ]
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    if missing_required:
        logging.error(f"Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    if missing_optional:
        logging.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")
    
    return True
