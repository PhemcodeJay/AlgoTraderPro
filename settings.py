import json
import os
from typing import Dict, Any
from logging_config import get_logger
from db import db_manager, DatabaseException

logger = get_logger(__name__)

def load_settings() -> Dict[str, Any]:
    default_settings = {
        "SCAN_INTERVAL": int(os.getenv("DEFAULT_SCAN_INTERVAL", 3600)),
        "TOP_N_SIGNALS": int(os.getenv("DEFAULT_TOP_N_SIGNALS", 5)),
        "MAX_LOSS_PCT": -15.0,
        "TP_PERCENT": 50.0,
        "SL_PERCENT": 10.0,
        "MAX_DRAWDOWN_PCT": -20.0,
        "LEVERAGE": float(os.getenv("LEVERAGE", 15)),
        "RISK_PCT": float(os.getenv("RISK_PCT", 0.02)),
        "ENTRY_BUFFER_PCT": float(os.getenv("ENTRY_BUFFER_PCT", 0.002)),
        "SYMBOLS": ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT"],
        "USE_WEBSOCKET": True,
        "MAX_POSITIONS": 5,
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
        if not db_manager.is_connected():
            logger.error("Database not connected, returning default settings")
            return default_settings

        settings = {}
        for key in default_settings:
            value = db_manager.get_setting(key)
            if value is not None:
                try:
                    # Convert stored string value to appropriate type
                    if isinstance(default_settings[key], bool):
                        settings[key] = value.lower() == "true"
                    elif isinstance(default_settings[key], (int, float)):
                        settings[key] = float(value)
                        if key in ["LEVERAGE", "RISK_PCT", "ENTRY_BUFFER_PCT"]:
                            if settings[key] <= 0:
                                logger.warning(f"Invalid {key} value {settings[key]}, using default: {default_settings[key]}")
                                settings[key] = default_settings[key]
                        if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"] and settings[key] > 0:
                            logger.warning(f"Invalid {key} value {settings[key]}, using default: {default_settings[key]}")
                            settings[key] = default_settings[key]
                    elif isinstance(default_settings[key], list):
                        settings[key] = eval(value) if value else default_settings[key]
                    else:
                        settings[key] = value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid {key} value {value}, using default: {default_settings[key]}")
                    settings[key] = default_settings[key]
            else:
                settings[key] = default_settings[key]
                logger.warning(f"Missing {key} in database, using default: {default_settings[key]}")

        logger.info("Settings loaded successfully from database")
        return settings

    except DatabaseException as e:
        logger.error(f"Database error loading settings: {e}")
        return default_settings
    except Exception as e:
        logger.error(f"Unexpected error loading settings: {e}")
        return default_settings

def save_settings(settings: Dict[str, Any]) -> bool:
    try:
        if not db_manager.is_connected():
            logger.error("Database not connected, cannot save settings")
            return False

        # Validate settings
        errors = []
        for key, value in settings.items():
            if key in ["LEVERAGE", "RISK_PCT", "ENTRY_BUFFER_PCT"] and float(value) <= 0:
                errors.append(f"{key}: {value} must be positive")
            if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"] and float(value) > 0:
                errors.append(f"{key}: {value} must be negative")

        if errors:
            logger.error(f"Validation errors: {'; '.join(errors)}")
            return False

        # Save each setting to the database
        for key, value in settings.items():
            # Convert value to string for storage
            if isinstance(value, (list, dict)):
                value = str(value)
            elif isinstance(value, bool):
                value = str(value).lower()
            else:
                value = str(value)
            if not db_manager.save_setting(key, value):
                logger.error(f"Failed to save setting {key}")
                return False

        logger.info("Settings saved successfully to database")
        return True

    except DatabaseException as e:
        logger.error(f"Database error saving settings: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving settings: {e}")
        return False

def validate_env() -> bool:
    required_vars = ["BYBIT_API_KEY", "BYBIT_API_SECRET"]
    optional_vars = ["DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DATABASE_URL"]
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    if missing_required:
        logger.error(f"Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    if missing_optional:
        logger.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")
    
    return True

def migrate_json_to_db(json_file_path: str = "settings.json") -> bool:
    try:
        if not os.path.exists(json_file_path):
            logger.warning(f"Settings file {json_file_path} not found, skipping migration")
            return True

        with open(json_file_path, "r") as f:
            json_settings = json.load(f)

        for key, value in json_settings.items():
            if isinstance(value, (list, dict)):
                value = str(value)
            elif isinstance(value, bool):
                value = str(value).lower()
            else:
                value = str(value)
            if not db_manager.save_setting(key, value):
                logger.error(f"Failed to migrate setting {key} to database")
                return False

        logger.info(f"Settings migrated successfully from {json_file_path} to database")
        # Optionally, rename or remove the JSON file to prevent re-migration
        os.rename(json_file_path, f"{json_file_path}.backup")
        return True

    except Exception as e:
        logger.error(f"Error migrating settings from {json_file_path} to database: {e}")
        return False