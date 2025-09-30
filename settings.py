import json
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv, set_key
from check_license import validate_license, format_expiration_date  # Import license validation functions

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
        "MIN_SIGNAL_SCORE": 60,
        "LICENSE_KEY": os.getenv("LICENSE_KEY", "")
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
                            
                    if key == "SYMBOLS" and not isinstance(settings[key], list):
                        logger.warning(f"Invalid SYMBOLS value {settings[key]}, using default: {value}")
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
            if key == "TOP_N_SIGNALS" and int(value) <= 0:
                logger.error(f"Invalid TOP_N_SIGNALS: {value} must be positive")
                return False
            if key == "SYMBOLS" and not isinstance(value, list):
                logger.error(f"Invalid SYMBOLS: {value} must be a list")
                return False
        
        with open("settings.json", "w") as f:
            json.dump(settings, f, indent=2)
        logger.info("Settings saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False

def validate_license_key(license_key: str, hostname: Optional[str] = None, mac: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate a license key using the external license server.
    
    Args:
        license_key (str): The license key to validate.
        hostname (str, optional): The hostname of the machine.
        mac (str, optional): The MAC address of the machine.
    
    Returns:
        dict: Response containing 'valid', 'message', 'tier', 'expiration_date', and 'formatted_expiration_date'.
    """
    try:
        # Only pass hostname and mac if they are not None to satisfy type checking
        kwargs = {}
        if hostname is not None:
            kwargs["hostname"] = hostname
        if mac is not None:
            kwargs["mac"] = mac
        result = validate_license(license_key, **kwargs)
        if result["valid"]:
            result["formatted_expiration_date"] = format_expiration_date(result.get("expiration_date"))
            logger.info(f"License validated: {license_key}, Tier: {result['tier']}, Expires: {result['formatted_expiration_date']}")
        else:
            logger.error(f"License validation failed: {license_key}, Message: {result['message']}")
        return result
    except Exception as e:
        logger.error(f"Error validating license {license_key}: {e}")
        return {"valid": False, "message": f"License validation failed: {str(e)}", "tier": None, "expiration_date": None}

def save_env_settings(env_vars: Dict[str, str]) -> bool:
    """
    Save environment variables to the .env file.
    
    Args:
        env_vars (dict): Dictionary of environment variable names and their values.
    
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        env_file = ".env"
        for key, value in env_vars.items():
            if value:  # Only save non-empty values
                set_key(env_file, key, str(value))
                logger.info(f"Saved environment variable {key}")
        logger.info("Environment variables saved successfully to .env")
        load_dotenv()  # Reload .env to ensure changes are applied
        return True
    except Exception as e:
        logger.error(f"Error saving environment variables: {e}")
        return False

def validate_env() -> bool:
    required_vars = [
        "BYBIT_API_KEY", "BYBIT_API_SECRET", "LICENSE_KEY"
    ]
    optional_vars = [
        "DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "WHATSAPP_TO", "DATABASE_URL"
    ]
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    if missing_required:
        logger.error(f"Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    if missing_optional:
        logger.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")
    
    return True