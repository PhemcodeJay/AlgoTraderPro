import streamlit as st
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
import sqlite3
try:
    import psycopg2
except ImportError:
    psycopg2 = None
from logging_config import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Configure the Render server URL
RENDER_SERVER_URL = os.getenv("RENDER_SERVER_URL", "http://localhost:8000")

# PostgreSQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "licenses_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DATABASE_URL = os.getenv("DATABASE_URL", f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# SQLite fallback database
SQLITE_DB = "algotrader-license.db"

# Helper function to get database connection (PostgreSQL first, then SQLite)
def get_db_connection():
    """
    Attempt to connect to PostgreSQL, fall back to SQLite if it fails.
    
    Returns:
        tuple: (connection, cursor, db_type) where db_type is 'postgresql' or 'sqlite'.
    """
    if psycopg2:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            cursor = conn.cursor()
            logger.info("Connected to PostgreSQL database")
            return conn, cursor, "postgresql"
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}. Falling back to SQLite.")
    
    # Fallback to SQLite
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.cursor()
        logger.info("Connected to SQLite database")
        return conn, cursor, "sqlite"
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite: {e}")
        raise Exception("Unable to connect to any database")

# Helper function to initialize client tables
def init_client_tables():
    """
    Initialize client_settings and client_license_logs tables in PostgreSQL or SQLite.
    """
    try:
        conn, cursor, db_type = get_db_connection()
        
        # Create client_settings table
        if db_type == "postgresql":
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS client_settings (
                    key VARCHAR PRIMARY KEY,
                    value TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS client_license_logs (
                    license_key TEXT,
                    is_valid BOOLEAN,
                    message TEXT,
                    tier TEXT,
                    expiration_date TEXT,
                    timestamp TEXT
                )
            """)
        else:  # SQLite
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS client_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS client_license_logs (
                    license_key TEXT,
                    is_valid BOOLEAN,
                    message TEXT,
                    tier TEXT,
                    expiration_date TEXT,
                    timestamp TEXT
                )
            """)
        
        conn.commit()
        logger.info(f"Initialized client tables in {db_type} database")
    except Exception as e:
        logger.error(f"Error initializing client tables in {db_type}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# Helper function to get license key from DB
def get_license_from_db():
    """
    Retrieve the license key from the client_settings table.
    
    Returns:
        str: License key or empty string if not found.
    """
    try:
        conn, cursor, db_type = get_db_connection()
        if db_type == "postgresql":
            cursor.execute("SELECT value FROM client_settings WHERE key = %s", ("license_key",))
        else:
            cursor.execute("SELECT value FROM client_settings WHERE key = ?", ("license_key",))
        result = cursor.fetchone()
        return result[0] if result else ""
    except Exception as e:
        logger.error(f"Error retrieving license from {db_type} database: {e}")
        return ""
    finally:
        cursor.close()
        conn.close()

# Helper function to save license key and log to DB
def save_license_to_db(license_key: str, result: dict) -> bool:
    """
    Save license key to client_settings and log validation result.
    
    Args:
        license_key (str): License key to save.
        result (dict): Validation result with 'valid', 'message', 'tier', 'expiration_date'.
    
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        conn, cursor, db_type = get_db_connection()
        if db_type == "postgresql":
            cursor.execute(
                "INSERT INTO client_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = %s",
                ("license_key", license_key, license_key)
            )
            cursor.execute(
                """
                INSERT INTO client_license_logs (license_key, is_valid, message, tier, expiration_date, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    license_key,
                    result["valid"],
                    result["message"],
                    result.get("tier"),
                    result.get("expiration_date"),
                    datetime.now().isoformat()
                )
            )
        else:  # SQLite
            cursor.execute(
                "INSERT OR REPLACE INTO client_settings (key, value) VALUES (?, ?)",
                ("license_key", license_key)
            )
            cursor.execute(
                """
                INSERT INTO client_license_logs (license_key, is_valid, message, tier, expiration_date, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    license_key,
                    result["valid"],
                    result["message"],
                    result.get("tier"),
                    result.get("expiration_date"),
                    datetime.now().isoformat()
                )
            )
        
        conn.commit()
        logger.info(f"Saved license {license_key} to {db_type} database, Valid: {result['valid']}")
        return True
    except Exception as e:
        logger.error(f"Error saving license to {db_type} database: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# API function to validate license
def validate_license(license_key, hostname=None, mac=None):
    """
    Validate a license key by making a POST request to the server.
    
    Args:
        license_key (str): The license key to validate.
        hostname (str, optional): The hostname of the machine.
        mac (str, optional): The MAC address of the machine.
    
    Returns:
        dict: Response containing 'valid', 'message', 'tier', and 'expiration_date'.
    """
    payload = {
        "license_key": license_key,
        "hostname": hostname,
        "mac": mac
    }
    try:
        response = requests.post(f"{RENDER_SERVER_URL}/validate", json=payload)
        response.raise_for_status()
        data = response.json()
        # Server does not return expiration_date, so keep it None
        data["expiration_date"] = None
        return data
    except requests.RequestException as e:
        logger.error(f"Error validating license {license_key}: {e}")
        return {
            "valid": False,
            "message": "Error connecting to license server. Please try again later.",
            "tier": None,
            "expiration_date": None
        }

# API function to create a new license
def create_license(email, tier="basic", days_valid=30):
    """
    Create a new license by making a POST request to the server.
    
    Args:
        email (str): User email for the license.
        tier (str): License tier (default: 'basic').
        days_valid (int): Number of days the license is valid (default: 30).
    
    Returns:
        dict: Response containing 'license_key', 'tier', 'days_valid', or 'error'.
    """
    payload = {
        "email": email,
        "tier": tier,
        "days_valid": days_valid
    }
    try:
        response = requests.post(f"{RENDER_SERVER_URL}/create", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error creating license: {e}")
        return {"error": "Failed to create license"}

# API function to activate a license
def activate_license(license_key):
    """
    Activate a license by making a POST request to the server.
    
    Args:
        license_key (str): The license key to activate.
    
    Returns:
        bool: True if activation was successful, False otherwise.
    """
    payload = {"license_key": license_key}
    try:
        response = requests.post(f"{RENDER_SERVER_URL}/activate", json=payload)
        response.raise_for_status()
        return response.json().get("success", False)
    except requests.RequestException as e:
        logger.error(f"Error activating license {license_key}: {e}")
        return False

# API function to deactivate a license
def deactivate_license(license_key):
    """
    Deactivate a license by making a POST request to the server.
    
    Args:
        license_key (str): The license key to deactivate.
    
    Returns:
        bool: True if deactivation was successful, False otherwise.
    """
    payload = {"license_key": license_key}
    try:
        response = requests.post(f"{RENDER_SERVER_URL}/deactivate", json=payload)
        response.raise_for_status()
        return response.json().get("success", False)
    except requests.RequestException as e:
        logger.error(f"Error deactivating license {license_key}: {e}")
        return False

# API function to delete a license
def delete_license(license_key):
    """
    Delete a license by making a POST request to the server.
    
    Args:
        license_key (str): The license key to delete.
    
    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    payload = {"license_key": license_key}
    try:
        response = requests.post(f"{RENDER_SERVER_URL}/delete", json=payload)
        response.raise_for_status()
        return response.json().get("success", False)
    except requests.RequestException as e:
        logger.error(f"Error deleting license {license_key}: {e}")
        return False

# API function to list licenses
def list_licenses(limit=500):
    """
    List licenses by making a GET request to the server.
    
    Args:
        limit (int): Maximum number of licenses to return (default: 500).
    
    Returns:
        list: List of licenses or empty list if request fails.
    """
    try:
        response = requests.get(f"{RENDER_SERVER_URL}/list_licenses?limit={limit}")
        response.raise_for_status()
        return response.json().get("licenses", [])
    except requests.RequestException as e:
        logger.error(f"Error listing licenses: {e}")
        return []

# API function to list logs
def list_logs(limit=500):
    """
    List license logs by making a GET request to the server.
    
    Args:
        limit (int): Maximum number of logs to return (default: 500).
    
    Returns:
        list: List of logs or empty list if request fails.
    """
    try:
        response = requests.get(f"{RENDER_SERVER_URL}/list_logs?limit={limit}")
        response.raise_for_status()
        return response.json().get("logs", [])
    except requests.RequestException as e:
        logger.error(f"Error listing logs: {e}")
        return []

# Helper function to format expiration date (exportable)
def format_expiration_date(expiration_str):
    """
    Format an ISO expiration date string into a human-readable format.
    
    Args:
        expiration_str (str): ISO format date string (e.g., '2025-10-30T21:37:00.123456').
    
    Returns:
        str: Formatted date string (e.g., 'October 30, 2025, 9:37 PM') or 'N/A' if invalid.
    """
    try:
        expiration_dt = datetime.fromisoformat(expiration_str.replace('Z', '+00:00'))
        return expiration_dt.strftime("%B %d, %Y, %I:%M %p")
    except (ValueError, TypeError):
        return "N/A"

# Function to check license and display input form if invalid
def check_license():
    """
    Check if a valid license exists in session state or database.
    If invalid or missing, display a license input form and block further access.
    
    Returns:
        tuple: (is_valid: bool, result: dict) where is_valid indicates if the license is valid,
               and result contains 'valid', 'message', 'tier', 'expiration_date', and 'formatted_expiration_date'.
    """
    init_client_tables()
    # Check for saved license key in session state or database
    if "license_key" not in st.session_state:
        st.session_state.license_key = get_license_from_db()

    # Validate license key
    license_result = {"valid": False, "message": "No license key provided", "tier": None, "expiration_date": None}
    if st.session_state.license_key:
        license_result = validate_license(st.session_state.license_key)
        save_license_to_db(st.session_state.license_key, license_result)
    
    # If license is invalid or missing, show license input form
    if not license_result["valid"]:
        st.error(f"Access Denied: {license_result['message']}")
        with st.form("license_form"):
            license_key_input = st.text_input(
                "License Key",
                value=st.session_state.license_key,
                placeholder="e.g., 123e4567-e89b-12d3-a456-426614174000",
                help="Enter a valid license key to access this page."
            )
            submitted = st.form_submit_button("Submit License")
            if submitted:
                if not license_key_input:
                    st.error("Please enter a license key.")
                    return False, license_result
                license_result = validate_license(license_key_input)
                if license_result["valid"]:
                    st.session_state.license_key = license_key_input
                    save_license_to_db(license_key_input, license_result)
                    license_result["formatted_expiration_date"] = format_expiration_date(license_result.get("expiration_date"))
                    st.success(f"License validated! Tier: {license_result['tier']}, Expires: {license_result['formatted_expiration_date']}")
                    st.rerun()
                    return True, license_result
                else:
                    st.error(f"Invalid license key: {license_result['message']}")
                    save_license_to_db(license_key_input, license_result)
                    return False, license_result
        return False, license_result

    # If license is valid, add formatted expiration date and return
    license_result["formatted_expiration_date"] = format_expiration_date(license_result.get("expiration_date"))
    st.markdown(f"**License Status:** ✅ Valid (Tier: {license_result['tier']}, Expires: {license_result['formatted_expiration_date']})")
    return True, license_result

# Export functions for use in other modules
__all__ = ["validate_license", "create_license", "activate_license", "deactivate_license", "delete_license", "list_licenses", "list_logs", "format_expiration_date", "check_license"]