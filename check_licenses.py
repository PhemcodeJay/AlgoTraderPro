import streamlit as st
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
import sqlite3
from logging_config import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Configure the Render server URL
RENDER_SERVER_URL = os.getenv("RENDER_SERVER_URL", "https://localhost:8000")

# Helper function to validate license (exportable)
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
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error validating license {license_key}: {e}")
        return {"valid": False, "message": f"Error connecting to server: {str(e)}", "tier": None, "expiration_date": None}

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

# Helper function to initialize license tables
def init_license_tables():
    try:
        conn = sqlite3.connect("algotrader-license.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS license_logs (
                license_key TEXT,
                is_valid BOOLEAN,
                message TEXT,
                tier TEXT,
                expiration_date TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error initializing license tables: {e}")

# Helper function to get license key from DB
def get_license_from_db():
    try:
        conn = sqlite3.connect("algotrader-license.db")
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", ("license_key",))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else ""
    except Exception as e:
        logger.error(f"Error retrieving license from database: {e}")
        return ""

# Helper function to save license key and log to DB
def save_license_to_db(license_key: str, result: dict) -> bool:
    try:
        conn = sqlite3.connect("algotrader-license.db")
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ("license_key", license_key))
        cursor.execute("""
            INSERT INTO license_logs (license_key, is_valid, message, tier, expiration_date, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            license_key,
            result["valid"],
            result["message"],
            result.get("tier"),
            result.get("expiration_date"),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        logger.info(f"Saved license {license_key} to database, Valid: {result['valid']}")
        return True
    except Exception as e:
        logger.error(f"Error saving license to database: {e}")
        return False

# Function to check license and display input form if invalid
def check_license():
    """
    Check if a valid license exists in session state or database.
    If invalid or missing, display a license input form and block further access.
    
    Returns:
        tuple: (is_valid: bool, result: dict) where is_valid indicates if the license is valid,
               and result contains 'valid', 'message', 'tier', 'expiration_date', and 'formatted_expiration_date'.
    """
    init_license_tables()
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
            submitted = st.form_submit_button("Validate License")
            if submitted:
                if not license_key_input:
                    st.error("Please enter a license key.")
                    return False, license_result
                license_result = validate_license(license_key_input)
                if license_result["valid"]:
                    st.session_state.license_key = license_key_input
                    save_license_to_db(license_key_input, license_result)
                    license_result["formatted_expiration_date"] = format_expiration_date(license_result["expiration_date"])
                    st.success(f"License validated! Tier: {license_result['tier']}, Expires: {license_result['formatted_expiration_date']}")
                    st.rerun()
                    return True, license_result
                else:
                    st.error(f"Invalid license key: {license_result['message']}")
                    save_license_to_db(license_key_input, license_result)
                    return False, license_result
        return False, license_result

    # If license is valid, add formatted expiration date and return
    license_result["formatted_expiration_date"] = format_expiration_date(license_result["expiration_date"])
    st.markdown(f"**License Status:** ✅ Valid (Tier: {license_result['tier']}, Expires: {license_result['formatted_expiration_date']})")
    return True, license_result

# Export functions for use in other modules
__all__ = ["validate_license", "format_expiration_date", "check_license"]