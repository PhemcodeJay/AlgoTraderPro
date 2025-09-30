import streamlit as st
import requests
from dotenv import load_dotenv
import os
from datetime import datetime

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
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.RequestException as e:
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
        # Parse ISO format string to datetime
        expiration_dt = datetime.fromisoformat(expiration_str.replace('Z', '+00:00'))
        # Format as "Month Day, Year, Hour:Minute AM/PM"
        return expiration_dt.strftime("%B %d, %Y, %I:%M %p")
    except (ValueError, TypeError):
        return "N/A"

# Streamlit UI
if __name__ == "__main__":
    st.set_page_config(page_title="License Validator", layout="wide")
    st.title("Check Your License Key")

    # Input for license key
    license_key = st.text_input("Enter License Key", placeholder="e.g., 123e4567-e89b-12d3-a456-426614174000")

    # Optional inputs for hostname and MAC address
    with st.expander("Advanced Options (Optional)"):
        hostname = st.text_input("Hostname (optional)", placeholder="e.g., my-computer")
        mac = st.text_input("MAC Address (optional)", placeholder="e.g., 00:14:22:01:23:45")

    # Validate button
    if st.button("Validate License"):
        if not license_key:
            st.error("Please enter a license key.")
        else:
            result = validate_license(license_key, hostname, mac)
            if result["valid"]:
                st.success(f"License is valid! {result['message']}")
                st.write(f"**Tier**: {result['tier']}")
                expiration_date = result.get('expiration_date', 'N/A')
                st.write(f"**Expiration Date**: {format_expiration_date(expiration_date)}")
            else:
                st.error(f"License is invalid. {result['message']}")

# Export functions for use in other modules
__all__ = ["validate_license", "format_expiration_date"]