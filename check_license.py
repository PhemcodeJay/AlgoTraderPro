import streamlit as st
from db import db_manager
from settings import load_settings
from exceptions import DatabaseException, create_error_context
import requests
import socket
from logging_config import get_logger
from dataclasses import asdict

logger = get_logger(__name__)

def check_and_validate_license():
    """Check and validate license status, display status, and prompt for key if invalid."""
    # Load settings
    settings = load_settings()
    LICENSE_KEY = settings.get("LICENSE_KEY")
    USE_LOCAL_VALIDATION = settings.get("USE_LOCAL_VALIDATION", True)
    LICENSE_SERVER_URL = settings.get("RENDER_SERVER_URL", "http://localhost:8000")

    # Initialize session state
    if "license_valid" not in st.session_state:
        st.session_state.license_valid = False
    if "license_key" not in st.session_state:
        st.session_state.license_key = LICENSE_KEY
    if "license_tier" not in st.session_state:
        st.session_state.license_tier = None

    def validate_license_key(key: str) -> dict:
        """Validate a license key using local or online validation."""
        try:
            if USE_LOCAL_VALIDATION:
                valid, message, info = db_manager.validate_license(key, socket.gethostname(), None)
                return {
                    "valid": valid,
                    "message": message,
                    "tier": info["tier"] if info else None,
                    "info": info
                }
            else:
                payload = {"license_key": key, "hostname": socket.gethostname(), "mac": None}
                resp = requests.post(f"{LICENSE_SERVER_URL}/validate", json=payload, timeout=5)
                if resp.status_code == 200:
                    return resp.json()
                return {"valid": False, "message": f"Server returned {resp.status_code}", "tier": None, "info": None}
        except Exception as e:
            context = create_error_context(module=__name__, function="validate_license_key", line_number=None)
            logger.error(f"License validation error: {str(e)}", extra=asdict(context))
            return {"valid": False, "message": f"License validation failed: {str(e)}", "tier": None, "info": None}

    # Check license validity
    if not st.session_state.license_valid or not st.session_state.license_key:
        # Try loading from database
        try:
            saved_license = db_manager.get_setting("license_key")
            if saved_license:
                result = validate_license_key(saved_license)
                if result.get("valid"):
                    st.session_state.license_valid = True
                    st.session_state.license_key = saved_license
                    st.session_state.license_tier = result.get("tier", "basic")
                    settings["LICENSE_KEY"] = saved_license
                    from settings import save_settings
                    save_settings(settings)
                    logger.info(f"Loaded valid license from DB: {saved_license}")
                else:
                    settings["LICENSE_KEY"] = None
                    from settings import save_settings
                    save_settings(settings)
                    logger.error(f"DB license invalid: {result.get('message')}")
        except DatabaseException as e:
            logger.error(f"Database error checking license: {e}")
            st.session_state.license_valid = False

    # Display license status
    st.markdown("### 🔑 License Status")
    if st.session_state.license_valid and st.session_state.license_key:
        try:
            info = db_manager.get_license_info(st.session_state.license_key) if USE_LOCAL_VALIDATION else validate_license_key(st.session_state.license_key).get("info")
            if info:
                st.write(f"**License Key:** {st.session_state.license_key}")
                st.write(f"**Tier:** {info['tier']}")
                st.write(f"**Email:** {info['user_email'] or 'N/A'}")
                st.write(f"**Active:** {'Yes' if info['is_active'] else 'No'}")
                st.write(f"**Expiration Date:** {info['expiration_date']}")
                st.write(f"**Created At:** {info['created_at']}")
                st.write(f"**Notes:** {info['notes'] or 'N/A'}")
                st.info("License is valid. You can proceed with trading.")
            else:
                st.warning("License info not available.")
                st.session_state.license_valid = False
        except DatabaseException as e:
            logger.error(f"Error fetching license info: {e}")
            st.error(f"Error fetching license info: {e.message}")
            st.session_state.license_valid = False
    else:
        st.error("No valid license found. Please enter a valid license key.")
        # License input form
        license_key = st.text_input("Enter your License Key", type="password", key=f"license_input_{st.session_state.get('current_page', 'page')}")
        if st.button("Validate License", key=f"validate_button_{st.session_state.get('current_page', 'page')}"):
            if license_key:
                result = validate_license_key(license_key)
                if result.get("valid"):
                    st.session_state.license_valid = True
                    st.session_state.license_key = license_key
                    st.session_state.license_tier = result.get("tier", "basic")
                    db_manager.save_setting("license_key", license_key)
                    settings["LICENSE_KEY"] = license_key
                    from settings import save_settings
                    save_settings(settings)
                    st.success(f"License validated successfully! Tier: {st.session_state.license_tier}")
                    logger.info(f"License validated: {license_key}, Tier: {st.session_state.license_tier}")
                    st.rerun()
                else:
                    st.error(f"License invalid: {result.get('message')}")
                    logger.error(f"License validation failed: {result.get('message')}")
            else:
                st.error("Please enter a license key.")
                logger.warning("License key input was empty")
        st.stop()  # Stop page execution if license is invalid

# Execute license check
check_and_validate_license()