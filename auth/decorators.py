import streamlit as st
from auth.auth_utils import is_authenticated, is_admin

def admin_required(func):
    """Decorator to restrict access to admin users only"""
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            st.error("🔐 Please log in to access this page.")
            return
        if not is_admin():
            st.error("⛔ Admin access required for this page.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

def login_required(func):
    """Decorator to require user authentication"""
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            st.error("🔐 Please log in to access this page.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper