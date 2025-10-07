import streamlit as st
import bcrypt
from auth.database import user_db

def initialize_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    if 'show_login' not in st.session_state:
        st.session_state.show_login = False
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False

def login_user(email, password):
    """Authenticate and login user"""
    success, result = user_db.authenticate_user(email, password)
    
    if success:
        st.session_state.authenticated = True
        st.session_state.user = result
        st.session_state.show_login = False
        user_db.log_action(result['id'], "USER_LOGIN", "User logged in successfully")
        st.success(f"🎉 Welcome back, {result['name']}!")
        return True
    else:
        st.error(result)
        return False

def logout_user():
    """Logout current user"""
    if st.session_state.authenticated and st.session_state.user:
        user_db.log_action(st.session_state.user['id'], "USER_LOGOUT", "User logged out")
    
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.current_page = "Home"
    st.success("👋 Logged out successfully!")

def is_authenticated():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def is_admin():
    """Check if current user is admin"""
    return is_authenticated() and st.session_state.user.get('role') == 'admin'

def get_current_user():
    """Get current user data"""
    return st.session_state.get('user')

def navigate_to_page(page_name):
    """Safe navigation to pages with error handling"""
    try:
        st.switch_page(f"pages/{page_name}")
    except Exception as e:
        st.error(f"Navigation error: {e}")
        st.info("Please use the sidebar for navigation")

def require_auth():
    """Decorator to require authentication"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                st.error("🔐 Please log in to access this page.")
                navigate_to_page("1_🔐_Login.py")
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_admin():
    """Decorator to require admin role"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                st.error("🔐 Please log in to access this page.")
                navigate_to_page("1_🔐_Login.py")
                return
            if not is_admin():
                st.error("⛔ Admin access required for this page.")
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator

def admin_required(func):
    """Decorator to restrict access to admin users only"""
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            st.error("🔐 Please log in to access this page.")
            navigate_to_page("1_🔐_Login.py")
            return
        if not is_admin():
            st.error("⛔ Admin access required for this page.")
            return
        return func(*args, **kwargs)
    return wrapper

def login_required(func):
    """Decorator to require user authentication"""
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            st.error("🔐 Please log in to access this page.")
            navigate_to_page("1_🔐_Login.py")
            return
        return func(*args, **kwargs)
    return wrapper