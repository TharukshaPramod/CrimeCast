import streamlit as st
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import login_user, is_authenticated, navigate_to_page

def show_login_form():
    """Display login form - standalone version for login page"""
    st.header("🔐 Login to CrimeCast")
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        remember_me = st.checkbox("Remember me")
        
        submit = st.form_submit_button("🚀 Login", use_container_width=True)
        
        if submit:
            if email and password:
                if login_user(email, password):
                    st.success("✅ Login successful! Redirecting...")
                    navigate_to_page("3_🏠_Home.py")
                # Note: No else needed here as login_user shows error
            else:
                st.error("❌ Please fill in all fields")

def main():
    st.set_page_config(
        page_title="Login - CrimeCast", 
        page_icon="🔐",
        layout="centered"
    )
    
    # Custom CSS for login page
    st.markdown("""
    <style>
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
        }
        .forgot-password {
            text-align: right;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # If already authenticated, redirect to home
    if is_authenticated():
        st.success("🎉 Already logged in! Redirecting to Home...")
        navigate_to_page("3_🏠_Home.py")
        return
    
    # Show login form
    show_login_form()
    
    st.markdown("---")
    
    # Additional options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### New to CrimeCast?")
        if st.button("📝 Sign Up Here", use_container_width=True, type="secondary"):
            navigate_to_page("2_📝_Sign_Up.py")
    
    with col2:
        st.markdown("### Need help?")
        if st.button("🔒 Forgot Password", use_container_width=True, type="secondary"):
            st.info("📧 Password reset feature coming soon!")

if __name__ == "__main__":
    main()