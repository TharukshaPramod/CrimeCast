import streamlit as st
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import navigate_to_page
from auth.database import user_db

def show_signup_form():
    """Display signup form - standalone version for signup page"""
    st.header("📝 Create Your CrimeCast Account")
    
    with st.form("signup_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*", placeholder="Enter your full name")
            email = st.text_input("Email*", placeholder="Enter your email")
        
        with col2:
            phone = st.text_input("Phone Number", placeholder="Enter your phone number")
            address = st.text_area("Address", placeholder="Enter your address")
        
        password = st.text_input("Password*", type="password", placeholder="Create a password (min. 6 characters)")
        confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Confirm your password")
        
        # Terms agreement
        agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy*")
        
        submit = st.form_submit_button("🚀 Create Account", use_container_width=True)
        
        if submit:
            if not all([name, email, password, confirm_password]):
                st.error("❌ Please fill in all required fields (*)")
            elif not agree_terms:
                st.error("❌ Please agree to the Terms of Service and Privacy Policy")
            elif password != confirm_password:
                st.error("❌ Passwords do not match")
            elif len(password) < 6:
                st.error("❌ Password must be at least 6 characters long")
            else:
                # Create user account
                success, message = user_db.create_user(
                    email=email,
                    password=password,
                    name=name,
                    phone=phone,
                    address=address
                )
                if success:
                    st.success("✅ Account created successfully! Please log in.")
                    st.balloons()
                    # Redirect to login after 2 seconds
                    st.session_state.signup_success = True
                else:
                    st.error(f"❌ {message}")

def main():
    st.set_page_config(
        page_title="Sign Up - CrimeCast", 
        page_icon="📝",
        layout="centered"
    )
    
    # Custom CSS for signup page
    st.markdown("""
    <style>
        .signup-container {
            max-width: 600px;
            margin: 0 auto;
            padding: 2rem;
        }
        .success-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 2rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if signup was successful
    if st.session_state.get('signup_success', False):
        st.markdown("""
        <div class="success-message">
            <h2>🎉 Welcome to CrimeCast!</h2>
            <p>Your account has been created successfully.</p>
            <p>You can now log in to access all features.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 Login Now", use_container_width=True):
                st.session_state.signup_success = False
                navigate_to_page("1_🔐_Login.py")
        with col2:
            if st.button("🏠 Go Home", use_container_width=True):
                st.session_state.signup_success = False
                navigate_to_page("3_🏠_Home.py")
        return
    
    # Show signup form
    show_signup_form()
    
    st.markdown("---")
    
    # Additional info and login redirect
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🔒 Your Security Matters
        - All passwords are encrypted and securely stored
        - We never share your personal information
        - Enterprise-grade security protocols
        """)
    
    with col2:
        st.markdown("### Already have an account?")
        if st.button("🔐 Login Here", use_container_width=True, type="secondary"):
            navigate_to_page("1_🔐_Login.py")

if __name__ == "__main__":
    main()