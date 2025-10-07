import streamlit as st
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, get_current_user, logout_user
from auth.database import user_db  # Import only once

def main():
    st.set_page_config(page_title="Profile - CrimeCast", layout="wide")
    
    # Check authentication manually since decorator might have issues
    if not is_authenticated():
        st.error("🔐 Please log in to access this page.")
        st.stop()
    
    st.header("👤 User Profile")
    
    user = get_current_user()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profile Picture")
        st.info("🖼️ Profile picture upload coming soon!")
        
        st.subheader("Account Info")
        st.write(f"**User ID:** {user['id']}")
        st.write(f"**Email:** {user['email']}")
        st.write(f"**Role:** {user.get('role', 'user').title()}")
        st.write(f"**Status:** ✅ Active")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
    
    with col2:
        st.subheader("Edit Profile")
        
        with st.form("profile_form"):
            name = st.text_input("Full Name", value=user.get('name', ''))
            phone = st.text_input("Phone Number", value=user.get('phone', ''))
            address = st.text_area("Address", value=user.get('address', ''))
            
            if st.form_submit_button("💾 Update Profile"):
                success, message = user_db.update_user(
                    user_id=user['id'],
                    name=name,
                    phone=phone,
                    address=address
                )
                if success:
                    st.success(message)
                    # Update session state
                    st.session_state.user.update({
                        'name': name,
                        'phone': phone,
                        'address': address
                    })
                    st.rerun()
                else:
                    st.error(message)
        
        st.subheader("Security")
        st.info("🔒 Password change feature coming soon!")
        
        with st.expander("Account Statistics"):
            try:
                users = user_db.get_all_users()
                st.write(f"**Total Users in System:** {len(users)}")
                st.write(f"**Your Member Since:** {user.get('created_at', 'N/A')}")
                st.write(f"**Last Login:** {user.get('last_login', 'N/A')}")
            except Exception as e:
                st.warning(f"Could not load account statistics: {e}")

if __name__ == "__main__":
    main()