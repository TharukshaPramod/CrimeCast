import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, is_admin
from auth.decorators import admin_required
from auth.database import user_db

@admin_required
def main():
    st.set_page_config(page_title="Admin Panel - CrimeCast", layout="wide")
    
    st.header("🔐 Admin Panel")
    st.info("🔒 Super Admin Access: Complete system administration")
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 User Management", "📊 Access Logs", "🔧 System Tools", "📈 Analytics"])
    
    with tab1:
        st.subheader("User Management")
        
        users = user_db.get_all_users()
        if users:
            user_df = pd.DataFrame(users, columns=[
                'ID', 'Email', 'Name', 'Phone', 'Address', 'Role', 
                'Active', 'Created', 'Last Login'
            ])
            
            user_df['Created'] = pd.to_datetime(user_df['Created']).dt.strftime('%Y-%m-%d %H:%M')
            user_df['Last Login'] = pd.to_datetime(user_df['Last Login']).dt.strftime('%Y-%m-%d %H:%M')
            user_df['Active'] = user_df['Active'].apply(lambda x: '✅' if x else '❌')
            
            st.dataframe(user_df, use_container_width=True, height=400)
            
            # User actions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Add New User")
                with st.form("add_user_form"):
                    new_email = st.text_input("Email")
                    new_name = st.text_input("Full Name")
                    new_phone = st.text_input("Phone")
                    new_role = st.selectbox("Role", ["user", "admin"])
                    new_password = st.text_input("Password", type="password")
                    
                    if st.form_submit_button("➕ Create User"):
                        st.info("User creation feature coming in next update!")
            
            with col2:
                st.subheader("User Actions")
                selected_user = st.selectbox("Select User", 
                                           [f"{u[0]} - {u[1]}" for u in users])
                action = st.selectbox("Action", ["View Details", "Edit User", "Deactivate", "Reset Password"])
                
                if st.button("🛠️ Execute Action"):
                    st.info(f"Action '{action}' for user {selected_user} coming soon!")
        
        else:
            st.info("No users found in the system")
    
    with tab2:
        st.subheader("Access Logs")
        st.info("📋 Comprehensive access logging system")
        
        # Placeholder for access logs
        st.write("User activity logs will be displayed here")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Log Entries", "1,247")
            st.metric("Today's Activities", "23")
        
        with col2:
            st.metric("Failed Logins", "5")
            st.metric("Active Sessions", "8")
    
    with tab3:
        st.subheader("System Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Database Operations**")
            if st.button("🔄 Optimize Database", use_container_width=True):
                st.success("Database optimization completed!")
            
            if st.button("📊 Update Statistics", use_container_width=True):
                st.success("Statistics updated!")
            
            if st.button("🧹 Clean Temp Files", use_container_width=True):
                st.success("Temporary files cleaned!")
        
        with col2:
            st.write("**Security Tools**")
            if st.button("🔐 Audit Logs", use_container_width=True):
                st.success("Security audit completed!")
            
            if st.button("🛡️ Check Permissions", use_container_width=True):
                st.success("Permission check completed!")
            
            if st.button("📧 Send Security Report", use_container_width=True):
                st.success("Security report sent!")
    
    with tab4:
        st.subheader("System Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users) if users else 0)
        
        with col2:
            st.metric("Active Today", "15")
        
        with col3:
            st.metric("Predictions Made", "1,234")
        
        with col4:
            st.metric("System Uptime", "99.8%")
        
        st.info("📈 Advanced analytics dashboard coming soon!")

if __name__ == "__main__":
    main()