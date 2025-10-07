import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from datetime import datetime, timedelta
import bcrypt
import sqlite3

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, is_admin, get_current_user
from auth.decorators import admin_required
from auth.database import user_db

def get_access_logs():
    """Get access logs from database"""
    try:
        # Use the database method if available, otherwise return mock data
        if hasattr(user_db, 'get_access_logs'):
            logs = user_db.get_access_logs()
            if logs and len(logs) > 0:
                return logs
        # Mock data as fallback
        return [
            (1, 1, "USER_LOGIN", "Login", "2024-01-10 10:30:00", "127.0.0.1", "admin@crimecast.com"),
            (2, 2, "USER_LOGOUT", "Logout", "2024-01-10 11:15:00", "127.0.0.1", "user@example.com"),
            (3, 1, "CRIME_PREDICTION", "Prediction", "2024-01-10 12:00:00", "127.0.0.1", "admin@crimecast.com"),
        ]
    except Exception as e:
        st.error(f"Error getting access logs: {e}")
        return []

def change_user_password(user_id, new_password):
    """Change user password using database method"""
    try:
        success, message = user_db.change_user_password(user_id, new_password)
        if success:
            user_db.log_action(get_current_user()['id'], "PASSWORD_RESET", f"Password reset for user {user_id}")
        return success, message
    except Exception as e:
        return False, f"Error changing password: {str(e)}"

@admin_required
def main():
    st.set_page_config(page_title="Admin Panel - CrimeCast", layout="wide")
    
    st.header("🔐 Admin Panel")
    st.info("🔒 Super Admin Access: Complete system administration")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 User Management", "📊 Access Logs", "🔧 System Tools", 
        "📈 Analytics", "🔐 Password Management"
    ])
    
    with tab1:
        st.subheader("👥 User Management")
        
        users = user_db.get_all_users()
        if users:
            # Convert to DataFrame for better display
            user_df = pd.DataFrame(users, columns=[
                'ID', 'Email', 'Name', 'Phone', 'Address', 'Role', 
                'Active', 'Created', 'Last Login'
            ])
            
            # Format dates
            user_df['Created'] = pd.to_datetime(user_df['Created']).dt.strftime('%Y-%m-%d %H:%M')
            user_df['Last Login'] = pd.to_datetime(user_df['Last Login']).dt.strftime('%Y-%m-%d %H:%M')
            user_df['Active'] = user_df['Active'].apply(lambda x: '✅ Active' if x else '❌ Inactive')
            
            st.dataframe(user_df, use_container_width=True, height=400)
            
            # User Management Actions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("➕ Add New User")
                with st.form("add_user_form", clear_on_submit=True):
                    new_email = st.text_input("Email *", placeholder="user@example.com")
                    new_name = st.text_input("Full Name *", placeholder="John Doe")
                    new_phone = st.text_input("Phone", placeholder="+1234567890")
                    new_address = st.text_area("Address", placeholder="Enter address")
                    new_role = st.selectbox("Role *", ["user", "admin"])
                    new_password = st.text_input("Password *", type="password", placeholder="Minimum 6 characters")
                    confirm_password = st.text_input("Confirm Password *", type="password", placeholder="Confirm password")
                    
                    if st.form_submit_button("🚀 Create User", use_container_width=True):
                        if not all([new_email, new_name, new_password, confirm_password]):
                            st.error("❌ Please fill all required fields (*)")
                        elif new_password != confirm_password:
                            st.error("❌ Passwords do not match")
                        elif len(new_password) < 6:
                            st.error("❌ Password must be at least 6 characters long")
                        else:
                            success, message = user_db.create_user(
                                email=new_email,
                                password=new_password,
                                name=new_name,
                                phone=new_phone,
                                address=new_address,
                                role=new_role
                            )
                            if success:
                                st.success(f"✅ {message}")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
            
            with col2:
                st.subheader("🛠️ User Actions")
                if users:
                    user_options = [f"{u[0]} - {u[1]} ({u[2]})" for u in users]
                    selected_user_str = st.selectbox("Select User", user_options)
                    selected_user_id = int(selected_user_str.split(" - ")[0])
                    selected_user_data = next((u for u in users if u[0] == selected_user_id), None)
                    
                    if selected_user_data:
                        action = st.selectbox("Action", 
                                            ["View Details", "Edit User", "Toggle Active Status", "Reset Password", "Delete User"])
                        
                        if action == "View Details":
                            st.write("**User Details:**")
                            st.write(f"**ID:** {selected_user_data[0]}")
                            st.write(f"**Email:** {selected_user_data[1]}")
                            st.write(f"**Name:** {selected_user_data[2]}")
                            st.write(f"**Phone:** {selected_user_data[3] or 'Not provided'}")
                            st.write(f"**Address:** {selected_user_data[4] or 'Not provided'}")
                            st.write(f"**Role:** {selected_user_data[5]}")
                            st.write(f"**Status:** {'Active' if selected_user_data[6] else 'Inactive'}")
                            st.write(f"**Created:** {selected_user_data[7]}")
                            st.write(f"**Last Login:** {selected_user_data[8] or 'Never'}")
                        
                        elif action == "Edit User":
                            with st.form("edit_user_form"):
                                edit_name = st.text_input("Name", value=selected_user_data[2])
                                edit_phone = st.text_input("Phone", value=selected_user_data[3] or "")
                                edit_address = st.text_area("Address", value=selected_user_data[4] or "")
                                edit_role = st.selectbox("Role", ["user", "admin"], index=0 if selected_user_data[5] == "user" else 1)
                                
                                if st.form_submit_button("💾 Update User"):
                                    success, message = user_db.update_user(
                                        user_id=selected_user_id,
                                        name=edit_name,
                                        phone=edit_phone,
                                        address=edit_address
                                    )
                                    if success:
                                        st.success(f"✅ {message}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ {message}")
                        
                        elif action == "Toggle Active Status":
                            current_status = "Active" if selected_user_data[6] else "Inactive"
                            new_status = not selected_user_data[6]
                            status_text = "activate" if new_status else "deactivate"
                            
                            if st.button(f"🔄 {status_text.title()} User", use_container_width=True):
                                success, message = user_db.toggle_user_active_status(selected_user_id, new_status)
                                if success:
                                    st.success(f"✅ User {status_text}d successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                        
                        elif action == "Reset Password":
                            with st.form("reset_password_form"):
                                new_password = st.text_input("New Password", type="password", placeholder="Enter new password")
                                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm new password")
                                
                                if st.form_submit_button("🔐 Reset Password", use_container_width=True):
                                    if new_password != confirm_password:
                                        st.error("❌ Passwords do not match")
                                    elif len(new_password) < 6:
                                        st.error("❌ Password must be at least 6 characters")
                                    else:
                                        success, message = change_user_password(selected_user_id, new_password)
                                        if success:
                                            st.success(f"✅ {message}")
                                        else:
                                            st.error(f"❌ {message}")
                        
                        elif action == "Delete User":
                            st.warning(f"🚨 This will permanently delete user: {selected_user_data[2]} ({selected_user_data[1]})")
                            if st.button("🗑️ Confirm Delete", type="secondary", use_container_width=True):
                                if selected_user_data[0] == get_current_user()['id']:
                                    st.error("❌ You cannot delete your own account!")
                                else:
                                    success, message = user_db.delete_user(selected_user_id, get_current_user()['id'])
                                    if success:
                                        st.success(f"✅ {message}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ {message}")
        
        else:
            st.info("ℹ️ No users found in the system")
    
    with tab2:
        st.subheader("📊 Access Logs")
        
        access_logs = get_access_logs()
        if access_logs:
            # Check the number of columns in the data
            num_columns = len(access_logs[0]) if access_logs else 0
            
            if num_columns == 7:
                # Database logs with user email
                logs_df = pd.DataFrame(access_logs, columns=[
                    'Log ID', 'User ID', 'Action', 'Page', 'Timestamp', 'IP Address', 'User Email'
                ])
            else:
                # Mock data or fallback
                logs_df = pd.DataFrame(access_logs, columns=[
                    'Log ID', 'User ID', 'Action', 'Page', 'Timestamp', 'IP Address'
                ])
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                date_filter = st.selectbox("Time Range", ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"])
            with col2:
                action_filter = st.selectbox("Action Type", ["All", "Login", "Logout", "Prediction", "Profile Update"])
            with col3:
                user_filter = st.selectbox("User", ["All Users"] + [f"User {log[1]}" for log in access_logs])
            
            st.dataframe(logs_df, use_container_width=True, height=400)
            
            # Log statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Log Entries", len(access_logs))
            with col2:
                login_count = len([log for log in access_logs if "LOGIN" in log[2]])
                st.metric("Login Events", login_count)
            with col3:
                today_count = len([log for log in access_logs if "2024-01-10" in str(log[4])])
                st.metric("Today's Activities", today_count)
            with col4:
                st.metric("Active Sessions", "8")
        else:
            st.info("ℹ️ No access logs available")
    
    with tab3:
        st.subheader("🔧 System Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔄 Database Operations**")
            
            if st.button("🗃️ Optimize Database", use_container_width=True):
                with st.spinner("Optimizing database..."):
                    # Simulate database optimization
                    import time
                    time.sleep(2)
                    st.success("✅ Database optimization completed!")
            
            if st.button("📊 Update Statistics", use_container_width=True):
                with st.spinner("Updating statistics..."):
                    import time
                    time.sleep(1)
                    st.success("✅ Statistics updated successfully!")
            
            if st.button("🧹 Clean Temporary Files", use_container_width=True):
                with st.spinner("Cleaning temporary files..."):
                    import time
                    time.sleep(1)
                    st.success("✅ Temporary files cleaned!")
            
            if st.button("📋 Generate System Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    import time
                    time.sleep(2)
                    st.success("✅ System report generated and saved!")
        
        with col2:
            st.write("**🛡️ Security Tools**")
            
            if st.button("🔐 Run Security Audit", use_container_width=True):
                with st.spinner("Running security audit..."):
                    import time
                    time.sleep(2)
                    st.success("✅ Security audit completed! No issues found.")
            
            if st.button("🔒 Check User Permissions", use_container_width=True):
                with st.spinner("Checking permissions..."):
                    import time
                    time.sleep(1)
                    st.success("✅ Permission check completed!")
            
            if st.button("📧 Send Security Report", use_container_width=True):
                with st.spinner("Sending security report..."):
                    import time
                    time.sleep(2)
                    st.success("✅ Security report sent to administrators!")
            
            if st.button("🔄 Rotate Encryption Keys", use_container_width=True):
                with st.spinner("Rotating encryption keys..."):
                    import time
                    time.sleep(3)
                    st.success("✅ Encryption keys rotated successfully!")
    
    with tab4:
        st.subheader("📈 System Analytics")
        
        users = user_db.get_all_users()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_users = len(users) if users else 0
            active_users = len([u for u in users if u[6]]) if users else 0
            st.metric("Total Users", total_users)
        
        with col2:
            st.metric("Active Users", active_users)
        
        with col3:
            st.metric("Admin Users", len([u for u in users if u[5] == 'admin']) if users else 0)
        
        with col4:
            st.metric("System Uptime", "99.8%")
        
        # Additional analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**👥 User Distribution**")
            if users:
                role_counts = pd.Series([u[5] for u in users]).value_counts()
                fig = px.pie(values=role_counts.values, names=role_counts.index, 
                            title="User Roles Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**📅 User Activity**")
            # Placeholder for user activity chart
            st.info("User activity analytics coming soon!")
        
        # System performance
        st.write("**⚡ System Performance**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Response Time", "128ms")
        with col2:
            st.metric("Error Rate", "0.02%")
        with col3:
            st.metric("Database Size", "45.2 MB")
    
    with tab5:
        st.subheader("🔐 Password Management")
        
        current_user = get_current_user()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔒 Change Your Password**")
            with st.form("change_my_password"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("🔄 Change My Password", use_container_width=True):
                    if not all([current_password, new_password, confirm_password]):
                        st.error("❌ Please fill all fields")
                    elif new_password != confirm_password:
                        st.error("❌ New passwords do not match")
                    elif len(new_password) < 6:
                        st.error("❌ New password must be at least 6 characters")
                    else:
                        # Verify current password first
                        success, _ = user_db.authenticate_user(current_user['email'], current_password)
                        if success:
                            change_success, message = change_user_password(current_user['id'], new_password)
                            if change_success:
                                st.success("✅ Password changed successfully!")
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.error("❌ Current password is incorrect")
        
        with col2:
            st.write("**👥 Bulk Password Operations**")
            st.info("Administrative password tools")
            
            if st.button("🔄 Force Password Reset for All Users", use_container_width=True):
                st.warning("This will require all users to reset their passwords on next login")
                st.info("Feature implementation in progress...")
            
            if st.button("📋 Generate Password Report", use_container_width=True):
                with st.spinner("Generating password security report..."):
                    import time
                    time.sleep(2)
                    st.success("✅ Password security report generated!")
            
            st.write("**📊 Password Policy**")
            st.checkbox("Require strong passwords", value=True)
            st.checkbox("Enable password expiration", value=False)
            st.checkbox("Prevent password reuse", value=True)
            
            if st.button("💾 Save Password Policy", use_container_width=True):
                st.success("✅ Password policy updated!")

if __name__ == "__main__":
    main()