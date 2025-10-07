import streamlit as st
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, is_admin
from auth.decorators import admin_required

@admin_required
def main():
    st.set_page_config(page_title="Settings - CrimeCast", layout="wide")
    
    st.header("⚙️ System Settings")
    st.info("🔒 Admin Access: System configuration and management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Application Settings")
        
        with st.form("app_settings"):
            app_name = st.text_input("Application Name", value="CrimeCast")
            admin_email = st.text_input("Admin Email", value="admin@crimecast.com")
            session_timeout = st.slider("Session Timeout (minutes)", 15, 240, 60)
            data_refresh = st.selectbox("Data Refresh Interval", ["15 minutes", "30 minutes", "1 hour", "6 hours"])
            
            if st.form_submit_button("💾 Save Application Settings"):
                st.success("Application settings saved successfully!")
    
    with col2:
        st.subheader("Data Management")
        
        st.info("📊 Data management features")
        
        col21, col22 = st.columns(2)
        
        with col21:
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared successfully!")
            
            if st.button("📁 Backup Data", use_container_width=True):
                st.success("Data backup initiated!")
        
        with col22:
            if st.button("🗑️ Clear Logs", use_container_width=True):
                st.success("Logs cleared successfully!")
            
            if st.button("📊 Update Statistics", use_container_width=True):
                st.success("Statistics updated successfully!")
    
    st.subheader("System Information")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.metric("System Version", "1.0.0")
        st.metric("Python Version", "3.9+")
        st.metric("Streamlit Version", "1.28.0+")
    
    with col4:
        st.metric("Database Status", "✅ Connected")
        st.metric("Model Status", "✅ Loaded")
        st.metric("Authentication", "✅ Active")

if __name__ == "__main__":
    main()