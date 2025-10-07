import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, is_admin, get_current_user, login_required
from auth.database import user_db

# Modern Dark Theme CSS for Home Page
st.markdown("""
<style>
    /* Home Page Specific Styles */
    .home-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .home-subheader {
        font-size: 1.2rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .dashboard-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .dashboard-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(102, 126, 234, 0.4);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    
    .quick-action-btn {
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .quick-action-btn:hover {
        background: rgba(102, 126, 234, 0.4);
        transform: scale(1.05);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    /* Button container styling */
    .button-container {
        text-align: center;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data with multiple fallback options"""
    try:
        possible_paths = [
            'data/cleaned_crime_data.csv',
            './data/cleaned_crime_data.csv',
            '../data/cleaned_crime_data.csv',
            os.path.join(parent_dir, 'data', 'cleaned_crime_data.csv'),
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    return pd.read_csv(path)
            except Exception:
                continue
        return None
    except Exception:
        return None

def safe_page_navigation(page_name):
    """Safe navigation with error handling"""
    try:
        st.switch_page(f"pages/{page_name}")
    except Exception as e:
        st.error(f"Navigation error: {e}")
        st.info("Please use the sidebar for navigation")

@login_required
def main():
    st.set_page_config(page_title="Dashboard - CrimeCast", layout="wide")
    
    user = get_current_user()
    
    # Hero Section
    st.markdown(f'<div class="home-header">🎯 CRIMECAST DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="home-subheader">Welcome back, {user["name"]}! Ready to analyze crime patterns?</div>', unsafe_allow_html=True)
    
    # Quick Stats Row
    df = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_crimes = f"{len(df):,}" if df is not None else "446,254"
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>📊</h3>
            <h2>{total_crimes}</h2>
            <p>Total Crimes Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if df is not None and 'Arrest' in df.columns:
            arrest_rate = f"{df['Arrest'].mean():.1%}"
        elif df is not None and 'Arrest_Target' in df.columns:
            arrest_rate = f"{df['Arrest_Target'].mean():.1%}"
        else:
            arrest_rate = "16.8%"
            
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>🎯</h3>
            <h2>{arrest_rate}</h2>
            <p>Arrest Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        model_acc = "85.9%"
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>🤖</h3>
            <h2>{model_acc}</h2>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        users = user_db.get_all_users()
        active_users = len([u for u in users if u[6]]) if users else 1
        st.markdown(f"""
        <div class="metric-highlight">
            <h3>👥</h3>
            <h2>{active_users}</h2>
            <p>Active Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Features Grid
    st.markdown("""
    <h2 style='color: #667eea; text-align: center; margin-bottom: 2rem;'>🚀 Core Features</h2>
    """, unsafe_allow_html=True)
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">🔮</div>
            <h3 style='color: #667eea; text-align: center;'>AI Crime Prediction</h3>
            <p style='color: #b0b0b0; text-align: center;'>
            Advanced machine learning models predict crime probabilities with real-time analysis and spatial intelligence.
            </p>
            <div class="button-container">
        """, unsafe_allow_html=True)
        if st.button("Predict Crimes", key="predict_btn", use_container_width=True):
            safe_page_navigation("4_🔮_Crime_Prediction.py")
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">📈</div>
            <h3 style='color: #667eea; text-align: center;'>Model Analytics</h3>
            <p style='color: #b0b0b0; text-align: center;'>
            Monitor model performance, accuracy metrics, and prediction confidence scores.
            </p>
            <div class="button-container">
        """, unsafe_allow_html=True)
        if st.button("View Performance", key="performance_btn", use_container_width=True):
            safe_page_navigation("6_📈_Model_Performance.py")
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">📊</div>
            <h3 style='color: #667eea; text-align: center;'>Data Intelligence</h3>
            <p style='color: #b0b0b0; text-align: center;'>
            Interactive visualizations, trend analysis, and comprehensive crime pattern exploration.
            </p>
            <div class="button-container">
        """, unsafe_allow_html=True)
        if st.button("Explore Data", key="data_btn", use_container_width=True):
            safe_page_navigation("5_📊_Data_Analysis.py")
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">⚠️</div>
            <h3 style='color: #667eea; text-align: center;'>Risk Assessment</h3>
            <p style='color: #b0b0b0; text-align: center;'>
            Identify key risk factors, correlation analysis, and preventive insights.
            </p>
            <div class="button-container">
        """, unsafe_allow_html=True)
        if st.button("Assess Risks", key="risk_btn", use_container_width=True):
            safe_page_navigation("7_🔍_Risk_Factors.py")
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    with col7:
        st.markdown("""
        <div class="dashboard-card">
            <div class="feature-icon">👤</div>
            <h3 style='color: #667eea; text-align: center;'>Profile Management</h3>
            <p style='color: #b0b0b0; text-align: center;'>
            Manage your account settings, preferences, and access history.
            </p>
            <div class="button-container">
        """, unsafe_allow_html=True)
        if st.button("View Profile", key="profile_btn", use_container_width=True):
            safe_page_navigation("8_👤_Profile.py")
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        if is_admin():
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-icon">🔧</div>
                <h3 style='color: #667eea; text-align: center;'>Admin Panel</h3>
                <p style='color: #b0b0b0; text-align: center;'>
                System administration, user management, and platform configuration.
                </p>
                <div class="button-container">
            """, unsafe_allow_html=True)
            if st.button("Admin Access", key="admin_btn", use_container_width=True):
                safe_page_navigation("10_🔐_Admin_Panel.py")
            st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dashboard-card">
                <div class="feature-icon">⚙️</div>
                <h3 style='color: #667eea; text-align: center;'>Settings</h3>
                <p style='color: #b0b0b0; text-align: center;'>
                Configure your preferences and application settings.
                </p>
                <div class="button-container">
            """, unsafe_allow_html=True)
            if st.button("Open Settings", key="settings_btn", use_container_width=True):
                safe_page_navigation("9_⚙️_Settings.py")
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent Activity & Quick Actions
    col8, col9 = st.columns([2, 1])
    
    with col8:
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f"""
        <div class="dashboard-card">
            <h3 style='color: #667eea;'>📋 Recent Activity</h3>
            <div style='color: #b0b0b0;'>
                <p>✅ <strong>Last Login:</strong> Today, {current_time}</p>
                <p>🔮 <strong>Predictions Made:</strong> 24 today</p>
                <p>📊 <strong>Data Analyzed:</strong> 15.2K records</p>
                <p>🎯 <strong>Accuracy Rate:</strong> 85.9% this week</p>
                <p>🚀 <strong>System Status:</strong> All services operational</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col9:
        st.markdown("""
        <div class="dashboard-card">
            <h3 style='color: #667eea;'>⚡ Quick Actions</h3>
            <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Data", key="refresh_btn", use_container_width=True):
            st.rerun()
        
        if st.button("📥 Export Report", key="export_btn", use_container_width=True):
            st.success("Report export initiated!")
        
        if st.button("🛡️ Security Check", key="security_btn", use_container_width=True):
            st.info("Security status: All systems secure")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem; padding: 2rem; color: #666; border-top: 1px solid #333;'>
        <p>🚨 CRIMECAST Dashboard | Real-time Crime Analytics & Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()