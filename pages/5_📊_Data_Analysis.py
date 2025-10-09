import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, is_admin, get_current_user, login_required

@st.cache_data
def load_data():
    """Load data with error handling"""
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
                    df = pd.read_csv(path)
                    st.success(f"✅ Data loaded successfully: {len(df)} records")
                    return df
            except Exception as e:
                st.warning(f"⚠️ Could not load from {path}: {e}")
                continue
        
        st.error("❌ No data file found in any expected location")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def safe_plotly_chart(fig, **kwargs):
    """Safely display Plotly chart with error handling"""
    try:
        if fig:
            st.plotly_chart(fig, use_container_width=True, **kwargs)
        else:
            st.warning("Could not generate chart")
    except Exception as e:
        st.error(f"Chart error: {e}")

@login_required
def main():
    st.set_page_config(
        page_title="Data Analysis - CrimeCast", 
        page_icon="📊",
        layout="wide"
    )
    
    st.header("📊 Advanced Crime Data Analysis")
    
    # Load data with progress indicator
    with st.spinner("Loading crime data..."):
        df = load_data()
    
    if df is None:
        st.error("""
        ❌ Data not available. Please ensure:
        - Data file exists in the data/ folder
        - File is named 'cleaned_crime_data.csv'
        - File contains valid CSV data
        """)
        return
    
    # Show basic data info
    with st.expander("📋 Dataset Overview"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            date_range = f"{df['Year'].min()}-{df['Year'].max()}" if 'Year' in df.columns else "N/A"
            st.metric("Date Range", date_range)
        with col4:
            arrest_rate = f"{df['Arrest'].mean():.1%}" if 'Arrest' in df.columns else "N/A"
            st.metric("Arrest Rate", arrest_rate)
    
    analysis_type = st.selectbox(
        "Select Analysis Type", 
        ["Crime Trends", "Spatial Distribution", "Crime Types", 
         "Temporal Patterns", "Arrest Analysis", "Geographic Insights"]
    )
    
    st.markdown("---")
    
    if analysis_type == "Crime Trends":
        st.subheader("📈 Crime Trends Over Time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Year' in df.columns:
                yearly_data = df['Year'].value_counts().sort_index()
                fig = px.line(
                    x=yearly_data.index, 
                    y=yearly_data.values,
                    title='Crime Trends by Year',
                    labels={'x': 'Year', 'y': 'Number of Crimes'}
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Year data not available")
        
        with col2:
            if 'Month' in df.columns:
                monthly_data = df['Month'].value_counts().sort_index()
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                fig = px.bar(
                    x=months, 
                    y=[monthly_data.get(i, 0) for i in range(1, 13)],
                    title='Crime Distribution by Month'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Month data not available")
    
    elif analysis_type == "Spatial Distribution":
        st.subheader("🗺️ Spatial Crime Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'District' in df.columns:
                district_data = df['District'].value_counts().sort_index()
                fig = px.bar(
                    x=district_data.index, 
                    y=district_data.values,
                    title='Crimes by Police District',
                    labels={'x': 'District', 'y': 'Number of Crimes'}
                )
                safe_plotly_chart(fig)
            else:
                st.warning("District data not available")
        
        with col2:
            if 'Community Area' in df.columns:
                community_data = df['Community Area'].value_counts().head(15)
                fig = px.bar(
                    x=community_data.values, 
                    y=community_data.index,
                    orientation='h',
                    title='Top 15 Community Areas by Crime Count'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Community Area data not available")
        
        # Geographic scatter plot (safe version)
        st.subheader("📍 Crime Location Distribution")
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            try:
                # Sample for better performance
                sample_df = df.dropna(subset=['Latitude', 'Longitude']).sample(n=min(1000, len(df)))
                
                fig = px.scatter(
                    sample_df,
                    x='Longitude',
                    y='Latitude',
                    title='Crime Locations (Sample)',
                    opacity=0.6
                )
                fig.update_layout(height=400)
                safe_plotly_chart(fig)
            except Exception as e:
                st.warning(f"Could not create location plot: {e}")
    
    elif analysis_type == "Crime Types":
        st.subheader("🔍 Crime Type Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Primary Type' in df.columns:
                crime_types = df['Primary Type'].value_counts().head(15)
                fig = px.bar(
                    x=crime_types.values,
                    y=crime_types.index,
                    orientation='h',
                    title='Top 15 Crime Types'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Primary Type data not available")
        
        with col2:
            if 'Primary Type' in df.columns:
                crime_dist = df['Primary Type'].value_counts().head(10)
                fig = px.pie(
                    values=crime_dist.values,
                    names=crime_dist.index,
                    title='Top 10 Crime Types Distribution'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Primary Type data not available")
    
    elif analysis_type == "Temporal Patterns":
        st.subheader("⏰ Temporal Crime Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Hour' in df.columns:
                hourly_data = df['Hour'].value_counts().sort_index()
                fig = px.bar(
                    x=hourly_data.index,
                    y=hourly_data.values,
                    title='Crimes by Hour of Day'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Hour data not available")
        
        with col2:
            if 'DayOfWeek' in df.columns:
                daily_data = df['DayOfWeek'].value_counts().sort_index()
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                fig = px.bar(
                    x=days,
                    y=[daily_data.get(i, 0) for i in range(7)],
                    title='Crimes by Day of Week'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("DayOfWeek data not available")
    
    elif analysis_type == "Arrest Analysis":
        st.subheader("👮 Arrest Rate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Primary Type' in df.columns and 'Arrest' in df.columns:
                arrest_rates = df.groupby('Primary Type')['Arrest'].mean().sort_values(ascending=False).head(15)
                fig = px.bar(
                    x=arrest_rates.values,
                    y=arrest_rates.index,
                    orientation='h',
                    title='Top 15 Arrest Rates by Crime Type'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Arrest or Primary Type data not available")
        
        with col2:
            if 'Hour' in df.columns and 'Arrest' in df.columns:
                hourly_arrest = df.groupby('Hour')['Arrest'].mean()
                fig = px.line(
                    x=hourly_arrest.index,
                    y=hourly_arrest.values,
                    title='Arrest Rate by Hour of Day'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Arrest or Hour data not available")
    
    elif analysis_type == "Geographic Insights":
        st.subheader("🌍 Geographic Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Ward' in df.columns:
                ward_data = df['Ward'].value_counts().head(15)
                fig = px.pie(
                    values=ward_data.values,
                    names=ward_data.index,
                    title='Crime Distribution by Ward (Top 15)'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Ward data not available")
        
        with col2:
            if 'Beat' in df.columns:
                beat_data = df['Beat'].value_counts().head(15)
                fig = px.bar(
                    x=beat_data.values,
                    y=beat_data.index,
                    orientation='h',
                    title='Top 15 Beats by Crime Count'
                )
                safe_plotly_chart(fig)
            else:
                st.warning("Beat data not available")
        
        # Location type analysis
        st.subheader("🏢 Location Type Analysis")
        if 'Location Description' in df.columns:
            location_data = df['Location Description'].value_counts().head(10)
            fig = px.bar(
                x=location_data.values,
                y=location_data.index,
                orientation='h',
                title='Top 10 Crime Locations'
            )
            safe_plotly_chart(fig)
        else:
            st.warning("Location Description data not available")
    
    # Data quality info
    st.markdown("---")
    with st.expander("🔍 Data Quality Information"):
        st.write("**Available Columns:**", list(df.columns))
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**Missing Values:**")
        missing_data = df.isnull().sum()
        st.write(missing_data[missing_data > 0])

if __name__ == "__main__":
    main()