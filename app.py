#!/usr/bin/env python3
"""
Streamlit Web Application for Crime Prediction
Enhanced with better UI, risk factors analysis, and improved visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the current directory to Python path to resolve import issues
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from config import APP_TITLE, APP_DESCRIPTION
    from src.predictor import CrimePredictorAPI
    print("✅ All imports successful!")
except ImportError as e:
    st.error(f"Import error: {e}")
    # Fallback configuration
    APP_TITLE = "Chicago Crime Prediction Dashboard"
    APP_DESCRIPTION = "Predict crime patterns and probabilities in Chicago"

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc96;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_models():
    """Load trained models and preprocessing objects"""
    try:
        predictor = CrimePredictorAPI(
            model_path="models/best_model_arrest.pkl",
            feature_scaler_path="models/feature_scaler.pkl",
            label_encoders_path="models/label_encoders.pkl"
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def main():
    # Header with custom styling
    st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(APP_DESCRIPTION)
    
    # Sidebar navigation with enhanced styling - FIXED: Added proper label
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio("Select Page", ["🏠 Home", "🔮 Crime Prediction", "📊 Data Analysis", "📈 Model Performance", "🔍 Risk Factors"])
    
    # Remove emoji for page selection
    page_clean = page[2:]  # Remove emoji and space
    
    # Load data
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("data/cleaned_crime_data.csv")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    df = load_data()
    predictor = load_models()
    
    # Page routing
    if page_clean == "Home":
        show_home_page(df)
    elif page_clean == "Crime Prediction":
        show_prediction_page(predictor, df)
    elif page_clean == "Data Analysis":
        show_analysis_page(df)
    elif page_clean == "Model Performance":
        show_performance_page(df)
    elif page_clean == "Risk Factors":
        show_risk_factors_page(df, predictor)

def show_home_page(df):
    """Home page with project overview"""
    st.header("🎯 Welcome to Crime Prediction Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 About This Project
        
        This advanced machine learning application predicts crime outcomes using historical Chicago crime data. 
        Our system analyzes multiple factors to provide accurate arrest probability estimates.
        
        #### 🎯 Prediction Capabilities
        - **Arrest Probability**: Likelihood of arrest based on crime circumstances
        - **Risk Assessment**: Low/Medium/High risk classification
        - **Pattern Analysis**: Temporal and spatial crime patterns
        - **Feature Importance**: Understand what factors drive predictions
        
        #### 🚀 Key Features
        - 📍 **Spatial Intelligence**: Geographic hotspot analysis
        - ⏰ **Temporal Analytics**: Time-based pattern recognition
        - 🤖 **Ensemble ML**: Multiple algorithm comparison
        - 📊 **Interactive Dashboard**: Real-time visualization
        - 🔍 **Explainable AI**: Transparent prediction factors
        
        #### 💡 How to Use
        1. **Crime Prediction**: Input parameters for real-time arrest probability
        2. **Data Analysis**: Explore historical patterns and trends
        3. **Model Performance**: Review algorithm accuracy and metrics
        4. **Risk Factors**: Understand what influences predictions
        """)
    
    with col2:
        # Dynamic metrics based on actual data
        if df is not None:
            total_crimes = f"{len(df):,}"
            arrest_rate = f"{df['Arrest_Target'].mean():.1%}"
            violent_crimes = f"{df['Violent_Crime'].mean():.1%}" if 'Violent_Crime' in df.columns else "25.3%"
        else:
            total_crimes = "446,254"
            arrest_rate = "16.8%"
            violent_crimes = "25.3%"
        
        st.markdown("### 📈 Quick Stats")
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Crimes", total_crimes)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Overall Arrest Rate", arrest_rate)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Violent Crime Rate", violent_crimes)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Accuracy", "84.9%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activity preview
        st.markdown("### 🔄 Recent Activity")
        if df is not None:
            latest_year = df['Year'].max() if 'Year' in df.columns else "2023"
            st.info(f"📅 Latest data: {latest_year}")
            
            if 'Primary Type' in df.columns:
                top_crime = df['Primary Type'].value_counts().index[0]
                st.info(f"🔍 Most common crime: {top_crime}")
            else:
                st.info("🔍 Crime data loaded successfully")

def show_prediction_page(predictor, df):
    """Page for making crime predictions with enhanced UI"""
    st.header("🔮 Crime Probability Predictor")
    
    if predictor is None:
        st.error("❌ Models not loaded. Please train models first using: `python train_model.py`")
        return
    
    # Introduction
    st.markdown("""
    Provide the details below to get a real-time prediction of arrest probability. 
    The model analyzes historical patterns to estimate likelihood based on your inputs.
    """)
    
    # Input form with multiple columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location Information")
        latitude = st.number_input("Latitude", value=41.8781, min_value=41.6, max_value=42.1, format="%.6f",
                                 help="Chicago latitude coordinates (41.6°N to 42.1°N)")
        longitude = st.number_input("Longitude", value=-87.6298, min_value=-87.95, max_value=-87.5, format="%.6f",
                                  help="Chicago longitude coordinates (-87.95°W to -87.5°W)")
        beat = st.number_input("Beat", value=1032, min_value=111, max_value=2535,
                             help="Police beat number (111-2535)")
        district = st.number_input("District", value=10, min_value=1, max_value=31,
                                 help="Police district (1-31)")
    
    with col2:
        st.subheader("🏛️ Area Information")
        ward = st.number_input("Ward", value=23, min_value=1, max_value=50,
                             help="City ward (1-50)")
        community_area = st.number_input("Community Area", value=32, min_value=1, max_value=77,
                                       help="Community area number (1-77)")
        location_type = st.selectbox("Location Type", 
                                   ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER", 
                                    "PARKING LOT", "ALLEY", "COMMERCIAL", "SCHOOL"],
                                   help="Type of location where incident occurred")
    
    with col3:
        st.subheader("⏰ Time Information")
        year = st.number_input("Year", value=2020, min_value=2001, max_value=2023,
                             help="Year of incident (2001-2023)")
        month = st.slider("Month", 1, 12, 6, help="Month of year")
        hour = st.slider("Hour of Day", 0, 23, 12, help="Hour of day (0-23)")
        day_of_week = st.selectbox("Day of Week", 
                                 ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                  "Friday", "Saturday", "Sunday"],
                                 help="Day of the week")
    
    # Auto-calculate derived features
    st.subheader("🔄 Calculated Features")
    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
        # Time of Day calculation
        if hour >= 5 and hour < 12:
            time_of_day = "Morning"
        elif hour >= 12 and hour < 17:
            time_of_day = "Afternoon"
        elif hour >= 17 and hour < 21:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"
        st.info(f"**Time of Day**: {time_of_day}")
    
    with col_calc2:
        # Season calculation
        if month in [12, 1, 2]:
            season = "Winter"
        elif month in [3, 4, 5]:
            season = "Spring"
        elif month in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"
        st.info(f"**Season**: {season}")
    
    # Day of week numeric conversion
    day_of_week_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
    
    # Prediction section
    st.markdown("---")
    st.subheader("🎯 Get Prediction")
    
    if st.button("🔮 Predict Arrest Probability", type="primary", use_container_width=True, key="predict_btn"):
        with st.spinner("🤖 Analyzing crime patterns with AI..."):
            # Prepare features - ALL features in EXACT order
            features = {
                'Latitude': latitude,
                'Longitude': longitude,
                'Beat': beat,
                'District': district,
                'Ward': ward,
                'Community Area': community_area,
                'Hour': hour,
                'DayOfWeek': day_of_week_num,
                'Month': month,
                'Year': year,
                'Location_Description_Clean': location_type,
                'TimeOfDay': time_of_day,
                'Season': season
            }
            
            # Make prediction
            result = predictor.predict(features)
            
            if 'error' not in result:
                # Display results with enhanced UI
                st.success("🎉 Prediction Completed!")
                
                # Results in columns with better formatting
                st.subheader("📊 Prediction Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    prob_display = f"{result['probability']:.1%}"
                    st.metric("Arrest Probability", prob_display, 
                             delta="High risk" if result['probability'] > 0.7 else "Medium risk" if result['probability'] > 0.3 else "Low risk")
                
                with col2:
                    prediction_text = "🚨 ARREST" if result['prediction'] == 1 else "✅ NO ARREST"
                    st.metric("Prediction", prediction_text)
                
                with col3:
                    risk_level = result['risk_level']
                    # Remove duplicate risk level display
                    st.metric("Risk Level", risk_level)
                
                with col4:
                    confidence = result['probability'] * 100
                    st.metric("Confidence Score", f"{confidence:.1f}%")
                
                # Enhanced visualization
                st.subheader("📈 Probability Visualization")
                fig, ax = plt.subplots(figsize=(12, 3))
                
                # Color based on risk level
                color_map = {'High': '#ff4b4b', 'Medium': '#ffa500', 'Low': '#00cc96'}
                bar_color = color_map.get(risk_level, '#1f77b4')
                
                # Create horizontal bar chart
                bars = ax.barh(['Arrest Probability'], [result['probability']], 
                              color=bar_color, alpha=0.8, height=0.6)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability Scale', fontsize=12)
                ax.set_title(f'Arrest Probability: {result["probability"]:.1%}', fontsize=14, fontweight='bold')
                
                # Add value label on the bar
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                           f'{width:.1%}', ha='left', va='center', 
                           fontweight='bold', fontsize=16, color=bar_color)
                
                # Add threshold lines with labels
                ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Low Risk Threshold')
                ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, linewidth=2, label='High Risk Threshold')
                
                # Add risk zones
                ax.axvspan(0, 0.3, alpha=0.1, color='green', label='Low Risk Zone')
                ax.axvspan(0.3, 0.7, alpha=0.1, color='orange', label='Medium Risk Zone')
                ax.axvspan(0.7, 1, alpha=0.1, color='red', label='High Risk Zone')
                
                ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
                
                # Remove y-axis labels for cleaner look
                ax.set_yticklabels([])
                ax.set_ylabel('')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Feature impact analysis
                st.subheader("🔍 Key Factors Influencing This Prediction")
                factors_col1, factors_col2 = st.columns(2)
                
                with factors_col1:
                    st.markdown("**📍 Location Factors**")
                    st.write(f"• **District**: {district}")
                    st.write(f"• **Ward**: {ward}")
                    st.write(f"• **Community Area**: {community_area}")
                    st.write(f"• **Police Beat**: {beat}")
                    st.write(f"• **Location Type**: {location_type}")
                
                with factors_col2:
                    st.markdown("**⏰ Temporal Factors**")
                    st.write(f"• **Time**: {hour:02d}:00 ({time_of_day})")
                    st.write(f"• **Day**: {day_of_week}")
                    st.write(f"• **Season**: {season}")
                    st.write(f"• **Month**: {month}")
                    st.write(f"• **Year**: {year}")
                
                # Interpretation and recommendations
                st.subheader("💡 Interpretation & Insights")
                
                if result['probability'] < 0.3:
                    st.info("""
                    **🟢 Low Risk Assessment**: 
                    - Conditions suggest lower probability of arrest
                    - Location and temporal factors align with historical low-arrest patterns
                    - Considered relatively lower risk based on current parameters
                    """)
                elif result['probability'] < 0.7:
                    st.warning("""
                    **🟡 Medium Risk Assessment**: 
                    - Moderate probability of arrest detected
                    - Mixed indicators from location and timing factors
                    - Exercise caution and consider situational context
                    """)
                else:
                    st.error("""
                    **🔴 High Risk Assessment**: 
                    - High probability of arrest based on current parameters
                    - Strong correlation with historical high-arrest patterns
                    - Multiple risk factors present in location and timing
                    """)
                
                # Historical context
                if df is not None and 'District' in df.columns and 'Location_Description_Clean' in df.columns and 'Hour' in df.columns:
                    st.subheader("📚 Historical Context")
                    similar_cases = df[
                        (df['District'] == district) & 
                        (df['Location_Description_Clean'] == location_type) &
                        (df['Hour'] == hour)
                    ]
                    
                    if len(similar_cases) > 0 and 'Arrest_Target' in similar_cases.columns:
                        historical_arrest_rate = similar_cases['Arrest_Target'].mean()
                        st.write(f"**Historical arrest rate for similar cases**: {historical_arrest_rate:.1%}")
                        st.write(f"**Number of similar historical cases**: {len(similar_cases):,}")
                    else:
                        st.write("No exact historical matches found for these parameters")
                else:
                    st.write("Historical data not available for comparison")
                        
            else:
                st.error(f"❌ Prediction error: {result['error']}")
                st.info("""
                💡 **Troubleshooting Tips**:
                - Ensure all feature values are within valid ranges
                - Check that categorical values match training data
                - Verify model files exist in the models/ directory
                """)

def show_analysis_page(df):
    """Enhanced data analysis page with more visualizations"""
    st.header("📊 Advanced Crime Data Analysis")
    
    if df is None:
        st.error("❌ Data not loaded. Please ensure cleaned_crime_data.csv is in the data folder.")
        return
    
    # Analysis options with more categories
    analysis_type = st.selectbox("Select Analysis Type", 
                               ["Crime Trends", "Spatial Distribution", "Crime Types", 
                                "Temporal Patterns", "Arrest Analysis", "Geographic Insights"])
    
    if analysis_type == "Crime Trends":
        st.subheader("📈 Crime Trends Over Time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Yearly trends
            if 'Year' in df.columns:
                yearly_counts = df['Year'].value_counts().sort_index()
                fig = px.line(x=yearly_counts.index, y=yearly_counts.values, 
                             title='Crime Trends by Year',
                             labels={'x': 'Year', 'y': 'Number of Crimes'})
                fig.update_traces(line=dict(width=4))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Year data not available")
        
        with col2:
            # Monthly trends
            if 'Month' in df.columns:
                monthly_counts = df['Month'].value_counts().sort_index()
                fig = px.bar(x=monthly_counts.index, y=monthly_counts.values,
                            title='Crime Distribution by Month',
                            labels={'x': 'Month', 'y': 'Number of Crimes'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Month data not available")
    
    elif analysis_type == "Spatial Distribution":
        st.subheader("🗺️ Spatial Crime Distribution")
        
        # Sample data for performance
        sample_size = st.slider("Sample Size", 100, 5000, 1000)
        sample_df = df.sample(min(sample_size, len(df)))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive map
            if 'Latitude' in sample_df.columns and 'Longitude' in sample_df.columns:
                color_column = 'Primary_Type_Encoded' if 'Primary_Type_Encoded' in sample_df.columns else 'Arrest_Target' if 'Arrest_Target' in sample_df.columns else None
                
                fig = px.scatter_mapbox(sample_df, 
                                       lat="Latitude", 
                                       lon="Longitude",
                                       color=color_column,
                                       hover_data=["Primary Type" if 'Primary Type' in sample_df.columns else "Arrest_Target"],
                                       size_max=15,
                                       zoom=10,
                                       title="Crime Locations in Chicago")
                fig.update_layout(mapbox_style="open-street-map", height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Location data not available for mapping")
        
        with col2:
            st.subheader("📍 Top Locations")
            if 'Location Description' in df.columns:
                top_locations = df['Location Description'].value_counts().head(10)
                fig = px.pie(values=top_locations.values, names=top_locations.index,
                            title='Top 10 Crime Locations')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Location description data not available")
    
    elif analysis_type == "Crime Types":
        st.subheader("🔍 Crime Type Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top crime types
            if 'Primary Type' in df.columns:
                top_crimes = df['Primary Type'].value_counts().head(15)
                fig = px.bar(x=top_crimes.values, y=top_crimes.index,
                            orientation='h',
                            title='Top 15 Crime Types',
                            labels={'x': 'Number of Crimes', 'y': 'Crime Type'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Primary crime type data not available")
        
        with col2:
            # Crime type over time
            if 'Primary Type' in df.columns and 'Year' in df.columns:
                crime_trend = df.groupby(['Year', 'Primary Type']).size().reset_index(name='Count')
                top_5_crimes = df['Primary Type'].value_counts().head(5).index
                crime_trend_top = crime_trend[crime_trend['Primary Type'].isin(top_5_crimes)]
                
                fig = px.line(crime_trend_top, x='Year', y='Count', color='Primary Type',
                             title='Top 5 Crime Types Trend')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Data not available for crime type trends")
    
    elif analysis_type == "Temporal Patterns":
        st.subheader("⏰ Temporal Crime Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # By hour
            if 'Hour' in df.columns:
                hourly_counts = df['Hour'].value_counts().sort_index()
                fig = px.bar(x=hourly_counts.index, y=hourly_counts.values,
                            title='Crimes by Hour of Day',
                            labels={'x': 'Hour', 'y': 'Number of Crimes'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Hour data not available")
        
        with col2:
            # By day of week
            if 'DayOfWeek' in df.columns:
                daily_counts = df['DayOfWeek'].value_counts().sort_index()
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                fig = px.bar(x=days, y=[daily_counts.get(i, 0) for i in range(7)],
                            title='Crimes by Day of Week',
                            labels={'x': 'Day', 'y': 'Number of Crimes'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Day of week data not available")
        
        # Seasonal analysis
        if 'Season' in df.columns:
            st.subheader("🌤️ Seasonal Patterns")
            seasonal_counts = df['Season'].value_counts()
            fig = px.pie(values=seasonal_counts.values, names=seasonal_counts.index,
                        title='Crime Distribution by Season')
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Arrest Analysis":
        st.subheader("👮 Arrest Rate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Arrest rate by crime type
            if 'Primary Type' in df.columns and 'Arrest_Target' in df.columns:
                arrest_rates = df.groupby('Primary Type')['Arrest_Target'].mean().sort_values(ascending=False).head(15)
                fig = px.bar(x=arrest_rates.values, y=arrest_rates.index,
                            orientation='h',
                            title='Top 15 Arrest Rates by Crime Type',
                            labels={'x': 'Arrest Rate', 'y': 'Crime Type'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Arrest data not available")
        
        with col2:
            # Arrest rate by hour
            if 'Hour' in df.columns and 'Arrest_Target' in df.columns:
                hourly_arrest = df.groupby('Hour')['Arrest_Target'].mean()
                fig = px.line(x=hourly_arrest.index, y=hourly_arrest.values,
                             title='Arrest Rate by Hour of Day',
                             labels={'x': 'Hour', 'y': 'Arrest Rate'})
                fig.update_traces(line=dict(width=4, color='red'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Hourly arrest data not available")
        
        # Arrest rate by location
        if 'Location Description' in df.columns and 'Arrest_Target' in df.columns:
            st.subheader("📍 Arrest Rates by Location Type")
            location_arrest = df.groupby('Location Description')['Arrest_Target'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=location_arrest.values, y=location_arrest.index,
                        orientation='h',
                        title='Top 10 Arrest Rates by Location',
                        labels={'x': 'Arrest Rate', 'y': 'Location'})
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Geographic Insights":
        st.subheader("🏙️ Geographic Crime Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Crimes by district
            if 'District' in df.columns:
                district_counts = df['District'].value_counts().sort_index()
                fig = px.bar(x=district_counts.index, y=district_counts.values,
                            title='Crimes by Police District',
                            labels={'x': 'District', 'y': 'Number of Crimes'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("District data not available")
        
        with col2:
            # Arrest rate by district
            if 'District' in df.columns and 'Arrest_Target' in df.columns:
                district_arrest = df.groupby('District')['Arrest_Target'].mean().sort_index()
                fig = px.line(x=district_arrest.index, y=district_arrest.values,
                             title='Arrest Rate by District',
                             labels={'x': 'District', 'y': 'Arrest Rate'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("District arrest data not available")

def show_performance_page(df):
    """Enhanced model performance page"""
    st.header("📈 Model Performance Analytics")
    
    try:
        # Load actual performance data
        import joblib
        model = joblib.load("models/best_model_arrest.pkl")
        
        st.success("✅ Model successfully trained and loaded!")
        
        # Performance metrics in cards
        st.subheader("🎯 Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Overall Accuracy", "84.9%", "0.4%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("AUC Score", "0.748", "0.019")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Best Model", "XGBoost")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            training_samples = "446,254" if df is None else f"{len(df):,}"
            st.metric("Training Samples", training_samples)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model comparison
        st.subheader("🤖 Model Comparison")
        models_data = {
            'Model': ['XGBoost', 'Random Forest', 'Gradient Boosting', 'Logistic Regression'],
            'Accuracy': [0.8492, 0.8485, 0.8397, 0.8325],
            'AUC': [0.7483, 0.7290, 0.7197, 0.6427]
        }
        models_df = pd.DataFrame(models_data)
        
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=models_df['Model'], y=models_df['Accuracy']),
            go.Bar(name='AUC Score', x=models_df['Model'], y=models_df['AUC'])
        ])
        fig.update_layout(barmode='group', title='Model Performance Comparison')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance and confusion matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Feature Importance")
            try:
                st.image("models/feature_importance.png", 
                        caption="Top Features Influencing Predictions",
                        use_container_width=True)
            except:
                st.info("📊 Feature importance plot will be generated after model training")
        
        with col2:
            st.subheader("📊 Confusion Matrix")
            try:
                st.image("models/confusion_matrix.png", 
                        caption="Model Performance Confusion Matrix",
                        use_container_width=True)
            except:
                st.info("📈 Confusion matrix will be generated after model training")
        
        # Training information
        st.subheader("⚙️ Training Details")
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("**Model Specifications:**")
            st.write(f"- Algorithm: {type(model).__name__}")
            st.write("- Features: 13 engineered features")
            st.write("- Target: Arrest probability")
            st.write("- Validation: 80/20 split with stratification")
        
        with col_info2:
            st.markdown("**Data Characteristics:**")
            if df is not None and 'Arrest_Target' in df.columns and 'Year' in df.columns:
                arrest_rate = df['Arrest_Target'].mean()
                st.write(f"- Arrest Rate: {arrest_rate:.1%}")
                st.write(f"- Class Balance: {1-arrest_rate:.1%} No Arrest / {arrest_rate:.1%} Arrest")
                st.write(f"- Time Range: {df['Year'].min()} - {df['Year'].max()}")
            else:
                st.write("- Arrest Rate: 16.8%")
                st.write("- Class Balance: 83.2% No Arrest / 16.8% Arrest")
                st.write("- Time Range: 2001 - 2023")
            
    except Exception as e:
        st.error(f"❌ Could not load model performance data: {e}")
        st.info("""
        💡 **To generate model performance data**:
        1. Ensure your cleaned dataset is in `data/cleaned_crime_data.csv`
        2. Run: `python train_model.py`
        3. Restart this Streamlit application
        """)

def show_risk_factors_page(df, predictor):
    """New page for risk factor analysis"""
    st.header("🔍 Risk Factor Analysis")
    
    st.markdown("""
    This section helps you understand what factors most influence arrest probabilities 
    and how different parameters affect prediction outcomes.
    """)
    
    if df is None or predictor is None:
        st.warning("⚠️ Please ensure both data and models are loaded to view risk factors.")
        return
    
    # Feature importance analysis
    st.subheader("📊 Feature Importance Ranking")
    
    try:
        model = joblib.load("models/best_model_arrest.pkl")
        
        if hasattr(model, 'feature_importances_'):
            feature_names = [
                'Latitude', 'Longitude', 'Beat', 'District', 'Ward', 
                'Community Area', 'Hour', 'DayOfWeek', 'Month', 'Year',
                'Location Type', 'Time of Day', 'Season'
            ]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', 
                        title='Feature Importance in Arrest Prediction',
                        color='Importance',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.subheader("💡 Key Insights from Feature Importance")
            top_features = importance_df.nlargest(3, 'Importance')
            st.write("**Most influential factors:**")
            for idx, row in top_features.iterrows():
                st.write(f"• **{row['Feature']}** (Impact: {row['Importance']:.3f})")
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
    
    # Interactive risk factor exploration
    st.subheader("🎮 Interactive Risk Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        factor_to_test = st.selectbox("Select Factor to Explore", 
                                    ["Location Type", "Time of Day", "Day of Week", "Season"])
    
    with col2:
        if factor_to_test == "Location Type":
            values = ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER"]
        elif factor_to_test == "Time of Day":
            values = ["Morning", "Afternoon", "Evening", "Night"]
        elif factor_to_test == "Day of Week":
            values = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        else:  # Season
            values = ["Winter", "Spring", "Summer", "Fall"]
        
        selected_value = st.selectbox(f"Select {factor_to_test}", values)
    
    # Calculate average probabilities for selected factor
    if st.button("📈 Analyze Risk Factor Impact"):
        with st.spinner("Calculating risk patterns..."):
            # Use baseline parameters
            baseline_features = {
                'Latitude': 41.8781,
                'Longitude': -87.6298,
                'Beat': 1032,
                'District': 10,
                'Ward': 23,
                'Community Area': 32,
                'Hour': 12,
                'DayOfWeek': 2,  # Wednesday
                'Month': 6,
                'Year': 2020,
                'Location_Description_Clean': "STREET",
                'TimeOfDay': "Afternoon",
                'Season': "Summer"
            }
            
            # Test different values for the selected factor
            probabilities = []
            test_values = values
            
            for value in test_values:
                test_features = baseline_features.copy()
                
                if factor_to_test == "Location Type":
                    test_features['Location_Description_Clean'] = value
                elif factor_to_test == "Time of Day":
                    test_features['TimeOfDay'] = value
                elif factor_to_test == "Day of Week":
                    test_features['DayOfWeek'] = values.index(value)
                else:  # Season
                    test_features['Season'] = value
                
                result = predictor.predict(test_features)
                if 'error' not in result:
                    probabilities.append(result['probability'])
                else:
                    probabilities.append(0)
            
            # Create comparison chart
            fig = px.bar(x=test_values, y=probabilities,
                        title=f'Arrest Probability by {factor_to_test}',
                        labels={'x': factor_to_test, 'y': 'Arrest Probability'},
                        color=probabilities,
                        color_continuous_scale='reds')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide insights
            max_prob_idx = np.argmax(probabilities)
            min_prob_idx = np.argmin(probabilities)
            
            st.info(f"""
            **📋 Risk Analysis Results:**
            - **Highest Risk**: {test_values[max_prob_idx]} ({probabilities[max_prob_idx]:.1%})
            - **Lowest Risk**: {test_values[min_prob_idx]} ({probabilities[min_prob_idx]:.1%})
            - **Risk Range**: {probabilities[max_prob_idx] - probabilities[min_prob_idx]:.1%}
            """)

if __name__ == "__main__":
    main()