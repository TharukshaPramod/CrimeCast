import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, get_current_user
from auth.decorators import login_required
from src.predictor import CrimePredictorAPI
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def geocode_address(address):
    """Convert address to latitude/longitude coordinates"""
    try:
        geolocator = Nominatim(user_agent="chicago_crime_predictor")
        location = geolocator.geocode(address + ", Chicago, IL, USA")
        
        if location:
            return location.latitude, location.longitude, None
        else:
            return None, None, "Address not found in Chicago area"
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        return None, None, f"Geocoding service error: {str(e)}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def load_models():
    """Load trained models"""
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

@st.cache_data
def load_data():
    """Load data"""
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

def get_feature_importance(predictor, features):
    """Get feature importance for the prediction"""
    try:
        # Mock feature importance (replace with actual model feature importance)
        importance_data = {
            'Location Type': 0.25,
            'Time of Day': 0.20,
            'Day of Week': 0.15,
            'District': 0.12,
            'Community Area': 0.10,
            'Hour': 0.08,
            'Season': 0.05,
            'Year': 0.03,
            'Month': 0.02
        }
        
        # Adjust based on actual input values
        if features.get('TimeOfDay') == 'Night':
            importance_data['Time of Day'] += 0.10
        if features.get('Location_Description_Clean') in ['ALLEY', 'PARKING LOT']:
            importance_data['Location Type'] += 0.08
            
        return importance_data
    except Exception as e:
        return {
            'Location Type': 0.3,
            'Time of Day': 0.25,
            'Day of Week': 0.2,
            'District': 0.15,
            'Community Area': 0.1
        }

@login_required
def main():
    st.set_page_config(page_title="Crime Prediction - CrimeCast", layout="wide")
    
    st.header("🔮 Crime Probability Predictor")
    
    predictor = load_models()
    df = load_data()
    
    if predictor is None:
        st.error("❌ Models not loaded. Please ensure models are trained.")
        return
    
    st.markdown("Provide the details below to get a real-time prediction of arrest probability.")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location Information")
        
        address = st.text_input("Address or Location", 
                               value="1060 W Addison St",
                               help="Enter a Chicago address")
        
        if st.button("🔍 Get Coordinates", key="geocode_btn"):
            with st.spinner("Finding location..."):
                lat, lon, error = geocode_address(address)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.session_state['latitude'] = lat
                    st.session_state['longitude'] = lon
                    st.success(f"✓ Found: {lat:.4f}°N, {lon:.4f}°W")
        
        latitude = st.number_input("Latitude", 
                                  value=st.session_state.get('latitude', 41.8781), 
                                  min_value=41.6, max_value=42.1, format="%.6f")
        longitude = st.number_input("Longitude", 
                                   value=st.session_state.get('longitude', -87.6298), 
                                   min_value=-87.95, max_value=-87.5, format="%.6f")
        beat = st.number_input("Beat", value=1032, min_value=111, max_value=2535)
        district = st.number_input("District", value=10, min_value=1, max_value=31)
    
    with col2:
        st.subheader("🏛️ Area Information")
        ward = st.number_input("Ward", value=23, min_value=1, max_value=50)
        community_area = st.number_input("Community Area", value=32, min_value=1, max_value=77)
        location_type = st.selectbox("Location Type", 
                                   ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER", 
                                    "PARKING LOT", "ALLEY", "COMMERCIAL", "SCHOOL"])
    
    with col3:
        st.subheader("⏰ Time Information")
        year = st.number_input("Year", value=2020, min_value=2001, max_value=2023)
        month = st.slider("Month", 1, 12, 6)
        hour = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox("Day of Week", 
                                 ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                  "Friday", "Saturday", "Sunday"])
    
    # Auto-calculate derived features
    st.subheader("🔄 Calculated Features")
    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
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
        if month in [12, 1, 2]:
            season = "Winter"
        elif month in [3, 4, 5]:
            season = "Spring"
        elif month in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"
        st.info(f"**Season**: {season}")
    
    day_of_week_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
    
    # Prediction section
    st.markdown("---")
    st.subheader("🎯 Get Prediction")
    
    if st.button("🔮 Predict Arrest Probability", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing crime patterns with AI..."):
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
            
            result = predictor.predict(features)
            
            if 'error' not in result:
                st.success("🎉 Prediction Completed!")
                
                # Display results
                st.subheader("📊 Prediction Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    prob_display = f"{result['probability']:.1%}"
                    st.metric("Arrest Probability", prob_display)
                
                with col2:
                    prediction_text = "🚨 ARREST" if result['prediction'] == 1 else "✅ NO ARREST"
                    st.metric("Prediction", prediction_text)
                
                with col3:
                    st.metric("Risk Level", result['risk_level'])
                
                with col4:
                    confidence = result['probability'] * 100
                    st.metric("Confidence Score", f"{confidence:.1f}%")
                
                # Visualization
                st.subheader("📈 Probability Visualization")
                fig, ax = plt.subplots(figsize=(12, 3))
                
                color_map = {'High': '#ff4b4b', 'Medium': '#ffa500', 'Low': '#00cc96'}
                bar_color = color_map.get(result['risk_level'], '#1f77b4')
                
                bars = ax.barh(['Arrest Probability'], [result['probability']], 
                              color=bar_color, alpha=0.8, height=0.6)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability Scale')
                ax.set_title(f'Arrest Probability: {result["probability"]:.1%}', fontweight='bold')
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                           f'{width:.1%}', ha='left', va='center', 
                           fontweight='bold', fontsize=16, color=bar_color)
                
                ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Low Risk')
                ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='High Risk')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # KEY FACTORS SECTION - RESTORED
                st.subheader("🔑 Key Factors Influencing This Prediction")
                
                # Get feature importance
                feature_importance = get_feature_importance(predictor, features)
                
                # Display key factors
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    st.markdown("### 📊 Feature Importance")
                    # Create horizontal bar chart for feature importance
                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    features_sorted = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                    
                    y_pos = np.arange(len(features_sorted))
                    ax_imp.barh(y_pos, list(features_sorted.values()), color='#667eea', alpha=0.8)
                    ax_imp.set_yticks(y_pos)
                    ax_imp.set_yticklabels(list(features_sorted.keys()))
                    ax_imp.set_xlabel('Importance Score')
                    ax_imp.set_title('Key Factors Influencing Prediction')
                    ax_imp.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels on bars
                    for i, v in enumerate(features_sorted.values()):
                        ax_imp.text(v + 0.01, i, f'{v:.2f}', va='center', fontweight='bold')
                    
                    st.pyplot(fig_imp)
                
                with col_f2:
                    st.markdown("### 💡 Risk Indicators")
                    
                    # Risk factors analysis
                    risk_factors = []
                    
                    # Analyze time factors
                    if time_of_day == 'Night':
                        risk_factors.append("🌙 **Night Time**: Higher risk period")
                    elif time_of_day == 'Evening':
                        risk_factors.append("🌆 **Evening Hours**: Moderate risk period")
                    
                    # Analyze location factors
                    if location_type in ['ALLEY', 'PARKING LOT']:
                        risk_factors.append("📍 **High-Risk Location**: Areas with limited visibility")
                    elif location_type in ['RESIDENCE', 'SCHOOL']:
                        risk_factors.append("🏠 **Controlled Environment**: Lower risk location")
                    
                    # Analyze day factors
                    if day_of_week in ['Friday', 'Saturday']:
                        risk_factors.append("🎉 **Weekend**: Higher activity periods")
                    
                    # Analyze seasonal factors
                    if season == 'Summer':
                        risk_factors.append("☀️ **Summer Season**: Typically higher crime rates")
                    
                    # Display risk factors
                    if risk_factors:
                        for factor in risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.info("📊 No significant risk factors identified for this scenario.")
                    
                    # Recommendations
                    st.markdown("### 🛡️ Safety Recommendations")
                    if result['risk_level'] == 'High':
                        st.warning("""
                        - 🚨 Consider increased patrol presence
                        - 📱 Ensure emergency contacts are accessible
                        - 🎯 Focus on high-visibility policing
                        - 🔄 Monitor the area frequently
                        """)
                    elif result['risk_level'] == 'Medium':
                        st.info("""
                        - 👀 Maintain regular patrol patterns
                        - 📊 Continue standard monitoring
                        - 🔔 Stay alert for unusual activity
                        - 📝 Document any incidents
                        """)
                    else:
                        st.success("""
                        - ✅ Normal patrol operations sufficient
                        - 📋 Continue community engagement
                        - 🎓 Focus on preventive measures
                        - 🌟 Maintain current safety protocols
                        """)
                
            else:
                st.error(f"❌ Prediction error: {result['error']}")

if __name__ == "__main__":
    main()