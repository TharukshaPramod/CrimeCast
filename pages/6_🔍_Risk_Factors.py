import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, is_admin, get_current_user, login_required
from src.predictor import CrimePredictorAPI

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

def load_predictor():
    """Load predictor"""
    try:
        return CrimePredictorAPI(
            model_path="models/best_model_arrest.pkl",
            feature_scaler_path="models/feature_scaler.pkl",
            label_encoders_path="models/label_encoders.pkl"
        )
    except Exception:
        return None

@login_required
def main():
    st.set_page_config(page_title="Risk Factors - CrimeCast", layout="wide")
    
    st.header("🔍 Risk Factor Analysis")
    
    df = load_data()
    predictor = load_predictor()
    
    if df is None or predictor is None:
        st.error("❌ Required data or models not loaded.")
        return
    
    st.markdown("Analyze what factors most influence arrest probabilities.")
    
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
                        orientation='h', title='Feature Importance in Arrest Prediction',
                        color='Importance', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            
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
    
    if st.button("📈 Analyze Risk Factor Impact"):
        with st.spinner("Calculating risk patterns..."):
            baseline_features = {
                'Latitude': 41.8781, 'Longitude': -87.6298, 'Beat': 1032,
                'District': 10, 'Ward': 23, 'Community Area': 32,
                'Hour': 12, 'DayOfWeek': 2, 'Month': 6, 'Year': 2020,
                'Location_Description_Clean': "STREET", 'TimeOfDay': "Afternoon", 'Season': "Summer"
            }
            
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
                        color=probabilities, color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()