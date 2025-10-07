import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from auth.auth_utils import is_authenticated, is_admin
from auth.decorators import admin_required

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

@admin_required
def main():
    st.set_page_config(page_title="Model Performance - CrimeCast", layout="wide")
    
    st.header("📈 Model Performance Analytics")
    st.info("🔒 Admin Access: Model performance and analytics")
    
    df = load_data()
    
    try:
        model = joblib.load("models/best_model_arrest.pkl")
        st.success("✅ Model successfully loaded!")
        
        # Performance metrics
        st.subheader("🎯 Key Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Accuracy", "84.9%", "0.4%")
        
        with col2:
            st.metric("AUC Score", "0.748", "0.019")
        
        with col3:
            st.metric("Best Model", "XGBoost")
        
        with col4:
            training_samples = "446,254" if df is None else f"{len(df):,}"
            st.metric("Training Samples", training_samples)
        
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
        
    except Exception as e:
        st.error(f"❌ Could not load model performance data: {e}")

if __name__ == "__main__":
    main()