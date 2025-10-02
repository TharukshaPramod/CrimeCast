import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_features(X, y, categorical_features):
    """Preprocess features for machine learning"""
    
    # Handle categorical variables
    X_processed = X.copy()
    label_encoders = {}
    
    for col in categorical_features:
        if col in X_processed.columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le
    
    # Scale numerical features
    numerical_features = X_processed.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_processed[numerical_features] = scaler.fit_transform(X_processed[numerical_features])
    
    return X_processed, label_encoders, scaler

def create_feature_set(df, target_type='arrest'):
    """Create complete feature set for modeling"""
    from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLES
    
    # Select features
    features = [f for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if f in df.columns]
    target = TARGET_VARIABLES[target_type]
    
    X = df[features]
    y = df[target]
    
    # Preprocess
    X_processed, label_encoders, scaler = preprocess_features(X, y, CATEGORICAL_FEATURES)
    
    return X_processed, y, label_encoders, scaler