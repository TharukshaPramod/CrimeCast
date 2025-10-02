import pandas as pd
import numpy as np
from config import DATA_PATH

def load_data():
    """Load the cleaned crime dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found at {DATA_PATH}")
        return None

def get_feature_target(df, target_type='arrest'):
    """Extract features and target variable"""
    from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLES
    
    # Select available features
    available_numerical = [f for f in NUMERICAL_FEATURES if f in df.columns]
    available_categorical = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    
    # Combine features
    features = available_numerical + available_categorical
    
    # Get target variable
    target = TARGET_VARIABLES.get(target_type)
    
    if target not in df.columns:
        print(f"❌ Target variable {target} not found in data")
        return None, None
    
    X = df[features]
    y = df[target]
    
    print(f"✅ Features: {len(features)}, Target: {target}")
    return X, y

def split_data_temporal(df, split_year=2020):
    """Split data based on time (temporal validation)"""
    train_data = df[df['Year'] < split_year]
    test_data = df[df['Year'] >= split_year]
    
    print(f"✅ Temporal split - Train: {len(train_data)}, Test: {len(test_data)}")
    return train_data, test_data