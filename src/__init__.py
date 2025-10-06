# src/__init__.py
from .predictor import CrimePredictorAPI
from .data_loader import load_data, get_feature_target, split_data_temporal

# Create alias for compatibility
load_and_clean_data = load_data

# Import all modules
from .feature_engineering import FeatureEngineer, create_feature_set, preprocess_features
from .model_trainer import ModelTrainer, CrimePredictor
from .utils import (
    save_model, load_model, evaluate_model, 
    plot_feature_importance, plot_confusion_matrix, generate_model_report
)

__all__ = [
    'CrimePredictorAPI',
    'load_data',
    'load_and_clean_data',
    'get_feature_target', 
    'split_data_temporal',
    'FeatureEngineer',
    'create_feature_set',
    'preprocess_features',
    'ModelTrainer', 
    'CrimePredictor',
    'save_model',
    'load_model',
    'evaluate_model',
    'plot_feature_importance',
    'plot_confusion_matrix',
    'generate_model_report'
]