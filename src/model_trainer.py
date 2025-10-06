import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class CrimePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def initialize_models(self):
        """Initialize ML models without xgboost for deployment"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=50)
        }
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        results = {}
        
        for name, model in self.models.items():
            print(f"🔄 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            accuracy = model.score(X_test, y_test)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            auc_display = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"   ✅ {name} - Accuracy: {accuracy:.4f}, AUC: {auc_display}")
            
            # Update best model
            if auc and auc > self.best_score:
                self.best_score = auc
                self.best_model = model
                self.best_model_name = name
        
        return results
    
    def save_best_model(self, filepath):
        """Save the best performing model"""
        if self.best_model:
            joblib.dump(self.best_model, filepath)
            print(f"✅ Best model ({self.best_model_name}) saved to {filepath}")
        else:
            print("❌ No model trained yet")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.best_model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
        return self.best_model

class ModelTrainer:
    def __init__(self):
        self.crime_predictor = CrimePredictor()
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
    
    def train_models(self, X_train, y_train):
        """Train models using CrimePredictor"""
        self.crime_predictor.initialize_models()
        
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        results = self.crime_predictor.train_models(X_tr, y_tr, X_val, y_val)
        
        self.best_model = self.crime_predictor.best_model
        self.best_model_name = self.crime_predictor.best_model_name
        self.best_score = self.crime_predictor.best_score
        
        return results
    
    def get_best_model(self, models, X_test, y_test):
        """Get the best model"""
        return self.best_model_name, self.best_model
