import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class CrimePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def initialize_models(self):
        """Initialize multiple ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
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
            
            # Fixed the formatting issue here
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