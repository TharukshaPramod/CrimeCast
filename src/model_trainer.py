import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost with fallback
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using alternative models")

class CrimePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize multiple ML models including XGBoost"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42, 
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            print("⚠️ XGBoost skipped - not installed")
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance"""
        results = {}
        
        for name, model in self.models.items():
            print(f"🔄 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Display results
                auc_display = f"{auc:.4f}" if auc is not None else "N/A"
                print(f"   ✅ {name} - Accuracy: {accuracy:.4f}, AUC: {auc_display}")
                
                # Update best model
                if auc and auc > self.best_score:
                    self.best_score = auc
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"   ❌ {name} training failed: {e}")
                results[name] = {
                    'model': None,
                    'accuracy': 0,
                    'auc': 0,
                    'error': str(e)
                }
        
        return results
    
    def save_best_model(self, filepath):
        """Save the best performing model"""
        if self.best_model:
            joblib.dump(self.best_model, filepath)
            print(f"✅ Best model ({self.best_model_name}) saved to {filepath}")
            return True
        else:
            print("❌ No model trained yet")
            return False
    
    def load_model(self, filepath):
        """Load a saved model"""
        try:
            self.best_model = joblib.load(filepath)
            print(f"✅ Model loaded from {filepath}")
            return self.best_model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None

class ModelTrainer:
    def __init__(self):
        self.crime_predictor = CrimePredictor()
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.training_results = {}
    
    def train_models(self, X_train, y_train, X_test=None, y_test=None):
        """Train models using proper train-validation split"""
        from sklearn.model_selection import train_test_split
        
        # If no test set provided, create validation split
        if X_test is None or y_test is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            X_test, y_test = X_val, y_val
        
        # Initialize and train models
        self.crime_predictor.initialize_models()
        self.training_results = self.crime_predictor.train_models(X_train, y_train, X_test, y_test)
        
        # Store best model info
        self.best_model = self.crime_predictor.best_model
        self.best_model_name = self.crime_predictor.best_model_name
        self.best_score = self.crime_predictor.best_score
        
        return self.training_results
    
    def get_best_model(self):
        """Get the best model and its name"""
        return self.best_model_name, self.best_model
    
    def get_model_performance(self):
        """Get performance metrics for all models"""
        performance = {}
        for name, result in self.training_results.items():
            performance[name] = {
                'accuracy': result.get('accuracy', 0),
                'auc': result.get('auc', 0),
                'has_model': result.get('model') is not None
            }
        return performance
    
    def save_training_report(self, filepath):
        """Save training report to file"""
        report = {
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'model_performance': self.get_model_performance(),
            'models_trained': list(self.training_results.keys())
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Training report saved to {filepath}")