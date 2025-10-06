import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(min(top_n, len(importances))), 
                importances[indices[:top_n]])
        plt.xticks(range(min(top_n, len(importances))), 
                  [feature_names[i] for i in indices[:top_n]], rotation=45)
        plt.tight_layout()
        return plt
    else:
        print("Model doesn't have feature_importances_ attribute")
        return None

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def generate_model_report(results, y_test):
    """Generate comprehensive model performance report"""
    report = {}
    
    for name, result in results.items():
        report[name] = {
            'accuracy': result['accuracy'],
            'auc': result['auc'],
            'classification_report': classification_report(y_test, result['predictions'], output_dict=True)
        }
    
    return report

# ADD MISSING FUNCTIONS FOR COMPATIBILITY
def save_model(model, filepath):
    """Save model to file"""
    joblib.dump(model, filepath)
    print(f"💾 Model saved to {filepath}")

def load_model(filepath):
    """Load model from file"""
    model = joblib.load(filepath)
    print(f"✅ Model loaded from {filepath}")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = None
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred
    }