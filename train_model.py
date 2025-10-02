#!/usr/bin/env python3
"""
Script to train crime prediction models
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.feature_engineering import create_feature_set
from src.model_trainer import CrimePredictor
from src.utils import plot_feature_importance, plot_confusion_matrix, generate_model_report
import joblib
from config import MODEL_SAVE_PATH, RANDOM_STATE

def main():
    print("🚀 Starting Crime Prediction Model Training...")
    
    # Load data
    df = load_data()
    if df is None:
        print("❌ Failed to load data. Please check if cleaned_crime_data.csv exists in data/ folder.")
        return
    
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    # Choose prediction task
    target_type = 'arrest'  # Change to 'violent_crime' or 'crime_type' for other tasks
    print(f"🎯 Prediction task: {target_type}")
    
    try:
        # Create feature set
        X, y, label_encoders, scaler = create_feature_set(df, target_type)
        print(f"✅ Features prepared: {X.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]:,} samples")
        print(f"📊 Test set: {X_test.shape[0]:,} samples")
        target_dist = y.value_counts(normalize=True).to_dict()
        print(f"🎯 Target distribution: {target_dist}")
        
        # Train models
        predictor = CrimePredictor()
        predictor.initialize_models()
        results = predictor.train_models(X_train, y_train, X_test, y_test)
        
        # Save best model
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        model_path = os.path.join(MODEL_SAVE_PATH, f'best_model_{target_type}.pkl')
        predictor.save_best_model(model_path)
        
        # Save preprocessing objects
        joblib.dump(scaler, os.path.join(MODEL_SAVE_PATH, 'feature_scaler.pkl'))
        joblib.dump(label_encoders, os.path.join(MODEL_SAVE_PATH, 'label_encoders.pkl'))
        
        # Generate plots
        if predictor.best_model:
            # Feature importance
            plt = plot_feature_importance(predictor.best_model, X.columns)
            if plt:
                plt.savefig(os.path.join(MODEL_SAVE_PATH, 'feature_importance.png'))
                plt.close()
                print("✅ Feature importance plot saved")
            
            # Confusion matrix for best model
            best_results = results[predictor.best_model_name]
            plt = plot_confusion_matrix(y_test, best_results['predictions'], predictor.best_model_name)
            if plt:
                plt.savefig(os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png'))
                plt.close()
                print("✅ Confusion matrix plot saved")
        
        # Generate report
        report = generate_model_report(results, y_test)
        print("\n🎉 Model Training Completed!")
        print(f"🏆 Best Model: {predictor.best_model_name} (AUC: {predictor.best_score:.4f})")
        
        # Print final results
        print("\n📊 Final Model Performance:")
        for name, result in results.items():
            auc_display = f"{result['auc']:.4f}" if result['auc'] is not None else "N/A"
            print(f"   {name}: Accuracy={result['accuracy']:.4f}, AUC={auc_display}")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()