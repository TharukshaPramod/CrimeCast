import pandas as pd
import numpy as np
import joblib
import traceback  # Add this import
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrimePredictorAPI:
    def __init__(self, model_path, feature_scaler_path=None, label_encoders_path=None):
        try:
            self.model = joblib.load(model_path)
            self.feature_scaler = joblib.load(feature_scaler_path) if feature_scaler_path else None
            self.label_encoders = joblib.load(label_encoders_path) if label_encoders_path else None
            
            # Define the exact feature order expected by the model
            self.expected_features = [
                'Latitude', 'Longitude', 'Beat', 'District', 'Ward', 
                'Community Area', 'Hour', 'DayOfWeek', 'Month', 'Year',
                'Location_Description_Clean', 'TimeOfDay', 'Season'
            ]
            
            logger.info("✅ CrimePredictorAPI initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize CrimePredictorAPI: {e}")
            raise
    
    def _encode_categorical_value(self, feature_name, value):
        """Helper method to encode categorical values safely"""
        if self.label_encoders and feature_name in self.label_encoders:
            try:
                return self.label_encoders[feature_name].transform([value])[0]
            except ValueError:
                # Handle unseen categories by using the most frequent class
                logger.warning(f"Unseen category '{value}' for feature '{feature_name}', using default encoding")
                return 0  # Usually encodes the most frequent class
        return value
    
    def _validate_input_features(self, features_dict):
        """Validate input features and provide warnings for missing ones"""
        missing_features = set(self.expected_features) - set(features_dict.keys())
        if missing_features:
            logger.warning(f"⚠️ Missing features: {missing_features}")
        
        extra_features = set(features_dict.keys()) - set(self.expected_features)
        if extra_features:
            logger.warning(f"⚠️ Extra features provided: {extra_features}")
    
    def create_feature_array(self, features_dict):
        """Create feature array in the exact order expected by the model"""
        # Validate input first
        self._validate_input_features(features_dict)
        
        feature_array = []
        
        for feature in self.expected_features:
            if feature in features_dict:
                value = features_dict[feature]
                
                # Handle categorical features
                if feature in ['Location_Description_Clean', 'TimeOfDay', 'Season']:
                    encoded_value = self._encode_categorical_value(feature, value)
                    feature_array.append(encoded_value)
                else:
                    # Numerical features - ensure they're numeric
                    try:
                        feature_array.append(float(value))
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ Non-numeric value for feature {feature}, using 0")
                        feature_array.append(0)
            else:
                # If feature is missing, use a default value
                logger.warning(f"⚠️ Missing feature {feature}, using default value 0")
                feature_array.append(0)
        
        return np.array([feature_array])
    
    def predict(self, features_dict):
        """Predict crime probability for given features"""
        try:
            # Create feature array in correct order
            features_array = self.create_feature_array(features_dict)
            
            logger.info(f"🔍 Features array shape: {features_array.shape}")
            logger.debug(f"🔍 Features: {self.expected_features}")
            logger.debug(f"🔍 Values: {features_array[0]}")
            
            # Scale features if scaler is available
            if self.feature_scaler:
                features_array = self.feature_scaler.transform(features_array)
            
            # Make prediction
            probability = self.model.predict_proba(features_array)[0, 1]
            prediction = self.model.predict(features_array)[0]
            
            # Determine risk level
            risk_level = self._get_risk_level(probability)
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'features_used': self.expected_features,
                'feature_values': features_array[0].tolist(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            logger.error(traceback.format_exc())  # Now traceback is defined
            return {
                'error': str(e),
                'success': False
            }
    
    def _get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability > 0.7:
            return 'High'
        elif probability > 0.4:  # Adjusted threshold for better distribution
            return 'Medium'
        else:
            return 'Low'
    
    def batch_predict(self, features_list):
        """Make predictions for multiple feature sets at once"""
        results = []
        for features in features_list:
            results.append(self.predict(features))
        return results
    
    def get_feature_info(self):
        """Get information about expected features"""
        return {
            'expected_features': self.expected_features,
            'categorical_features': ['Location_Description_Clean', 'TimeOfDay', 'Season'],
            'numerical_features': [f for f in self.expected_features if f not in ['Location_Description_Clean', 'TimeOfDay', 'Season']]
        }