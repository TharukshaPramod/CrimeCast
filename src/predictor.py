import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CrimePredictorAPI:
    def __init__(self, model_path, feature_scaler_path=None, label_encoders_path=None):
        self.model = joblib.load(model_path)
        self.feature_scaler = joblib.load(feature_scaler_path) if feature_scaler_path else None
        self.label_encoders = joblib.load(label_encoders_path) if label_encoders_path else None
        
        # Define the exact feature order expected by the model
        self.expected_features = [
            'Latitude', 'Longitude', 'Beat', 'District', 'Ward', 
            'Community Area', 'Hour', 'DayOfWeek', 'Month', 'Year',
            'Location_Description_Clean', 'TimeOfDay', 'Season'
        ]
        
    def create_feature_array(self, features_dict):
        """Create feature array in the exact order expected by the model"""
        feature_array = []
        
        for feature in self.expected_features:
            if feature in features_dict:
                value = features_dict[feature]
                
                # Handle categorical features
                if feature in ['Location_Description_Clean', 'TimeOfDay', 'Season']:
                    if self.label_encoders and feature in self.label_encoders:
                        # Convert categorical value to encoded number
                        try:
                            encoded_value = self.label_encoders[feature].transform([value])[0]
                            feature_array.append(encoded_value)
                        except ValueError:
                            # If value not seen during training, use most common
                            feature_array.append(0)
                    else:
                        feature_array.append(value)
                else:
                    # Numerical features
                    feature_array.append(float(value))
            else:
                # If feature is missing, use a default value
                print(f"⚠️  Warning: Missing feature {feature}, using default value 0")
                feature_array.append(0)
        
        return np.array([feature_array])
    
    def predict(self, features_dict):
        """Predict crime probability for given features"""
        try:
            # Create feature array in correct order
            features_array = self.create_feature_array(features_dict)
            
            print(f"🔍 Features array shape: {features_array.shape}")
            print(f"🔍 Features: {self.expected_features}")
            print(f"🔍 Values: {features_array[0]}")
            
            # Scale features if scaler is available
            if self.feature_scaler:
                features_array = self.feature_scaler.transform(features_array)
            
            # Make prediction
            probability = self.model.predict_proba(features_array)[0, 1]
            prediction = self.model.predict(features_array)[0]
            
            # Determine risk level
            if probability > 0.7:
                risk_level = 'High'
            elif probability > 0.3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'features_used': self.expected_features,
                'feature_values': features_array[0].tolist()
            }
            
        except Exception as e:
            print(f"❌ Prediction error details: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}