# Configuration file for the project

# Data paths
DATA_PATH = "data/cleaned_crime_data.csv"
MODEL_SAVE_PATH = "models/"

# Feature configuration
NUMERICAL_FEATURES = ['Latitude', 'Longitude', 'Beat', 'District', 'Ward', 
                     'Community Area', 'Hour', 'DayOfWeek', 'Month', 'Year']

CATEGORICAL_FEATURES = ['Location_Description_Clean', 'TimeOfDay', 'Season']

TARGET_VARIABLES = {
    'arrest': 'Arrest_Target',
    'violent_crime': 'Violent_Crime',
    'crime_type': 'Primary_Type_Encoded'
}

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Streamlit app configuration
APP_TITLE = "Chicago Crime Prediction Dashboard"
APP_DESCRIPTION = "Predict crime patterns and probabilities in Chicago"