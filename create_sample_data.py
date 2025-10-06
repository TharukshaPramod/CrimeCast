import pandas as pd
import numpy as np

def create_sample_data():
    """Create a small sample dataset for deployment"""
    np.random.seed(42)
    n_samples = 5000  # Small sample for deployment
    
    sample_data = {
        'Latitude': np.random.uniform(41.6, 42.1, n_samples),
        'Longitude': np.random.uniform(-87.95, -87.5, n_samples),
        'Beat': np.random.randint(100, 2000, n_samples),
        'District': np.random.randint(1, 25, n_samples),
        'Ward': np.random.randint(1, 50, n_samples),
        'Community Area': np.random.randint(1, 77, n_samples),
        'Hour': np.random.randint(0, 24, n_samples),
        'DayOfWeek': np.random.randint(0, 7, n_samples),
        'Month': np.random.randint(1, 13, n_samples),
        'Year': np.random.randint(2018, 2024, n_samples),
        'Location_Description_Clean': np.random.choice(['STREET', 'RESIDENCE', 'PARKING LOT', 'APARTMENT', 'SIDEWALK'], n_samples),
        'TimeOfDay': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples),
        'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_samples),
        'Arrest_Target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/sample_crime_data.csv', index=False)
    print(f"✅ Created sample dataset: {df.shape}")

if __name__ == "__main__":
    create_sample_data()
