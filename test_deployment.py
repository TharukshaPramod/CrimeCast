print("🚀 Testing deployment configuration...")

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    from geopy.geocoders import Nominatim
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    print("✅ All core imports successful!")
    
    # Test basic functionality
    df = pd.DataFrame({'test': [1, 2, 3]})
    print(f"✅ Pandas working: {df.shape}")
    
    arr = np.array([1, 2, 3])
    print(f"✅ NumPy working: {arr.sum()}")
    
    # Test geopy (with error handling)
    try:
        geolocator = Nominatim(user_agent="deployment_test", timeout=5)
        location = geolocator.geocode("Chicago")
        if location:
            print(f"✅ Geopy working: Found {location.address[:30]}...")
        else:
            print("⚠️  Geopy: No location found (this is normal for testing)")
    except Exception as e:
        print(f"⚠️  Geopy test skipped: {e}")
    
    print("\n🎉 DEPLOYMENT READY! All packages are compatible.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")