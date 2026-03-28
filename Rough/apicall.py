import numpy as np
# Temporary patch for Sweetviz compatibility with NumPy 2.x
if not hasattr(np, "VisibleDeprecationWarning"):
    class VisibleDeprecationWarning(Warning):
        pass
    np.VisibleDeprecationWarning = VisibleDeprecationWarning

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())

df['Target'] = data.target  # target is the median house value
print(df.head())

import sweetviz as sv
# report = sv.analyze(df)
# report.show_html('california_housing_report.html')

# Geocoding setup (using OpenCage instead of Nominatim)
from geopy.geocoders import OpenCage
import time
API_KEY = "d88160a978aa4aedb7ed03984eb05dba"
geolocator = OpenCage(api_key=API_KEY)

# Resume progress if previous data exists
try:
    loc_update = pickle.load(open('loc_update.pkl', 'rb'))
    start_index = len(loc_update['County'])
    print(f"✅ Resuming from index {start_index}")
except FileNotFoundError:
    loc_update = {'County': [], 'road': []}
    start_index = 0
    print("🆕 Starting from scratch")

# Safe reverse geocoding function
def get_location(cord):
    Latitude = str(cord[0]) 
    Longitude = str(cord[1])
    try:
        location = geolocator.reverse((Latitude, Longitude))
        if location and 'components' in location.raw:
            address = location.raw['components']
            county = address.get('county', None)
            road = address.get('road', None)
        else:
            county, road = None, None
    except Exception as e:
        print("⚠️ Error:", e)
        county, road = None, None
    return county, road

# Loop through coordinates safely
coords = df.iloc[:, 6:8].values  # columns: Latitude, Longitude

for i, cord in enumerate(coords[start_index:], start=start_index):
    county, road = get_location(cord)
    loc_update['County'].append(county)
    loc_update['road'].append(road)

    # Save progress every 100 rows
    if i % 100 == 0:
        pickle.dump(loc_update, open('loc_update.pkl', 'wb'))
        print(f"Processed {i}/{len(coords)} rows...")

    # Wait between requests (avoid API limit)
    time.sleep(0.3)

# Final save
pickle.dump(loc_update, open('loc_update.pkl', 'wb'))
print("✅ Done! All results saved to loc_update.pkl")