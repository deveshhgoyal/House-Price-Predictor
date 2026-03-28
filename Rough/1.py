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

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())
# print(df.describe())

df['Target'] = data.target  # target is the median house value
print(df.head())

import sweetviz as sv
report = sv.analyze(df)
report.show_html('california_housing_report.html')


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="price_prediction_app")
# location = geolocator.reverse("37.88, -122.23").raw['address']
# print(location)

# Initialize loc_update dictionary
loc_update = {'County': [], 'road': []}

def get_location(cord):
    Latitude = str(cord[0])
    Longitude = str(cord[1])
    location = geolocator.reverse(Latitude + "," + Longitude).raw['address'] 

    if location.get('road') is None:
        location["road"] = None
    if location.get('County') is None:
        location["County"] = None
    
    loc_update['County'].append(location['County'])
    loc_update['road'].append(location['road'])

# for i,cord in enumerate(df.iloc[:,6:-1].values):
#     get_location(cord)

#     pickle.dump(loc_update, open('loc_update.pkl', 'wb'))

#     if i % 100 == 0:
#         print(i)


import pickle

loc_update = pickle.load(open(r'C:\Users\gargd\Downloads\PricePridiction\loc_update.pkl', 'rb'))

loc = pd.DataFrame(loc_update)
print(loc.head())
print(loc.info())

df.drop(labels=['Latitude', 'Longitude'] , axis=1, inplace=True)

for i in loc_update.keys():
    df[i] = loc_update[i]

df = df.sample(axis=0, frac=1)
print(df.head(10))
print(df.info())

df.replace({None: np.nan}, inplace=True)  # Convert all None → NaN
df.dropna(subset=['road', 'County'], inplace=True)


print(df.info())

from sklearn.preprocessing import LabelEncoder
le_road = LabelEncoder()
df['road'] = le_road.fit_transform(df['road'])

le_county = LabelEncoder()
df['County'] = le_county.fit_transform(df['County'])

print(df.info())
print(df.head())


# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
y = df['Target'].values
df.drop('Target', axis=1, inplace=True)
X = df.iloc[:,:].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
print("R2 Score:", r2_score(y_test, y_pred))

inp = np.array([[2.0096,22.0,4.027027,1.045045,754.0,3.396396,19,2131]])  # Example input
ip = inp.reshape(1, -1)

print(model.predict(ip))

