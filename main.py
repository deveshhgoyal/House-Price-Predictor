import numpy as np
# Temporary patch for Sweetviz compatibility with NumPy 2.x
if not hasattr(np, "VisibleDeprecationWarning"):
    class VisibleDeprecationWarning(Warning):
        pass
    np.VisibleDeprecationWarning = VisibleDeprecationWarning

import pickle
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load Dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target

print(df.head())


import sweetviz as sv
report = sv.analyze(df)
report.show_html('california_housing_report.html')


# Load pre-generated location mapping
# (You already generated loc_update.pkl)
loc_update = pickle.load(open(r'C:\Users\gargd\Downloads\PricePridiction\loc_update.pkl', 'rb'))

loc = pd.DataFrame(loc_update)
df.drop(labels=["Latitude", "Longitude"], axis=1, inplace=True)

for col in loc_update.keys():
    df[col] = loc_update[col]

df = df.sample(frac=1,random_state=42)
df.replace({None: np.nan}, inplace=True)
df.dropna(subset=["County"], inplace=True)
df.drop("road", axis=1, inplace=True)  # Temporarily drop to avoid issues

print(df.info())


# Label Encoding
from sklearn.preprocessing import LabelEncoder

# le_road = LabelEncoder()
# df["road"] = le_road.fit_transform(df["road"])

le_county = LabelEncoder()


df["County"] = le_county.fit_transform(df["County"])

# Save encoders for prediction
# pickle.dump(le_road, open("road_encoder.pkl", "wb"))
pickle.dump(le_county, open("county_encoder.pkl", "wb"))

print(df.head())


# Prepare Train/Test
from sklearn.model_selection import train_test_split

y = df["Target"].values
df.drop("Target", axis=1, inplace=True)
X = df.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# MODEL 1: RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)


# MODEL 2: Linear Regression (Baseline)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)


# MODEL 3: LightGBM
from lightgbm import LGBMRegressor

lgb = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
lgb.fit(X_train, y_train)
pred_lgb = lgb.predict(X_test)

pickle.dump(lgb, open("model.pkl", "wb"))

# Compare R2 Scores
print("\n========== MODEL PERFORMANCE ==========")
print("RandomForest R2:", r2_score(y_test, pred_rf))
print("Linear Regression R2:", r2_score(y_test, pred_lr))
print("LightGBM R2:", r2_score(y_test, pred_lgb))
print("=======================================\n")


# CLEAN Prediction Function
def predict_house_price(row_dict):
    """
    row_dict example:
    {
      "MedInc": 2.3,
      "HouseAge": 20,
      "AveRooms": 5.1,
      "AveBedrms": 1.0,
      "Population": 800,
      "AveOccup": 3.0,
      "County": "Alameda County"
    }
    """

    # Load encoders
    # le_road = pickle.load(open("road_encoder.pkl", "rb"))
    le_county = pickle.load(open("county_encoder.pkl", "rb"))

    # Convert categorical features properly
    # row_dict["road"] = le_road.transform([row_dict["road"]])[0]
    row_dict["County"] = le_county.transform([row_dict["County"]])[0]

    # Convert to DataFrame
    df_in = pd.DataFrame([row_dict])

    # Predict using best model (LightGBM)
    return lgb.predict(df_in)[0]


# Example prediction (YOU MUST GIVE REAL road + county names)
print(predict_house_price({
    "MedInc": 2.0096,
    "HouseAge": 22.0,
    "AveRooms": 4.027027,
    "AveBedrms": 1.045045,
    "Population": 754.0,
    "AveOccup": 3.396396,
    "County": "Alameda County"
}))
