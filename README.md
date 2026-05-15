# 🏠 California House Price Prediction

A machine learning web application that predicts California house prices using the **LightGBM** regression model, enriched with geolocation-based county mapping and served via a **Flask** web interface.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Input Features](#input-features)
- [Sample Prediction](#sample-prediction)
- [Results](#results)

---

## 📖 Overview

This project uses the **California Housing Dataset** from `sklearn.datasets` to train regression models for predicting median house prices across California counties. A Flask-based web application allows users to input housing features and get real-time price predictions.

---

## ✨ Features

- 🔍 **EDA Report** generated using **Sweetviz** (`california_housing_report.html`)
- 🗺️ **Geolocation Mapping** — Latitude/Longitude converted to County names via reverse geocoding (`loc_update.pkl`)
- 🤖 **Multiple Models Compared** — RandomForest, Linear Regression, and LightGBM
- 🏆 **Best Model** — LightGBM (`model.pkl`) selected based on R² score
- 🌐 **Flask Web App** — Interactive UI for predictions
- ✅ **Input Validation** — Rejects zero or negative input values with meaningful error messages

---

## 🛠️ Tech Stack

| Category        | Technology                          |
|----------------|--------------------------------------|
| Language        | Python 3.x                          |
| ML Framework    | scikit-learn, LightGBM              |
| Data Analysis   | pandas, NumPy, Sweetviz             |
| Web Framework   | Flask                               |
| Serialization   | pickle                              |
| Frontend        | HTML, CSS (Jinja2 templates)        |

---

## 📁 Project Structure

```
PricePridiction/
│
├── app.py                          # Flask web application
├── main.py                         # Model training & EDA script
│
├── model.pkl                       # Trained LightGBM model
├── county_encoder.pkl              # LabelEncoder for County feature
├── road_encoder.pkl                # LabelEncoder for Road feature
├── loc_update.pkl                  # Lat/Long → County mapping
│
├── california_housing_report.html  # Sweetviz EDA report
│
├── templates/
│   ├── index.html                  # Main prediction form
│   └── result.html                 # Prediction result page
│
├── static/
│   └── style.css                   # Stylesheet
│
└── Rough/                          # Scratch/experimental files
```

---

## 📊 Dataset

- **Source:** `sklearn.datasets.fetch_california_housing`
- **Original Features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- **Engineered Feature:** `County` — derived from Latitude/Longitude via reverse geocoding
- **Target:** Median house value (in $100,000 units)

---

## 🤖 Models Used

| Model                | R² Score (approx.) |
|---------------------|---------------------|
| Linear Regression    | ~0.60               |
| Random Forest        | ~0.81               |
| **LightGBM** ✅      | **~0.84**           |

> LightGBM was selected as the final model due to its highest R² score and efficient training.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/DeveshGarg19/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install flask numpy pandas scikit-learn lightgbm sweetviz pickle5
```

---

## 🚀 Usage

### Run the Flask Web App
```bash
python app.py
```
Then open your browser and navigate to: **http://127.0.0.1:5000**

### Retrain the Model (optional)
```bash
python main.py
```
> ⚠️ Requires `loc_update.pkl` to be present for county mapping.

---

## 📥 Input Features

| Feature      | Description                                  | Example  |
|-------------|----------------------------------------------|----------|
| `MedInc`    | Median income in the block (in $10,000s)     | `8.33`   |
| `HouseAge`  | Median age of houses in the block            | `41`     |
| `AveRooms`  | Average number of rooms per household        | `6.98`   |
| `AveBedrms` | Average number of bedrooms per household     | `1.02`   |
| `Population`| Total population in the block                | `322`    |
| `AveOccup`  | Average number of occupants per household    | `2.56`   |
| `County`    | California county (selected from dropdown)   | `Alameda County` |

---

## 🧪 Sample Prediction

```python
predict_house_price({
    "MedInc": 2.0096,
    "HouseAge": 22.0,
    "AveRooms": 4.027027,
    "AveBedrms": 1.045045,
    "Population": 754.0,
    "AveOccup": 3.396396,
    "County": "Alameda County"
})
# Output: ~1.83 (i.e., $183,000)
```

---

## 📈 Results

The **LightGBM** model achieved the best performance with:
- **R² Score:** ~0.84
- **Training Time:** Efficient even with 800 estimators
- **Hyperparameters Used:**
  - `n_estimators`: 800
  - `learning_rate`: 0.05
  - `num_leaves`: 31
  - `subsample`: 0.9
  - `colsample_bytree`: 0.9

---

## 👤 Author

**Devesh Garg**  
GitHub: [@DeveshGarg19](https://github.com/DeveshGarg19)

---

## 📄 License

This project is intended for educational purposes.
