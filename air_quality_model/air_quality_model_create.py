import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1️⃣ Load the standardized AQI dataset
df = pd.read_csv("csv/standardized_city_aqi.csv")

# 2️⃣ Handle missing values
df = df.dropna(subset=["Computed_AQI"])  # remove rows with no AQI

# 3️⃣ Convert Date to datetime and extract useful features
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# 4️⃣ Encode City (label encoding)
le_city = LabelEncoder()
df["City_encoded"] = le_city.fit_transform(df["City"])

# 5️⃣ Prepare features and target
X = df[["City_encoded", "Year", "Month", "Day", "DayOfWeek"]]
y = df["Computed_AQI"]

# 6️⃣ Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️⃣ Train a regression model (Random Forest)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Evaluate performance (optional)
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 9️⃣ Define category mapping function
def get_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# 1️⃣0️⃣ Prediction function that can be reused
def predict_aqi(city, date_str, model, le_city):
    date = pd.to_datetime(date_str)
    features = pd.DataFrame({
        "City_encoded": [le_city.transform([city])[0]],
        "Year": [date.year],
        "Month": [date.month],
        "Day": [date.day],
        "DayOfWeek": [date.dayofweek]
    })

    aqi_value = model.predict(features)[0]
    category = get_category(aqi_value)
    return round(aqi_value, 2), category

# 1️⃣1️⃣ Save model and label encoder to .pkl files for reuse
joblib.dump(model, 'aqi_model.pkl')
joblib.dump(le_city, 'city_label_encoder.pkl')

# Example usage after loading:
# model = joblib.load('aqi_model.pkl')
# le_city = joblib.load('city_label_encoder.pkl')
# aqi, category = predict_aqi('Delhi', '2018-11-10', model, le_city)
# print(aqi, category)
