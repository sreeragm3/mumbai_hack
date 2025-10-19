from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and label encoder at startup
model = joblib.load('air_quality_model/aqi_model.pkl')
le_city = joblib.load('air_quality_model/city_label_encoder.pkl')

# Input data schema
class PredictionInput(BaseModel):
    city: str
    date: str  # expects 'YYYY-MM-DD' format

# AQI category mapping function
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

def predict_aqi(city, date_str):
    try:
        city_encoded = le_city.transform([city])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail=f"City '{city}' not recognized.")
    
    date = pd.to_datetime(date_str)
    features = pd.DataFrame({
        "City_encoded": [city_encoded],
        "Year": [date.year],
        "Month": [date.month],
        "Day": [date.day],
        "DayOfWeek": [date.dayofweek]
    })
    aqi_value = model.predict(features)[0]
    category = get_category(aqi_value)
    return round(aqi_value, 2), category

@app.post("/predict")
def get_prediction(data: PredictionInput):
    aqi, category = predict_aqi(data.city, data.date)
    return {"predicted_aqi": aqi, "category": category}
