from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()


#########HEALTH IMPACT MODEL SECTION#########


# Load models and scaler - health
scaler = joblib.load('health_model/scaler.pkl')       
regressor = joblib.load('health_model/regressor_model.pkl')
classifier = joblib.load('health_model/classifier_model.pkl')


# Define input data model
class AQIInput(BaseModel):
    AQI: float


# Health impact prediction function
def predict_health_impact(aqi_value):
    input_df = pd.DataFrame({'AQI': [aqi_value]})
    aqi_scaled = scaler.transform(input_df)
    reg_preds = regressor.predict(aqi_scaled)[0]
    clf_pred = classifier.predict(aqi_scaled)[0]
    return {
        'RespiratoryCases': reg_preds[0],
        'CardiovascularCases': reg_preds[1],
        'HospitalAdmissions': reg_preds[2],
        'HealthImpactScore': reg_preds[3],
        'HealthImpactClass': clf_pred
    }


##########AIR QUALITY PREDICTION MODEL SECTION#########

# Load models and scaler - air quality
model = joblib.load('air_quality_model/aqi_model.pkl')
le_city = joblib.load('air_quality_model/city_label_encoder.pkl')

class PredictionInput(BaseModel):
    city: str
    date: str  # expects 'YYYY-MM-DD' format    

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

##########API ENDPOINTS#########
@app.post("/predicthealth")
def get_prediction(data: AQIInput):
    result = predict_health_impact(data.AQI)
    return result

@app.post("/predictaqi")
def get_prediction(data: PredictionInput):
    aqi, category = predict_aqi(data.city, data.date)
    return {"predicted_aqi": aqi, "category": category}
