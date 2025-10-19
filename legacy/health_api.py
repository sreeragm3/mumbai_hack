from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load models and scaler once at startup
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

# API endpoint
@app.post("/predict")
def get_prediction(data: AQIInput):
    result = predict_health_impact(data.AQI)
    return result
