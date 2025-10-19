import pandas as pd
import joblib
import numpy as np

# Load trained models and scaler
scaler = joblib.load('health_model/scaler.pkl')
regressor = joblib.load('health_model/regressor_model.pkl')
classifier = joblib.load('health_model/classifier_model.pkl')

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

# Example prediction
print(predict_health_impact(75))
