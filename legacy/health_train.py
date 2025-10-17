import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("csv/air_quality_health_impact_data.csv")

# Use only 'AQI' as input feature
X = df[['AQI']]

# Define targets
y_reg = df[['RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions', 'HealthImpactScore']]
y_clf = df['HealthImpactClass']

# Scale the AQI feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train multi-output regressor for continuous targets
regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
regressor.fit(X_scaled, y_reg)

# Train classifier for categorical target
classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
classifier.fit(X_scaled, y_clf)

# Prediction function for a single AQI input value
def predict_health_impact(aqi_value):
    input_df = pd.DataFrame({'AQI': [aqi_value]})  # Matches original feature column name
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

# Example usage
output = predict_health_impact(75)
print(output)
