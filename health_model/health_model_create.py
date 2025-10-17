import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset
df = pd.read_csv("csv/air_quality_health_impact_data.csv")

# Use only 'AQI' as input feature
X = df[['AQI']]
y_reg = df[['RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions', 'HealthImpactScore']]
y_clf = df['HealthImpactClass']

# Scale feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
regressor.fit(X_scaled, y_reg)
classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
classifier.fit(X_scaled, y_clf)

# Save models and scaler to .pkl files
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(regressor, 'regressor_model.pkl')
joblib.dump(classifier, 'classifier_model.pkl')
