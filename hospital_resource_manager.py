# hospital_resource_manager.py - Comprehensive Hospital Resource Management System
"""
Integrates:
- Set 1: Festival Surge Predictions (blood units, ED volume)
- Set 2: AQI Health Impact (respiratory/cardiovascular cases, hospital admissions)
- Set 3: Patient Count Predictions (pollution-based patient volume forecasts)

Provides unified bed capacity planning and resource allocation
"""

import operator
from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import os
import sys

# Add chinmay_mumabaihacks subdirectory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
hsr_dir = os.path.join(current_dir, 'chinmay_mumabaihacks')
sys.path.insert(0, hsr_dir)

# Import Set 1 Hospital Surge modules
try:
    from data_fetcher import (
        fetch_festival_dates_calendarific,
        get_events_for_alerting,
        get_forecast_inputs
    )
    from model_trainer import predict_surge_needs
    from config import TARGET_YEAR, ALERT_LEAD_TIMES
    import joblib as jb
    print("✅ Loaded Set 1 modules (Festival Surge)")
    SET1_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Could not load Set 1 modules: {e}")
    SET1_AVAILABLE = False

# Load all ML models
try:
    aqi_model = joblib.load('air_quality_model/aqi_model.pkl')
    city_encoder = joblib.load('air_quality_model/city_label_encoder.pkl')
    health_regressor = joblib.load('health_model/regressor_model.pkl')
    health_classifier = joblib.load('health_model/classifier_model.pkl')
    scaler = joblib.load('health_model/scaler.pkl')
    print("✅ Loaded Set 2 models (AQI & Health Impact)")
    SET2_AVAILABLE = True
except Exception as e:
    print(f"❌ Error loading Set 2 models: {e}")
    SET2_AVAILABLE = False

# Load Hospital Surge model
surge_model = None
surge_features = None
if SET1_AVAILABLE:
    try:
        surge_model_path = os.path.join(hsr_dir, 'data', 'surge_predictor.joblib')
        surge_model = jb.load(surge_model_path)
        surge_features = ['ed_volume_change_percent', 'trauma_cases', 'burn_cases', 
                         'respiratory_cases', 'festival_Diwali', 'festival_Holi']
        print("✅ Loaded Hospital Surge model")
    except Exception as e:
        print(f"⚠️ Hospital Surge model not available: {e}")

# Set 3: Patient Count Prediction (pollution-based)
def initialize_patient_predictor():
    """Initialize Set 3 patient count prediction model"""
    from sklearn.tree import DecisionTreeRegressor

    # Dummy historical data for Kochi
    data = {
        "Date": pd.date_range("2025-10-01", periods=10),
        "PM2.5": [45, 50, 70, 80, 60, 55, 95, 40, 75, 65],
        "PM10": [80, 90, 110, 100, 85, 95, 120, 75, 105, 90],
        "NO2": [60, 65, 70, 85, 78, 72, 88, 55, 90, 82],
        "Total_Patients": [35, 36, 38, 40, 39, 37, 45, 33, 43, 41],
    }
    df = pd.DataFrame(data)
    df["Prev_Patient"] = df["Total_Patients"].shift(1).fillna(method="bfill")

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(df[["Prev_Patient"]], df["Total_Patients"])

    return model, df

try:
    patient_model, patient_df = initialize_patient_predictor()
    print("✅ Loaded Set 3 model (Patient Count Predictor)")
    SET3_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Could not initialize Set 3: {e}")
    SET3_AVAILABLE = False

# CPCB pollution limits
POLLUTION_LIMITS = {
    "PM2.5": (60, "Respiratory"),
    "PM10": (100, "Respiratory"),
    "NO2": (80, "Respiratory"),
}

# Define Enhanced State for Resource Management
class HospitalState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

    # Set 2: Daily Health Data
    aqi_data: dict
    health_predictions: dict

    # Set 1: Festival Surge Data
    festival_predictions: dict
    surge_predictions: dict

    # Set 3: Patient Count Predictions
    patient_count_forecast: dict

    # Epidemic Monitoring
    epidemic_alerts: List[str]

    # NEW: Unified Resource Management
    bed_capacity_plan: dict
    resource_allocation: dict
    critical_alerts: List[str]

    task_type: str

# Helper Functions
def get_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

def get_health_impact_class(score):
    if score == 0: return "Low"
    elif score == 1: return "Moderate"
    elif score == 2: return "High"
    else: return "Severe"

# Node 1: Predict Daily AQI & Health Impact
def predict_daily_health(state: HospitalState) -> HospitalState:
    """Predict AQI and health impacts (Set 2)"""
    print("\n📊 [Set 2] Predicting Daily AQI & Health Impact...")

    if not SET2_AVAILABLE:
        state['aqi_data'] = {}
        state['health_predictions'] = {}
        return state

    now = datetime.now()
    features = pd.DataFrame([[
        city_encoder.transform(['Delhi'])[0],
        now.year,
        now.month,
        now.day,
        now.weekday()
    ]], columns=['City_encoded', 'Year', 'Month', 'Day', 'DayOfWeek'])

    aqi_value = aqi_model.predict(features)[0]
    category = get_category(aqi_value)

    aqi_scaled = scaler.transform([[aqi_value]])
    reg_pred = health_regressor.predict(aqi_scaled)[0]
    class_pred = health_classifier.predict(aqi_scaled)[0]

    state['aqi_data'] = {
        'aqi': round(aqi_value, 2),
        'category': category,
        'date': now.strftime('%Y-%m-%d'),
        'city': 'Delhi'
    }

    state['health_predictions'] = {
        'respiratory_cases': int(reg_pred[0]),
        'cardiovascular_cases': int(reg_pred[1]),
        'hospital_admissions': int(reg_pred[2]),
        'health_impact_score': round(reg_pred[3], 2),
        'health_impact_class': get_health_impact_class(class_pred)
    }

    print(f"   AQI: {aqi_value:.1f} ({category})")
    print(f"   Expected Admissions: {int(reg_pred[2])}")

    return state

# Node 2: Predict Festival Surge
def predict_festival_surge(state: HospitalState) -> HospitalState:
    """Fetch upcoming festivals and predict hospital surge (Set 1)"""
    print("\n🎉 [Set 1] Predicting Festival-Related Surge...")

    if not SET1_AVAILABLE or not surge_model:
        state['festival_predictions'] = {}
        state['surge_predictions'] = {}
        return state

    try:
        calendar_df = fetch_festival_dates_calendarific(TARGET_YEAR)
        today = datetime.now().date()
        events_to_alert = get_events_for_alerting(calendar_df, today)

        if not events_to_alert:
            state['festival_predictions'] = {'festivals': [], 'total_festivals': 0}
            state['surge_predictions'] = {}
            return state

        festival_data = []
        for event in events_to_alert:
            festival_name = event['festival_name']
            festival_date = event['festival_date']
            days_until = (festival_date - today).days

            forecast = get_forecast_inputs(festival_name)
            surge_pred = predict_surge_needs(
                surge_model,
                surge_features,
                festival_name,
                forecast['ed_volume_change'],
                forecast['trauma_cases']
            )

            festival_data.append({
                'festival': festival_name,
                'date': festival_date.strftime('%Y-%m-%d'),
                'days_until': days_until,
                'ed_surge_multiplier': forecast['ed_volume_change'],
                'blood_units_needed': surge_pred['predicted_blood_units'],
                'alert_level': surge_pred['alert_level']
            })

            print(f"   {festival_name} in {days_until} days: {surge_pred['alert_level']}")

        state['festival_predictions'] = {'festivals': festival_data, 'total_festivals': len(festival_data)}
        state['surge_predictions'] = {'model_available': True}

    except Exception as e:
        print(f"   Error: {e}")
        state['festival_predictions'] = {}
        state['surge_predictions'] = {}

    return state

# Node 3: Predict Patient Count
def predict_patient_count(state: HospitalState) -> HospitalState:
    """Predict patient count based on pollution (Set 3)"""
    print("\n👥 [Set 3] Predicting Patient Volume...")

    if not SET3_AVAILABLE:
        state['patient_count_forecast'] = {}
        return state

    # Get latest patient count
    latest_prev = patient_df["Total_Patients"].iloc[-1]
    base_pred = patient_model.predict([[latest_prev]])[0]

    # Pollution adjustment based on current AQI
    aqi = state.get('aqi_data', {}).get('aqi', 100)
    pollution_impact = max(0, (aqi - 100) * 0.2)  # Extra patients per AQI point above 100

    final_pred = base_pred + pollution_impact
    final_rounded = round(final_pred)

    state['patient_count_forecast'] = {
        'base_patients': int(base_pred),
        'pollution_adjustment': int(pollution_impact),
        'total_expected_patients': final_rounded,
        'confidence_range': (final_rounded - 3, final_rounded + 3)
    }

    print(f"   Expected Patients: {final_rounded} (+{int(pollution_impact)} from pollution)")

    return state

# Node 4: Unified Bed Capacity Planning
def calculate_bed_capacity(state: HospitalState) -> HospitalState:
    """
    CORE INTEGRATION NODE
    Combines all predictions to optimize bed allocation
    """
    print("\n🏥 [INTEGRATION] Calculating Bed Capacity & Resource Needs...")

    # Baseline capacity
    TOTAL_BEDS = 200
    CURRENT_OCCUPANCY = 150
    available_beds = TOTAL_BEDS - CURRENT_OCCUPANCY

    # Calculate total expected admissions
    daily_health_admissions = state.get('health_predictions', {}).get('hospital_admissions', 0)
    patient_volume = state.get('patient_count_forecast', {}).get('total_expected_patients', 0)

    # Check for festival surge
    festival_surge_multiplier = 1.0
    festival_info = ""
    festivals = state.get('festival_predictions', {}).get('festivals', [])
    if festivals:
        # Get nearest festival
        nearest = min(festivals, key=lambda x: x['days_until'])
        festival_surge_multiplier = nearest['ed_surge_multiplier']
        festival_info = f"{nearest['festival']} in {nearest['days_until']} days"

    # Calculate adjusted admissions
    base_admissions = daily_health_admissions + patient_volume
    surge_adjusted_admissions = int(base_admissions * festival_surge_multiplier)

    # Bed requirements
    avg_stay_days = 4  # Average patient stay
    beds_needed = surge_adjusted_admissions * (avg_stay_days / 7)  # Weekly demand

    # Capacity status
    capacity_status = "Normal"
    if beds_needed > available_beds:
        capacity_status = "🚨 OVERCAPACITY ALERT"
    elif beds_needed > available_beds * 0.8:
        capacity_status = "⚠️ Near Capacity"

    state['bed_capacity_plan'] = {
        'total_beds': TOTAL_BEDS,
        'current_occupancy': CURRENT_OCCUPANCY,
        'available_beds': available_beds,
        'daily_expected_admissions': surge_adjusted_admissions,
        'beds_needed_weekly': int(beds_needed),
        'capacity_status': capacity_status,
        'festival_impact': festival_info if festival_info else "No festivals in alert window"
    }

    print(f"   Status: {capacity_status}")
    print(f"   Expected Admissions: {surge_adjusted_admissions}/day")
    print(f"   Beds Needed (weekly): {int(beds_needed)}/{available_beds} available")

    return state

# Node 5: Resource Allocation Planning
def calculate_resources(state: HospitalState) -> HospitalState:
    """
    CORE INTEGRATION NODE
    Plans staffing and equipment based on health impacts and patient volume
    """
    print("\n📋 [INTEGRATION] Planning Resource Allocation...")

    # Get predictions
    respiratory = state.get('health_predictions', {}).get('respiratory_cases', 0)
    cardiovascular = state.get('health_predictions', {}).get('cardiovascular_cases', 0)
    admissions = state.get('bed_capacity_plan', {}).get('daily_expected_admissions', 0)

    # Staff calculations (1 nurse per 5 patients, 1 doctor per 15 patients)
    nurses_needed = max(10, admissions // 5)
    doctors_needed = max(5, admissions // 15)

    # Equipment needs
    ventilators_needed = max(5, respiratory // 3)
    cardiac_monitors = max(5, cardiovascular // 2)

    # Blood units (from festival surge if applicable)
    blood_units = 0
    festivals = state.get('festival_predictions', {}).get('festivals', [])
    if festivals:
        blood_units = festivals[0].get('blood_units_needed', 0)

    state['resource_allocation'] = {
        'staffing': {
            'nurses_required': nurses_needed,
            'doctors_required': doctors_needed,
            'respiratory_specialists': max(2, respiratory // 10)
        },
        'equipment': {
            'ventilators': ventilators_needed,
            'cardiac_monitors': cardiac_monitors,
            'pulse_oximeters': admissions // 2
        },
        'supplies': {
            'blood_units': blood_units,
            'respiratory_medications': f"{respiratory * 2} doses",
            'cardiac_medications': f"{cardiovascular * 2} doses"
        }
    }

    print(f"   Staff: {nurses_needed} nurses, {doctors_needed} doctors")
    print(f"   Equipment: {ventilators_needed} ventilators, {cardiac_monitors} cardiac monitors")
    if blood_units:
        print(f"   Blood Supply: {blood_units} units needed")

    return state

# Node 6: Generate Critical Alerts
def generate_alerts(state: HospitalState) -> HospitalState:
    """Generate prioritized alerts for hospital administration"""
    print("\n⚠️ Generating Critical Alerts...")

    alerts = []

    # Check AQI
    aqi = state.get('aqi_data', {}).get('aqi', 0)
    if aqi > 200:
        alerts.append(f"🔴 CRITICAL: AQI {aqi} - Prepare for respiratory surge")

    # Check bed capacity
    capacity = state.get('bed_capacity_plan', {}).get('capacity_status', '')
    if 'OVERCAPACITY' in capacity:
        alerts.append("🔴 CRITICAL: Bed capacity exceeded - Activate overflow protocols")
    elif 'Near Capacity' in capacity:
        alerts.append("🟡 WARNING: Approaching bed capacity")

    # Check festival surge
    festivals = state.get('festival_predictions', {}).get('festivals', [])
    for fest in festivals:
        if fest['days_until'] <= 7:
            alerts.append(f"🟡 ALERT: {fest['festival']} in {fest['days_until']} days - {fest['alert_level']}")

    # Check epidemic
    epidemic_alerts = state.get('epidemic_alerts', [])
    if epidemic_alerts:
        for alert in epidemic_alerts[:2]:
            if alert.get('severity') == 'HIGH':
                alerts.append(f"🔴 EPIDEMIC: {alert.get('headline', 'Unknown')}")

    state['critical_alerts'] = alerts

    for alert in alerts:
        print(f"   {alert}")

    return state

# Node 7: Epidemic Scanner (from Set 2)
def scrape_epidemic_news(state: HospitalState) -> HospitalState:
    """Scrape WHO disease outbreak news"""
    print("\n🦠 Scanning for Epidemic Alerts...")

    alerts = []
    try:
        url = "https://www.who.int/emergencies/disease-outbreak-news"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = soup.find_all(['h2', 'h3', 'a'], limit=10)

        keywords = ['outbreak', 'epidemic', 'disease', 'pandemic']
        for headline in headlines:
            text = headline.get_text().lower()
            if any(keyword in text for keyword in keywords):
                alerts.append({
                    'headline': headline.get_text().strip(),
                    'severity': 'HIGH',
                    'timestamp': datetime.now().isoformat()
                })

        print(f"   Found {len(alerts)} potential alerts")
    except Exception as e:
        print(f"   Scraping failed: {e}")
        alerts = []

    state['epidemic_alerts'] = alerts
    return state

# Router
def route_task(state: HospitalState) -> str:
    task = state.get('task_type', 'full_analysis')
    if task == 'full_analysis':
        return 'daily_health'
    elif task == 'epidemic_only':
        return 'epidemic_scan'
    else:
        return END

# Build Workflow
workflow = StateGraph(HospitalState)

# Add nodes
workflow.add_node("daily_health", predict_daily_health)
workflow.add_node("festival_surge", predict_festival_surge)
workflow.add_node("patient_count", predict_patient_count)
workflow.add_node("bed_capacity", calculate_bed_capacity)
workflow.add_node("resources", calculate_resources)
workflow.add_node("alerts", generate_alerts)
workflow.add_node("epidemic_scan", scrape_epidemic_news)

# Add edges for full analysis workflow
workflow.add_conditional_edges(START, route_task)
workflow.add_edge("daily_health", "festival_surge")
workflow.add_edge("festival_surge", "patient_count")
workflow.add_edge("patient_count", "epidemic_scan")
workflow.add_edge("epidemic_scan", "bed_capacity")
workflow.add_edge("bed_capacity", "resources")
workflow.add_edge("resources", "alerts")
workflow.add_edge("alerts", END)

# Compile
hospital_agent = workflow.compile()

# Test Function
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏥 INTEGRATED HOSPITAL RESOURCE MANAGEMENT SYSTEM")
    print("="*70)

    result = hospital_agent.invoke({"task_type": "full_analysis", "messages": []})

    print("\n" + "="*70)
    print("📊 EXECUTIVE SUMMARY")
    print("="*70)

    # Print key metrics
    print("\n🌫️ AIR QUALITY & HEALTH IMPACT:")
    print(json.dumps(result.get('aqi_data'), indent=2))
    print(json.dumps(result.get('health_predictions'), indent=2))

    print("\n🏥 BED CAPACITY PLAN:")
    print(json.dumps(result.get('bed_capacity_plan'), indent=2))

    print("\n📋 RESOURCE ALLOCATION:")
    print(json.dumps(result.get('resource_allocation'), indent=2))

    print("\n⚠️ CRITICAL ALERTS:")
    for alert in result.get('critical_alerts', []):
        print(f"   {alert}")

    print("\n" + "="*70)
