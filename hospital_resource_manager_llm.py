# hospital_resource_manager_llm.py - LLM-Enhanced Hospital Resource Management
"""
Integrates Google Gemini LLM for natural language insights and recommendations.

Requirements:
    pip install google-generativeai langgraph pandas scikit-learn joblib requests beautifulsoup4
"""

import operator
from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

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

# Load ML models
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

# Initialize Gemini LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ GOOGLE_API_KEY not found in .env file")
    LLM_AVAILABLE = False
    llm = None
else:
    # Try multiple model versions in order of preference
    model_options = [
        "gemini-2.0-flash",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash",
        "gemini-1.5-pro-002",
        "gemini-pro"
    ]
    
    llm = None
    for model_name in model_options:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            llm = genai.GenerativeModel(model_name)
            # Test the model with a simple query
            test_response = llm.generate_content("Say 'OK'")
            print(f"✅ Loaded Gemini LLM ({model_name})")
            LLM_AVAILABLE = True
            break
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}: {str(e)[:50]}...")
            if model_name == model_options[-1]:
                print(f"❌ All models failed. LLM features disabled.")
                LLM_AVAILABLE = False
                llm = None
            continue

# Set 3: Patient Count Prediction
def initialize_patient_predictor():
    from sklearn.tree import DecisionTreeRegressor
    data = {
        "Date": pd.date_range("2025-10-01", periods=10),
        "PM2.5": [45, 50, 70, 80, 60, 55, 95, 40, 75, 65],
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

# Define Enhanced State
class HospitalState(TypedDict):
    aqi_data: dict
    health_predictions: dict
    festival_predictions: dict
    surge_predictions: dict
    patient_count_forecast: dict
    epidemic_alerts: List[str]
    bed_capacity_plan: dict
    resource_allocation: dict
    critical_alerts: List[str]
    llm_insights: dict
    user_query: str
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
    aqi_scaled = scaler.transform([[aqi_value]])
    reg_pred = health_regressor.predict(aqi_scaled)[0]
    class_pred = health_classifier.predict(aqi_scaled)[0]

    state['aqi_data'] = {
        'aqi': round(aqi_value, 2),
        'category': get_category(aqi_value),
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

    return state

# Node 2: Predict Festival Surge
def predict_festival_surge(state: HospitalState) -> HospitalState:
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
                'days_until': (festival_date - today).days,
                'ed_surge_multiplier': forecast['ed_volume_change'],
                'blood_units_needed': surge_pred['predicted_blood_units'],
                'alert_level': surge_pred['alert_level']
            })

        state['festival_predictions'] = {'festivals': festival_data, 'total_festivals': len(festival_data)}
        state['surge_predictions'] = {'model_available': True}

    except Exception as e:
        state['festival_predictions'] = {}
        state['surge_predictions'] = {}

    return state

# Node 3: Predict Patient Count
def predict_patient_count(state: HospitalState) -> HospitalState:
    if not SET3_AVAILABLE:
        state['patient_count_forecast'] = {}
        return state

    latest_prev = patient_df["Total_Patients"].iloc[-1]
    base_pred = patient_model.predict([[latest_prev]])[0]
    aqi = state.get('aqi_data', {}).get('aqi', 100)
    pollution_impact = max(0, (aqi - 100) * 0.2)

    state['patient_count_forecast'] = {
        'base_patients': int(base_pred),
        'pollution_adjustment': int(pollution_impact),
        'total_expected_patients': round(base_pred + pollution_impact)
    }
    return state

# Node 4: Calculate Bed Capacity
def calculate_bed_capacity(state: HospitalState) -> HospitalState:
    TOTAL_BEDS = 200
    CURRENT_OCCUPANCY = 150
    available_beds = TOTAL_BEDS - CURRENT_OCCUPANCY

    daily_health_admissions = state.get('health_predictions', {}).get('hospital_admissions', 0)
    patient_volume = state.get('patient_count_forecast', {}).get('total_expected_patients', 0)

    festival_surge_multiplier = 1.0
    festivals = state.get('festival_predictions', {}).get('festivals', [])
    if festivals:
        nearest = min(festivals, key=lambda x: x['days_until'])
        festival_surge_multiplier = nearest['ed_surge_multiplier']

    surge_adjusted_admissions = int((daily_health_admissions + patient_volume) * festival_surge_multiplier)
    beds_needed = surge_adjusted_admissions * (4 / 7)

    capacity_status = "Normal"
    if beds_needed > available_beds:
        capacity_status = "🚨 OVERCAPACITY ALERT"
    elif beds_needed > available_beds * 0.8:
        capacity_status = "⚠️ Near Capacity"

    state['bed_capacity_plan'] = {
        'total_beds': TOTAL_BEDS,
        'available_beds': available_beds,
        'daily_expected_admissions': surge_adjusted_admissions,
        'beds_needed_weekly': int(beds_needed),
        'capacity_status': capacity_status
    }
    return state

# Node 5: Calculate Resources
def calculate_resources(state: HospitalState) -> HospitalState:
    respiratory = state.get('health_predictions', {}).get('respiratory_cases', 0)
    cardiovascular = state.get('health_predictions', {}).get('cardiovascular_cases', 0)
    admissions = state.get('bed_capacity_plan', {}).get('daily_expected_admissions', 0)

    nurses_needed = max(10, admissions // 5)
    doctors_needed = max(5, admissions // 15)
    ventilators_needed = max(5, respiratory // 3)
    cardiac_monitors = max(5, cardiovascular // 2)

    blood_units = 0
    festivals = state.get('festival_predictions', {}).get('festivals', [])
    if festivals:
        blood_units = festivals[0].get('blood_units_needed', 0)

    state['resource_allocation'] = {
        'staffing': {
            'nurses_required': nurses_needed,
            'doctors_required': doctors_needed
        },
        'equipment': {
            'ventilators': ventilators_needed,
            'cardiac_monitors': cardiac_monitors
        },
        'supplies': {
            'blood_units': blood_units
        }
    }
    return state

# Node 6: Generate Alerts
def generate_alerts(state: HospitalState) -> HospitalState:
    alerts = []

    aqi = state.get('aqi_data', {}).get('aqi', 0)
    if aqi > 200:
        alerts.append(f"🔴 CRITICAL: AQI {aqi} - Respiratory surge expected")

    capacity = state.get('bed_capacity_plan', {}).get('capacity_status', '')
    if 'OVERCAPACITY' in capacity:
        alerts.append("🔴 CRITICAL: Bed capacity exceeded")

    festivals = state.get('festival_predictions', {}).get('festivals', [])
    for fest in festivals:
        if fest['days_until'] <= 7:
            alerts.append(f"🟡 ALERT: {fest['festival']} in {fest['days_until']} days")

    state['critical_alerts'] = alerts
    return state

# Node 7: LLM Insights Generator
def generate_llm_insights(state: HospitalState) -> HospitalState:
    print("\n🤖 [LLM] Generating AI Insights...")

    if not LLM_AVAILABLE:
        state['llm_insights'] = {'error': 'LLM not available'}
        return state

    context = f"""You are a hospital resource management AI assistant. Analyze this data and provide:
1. Executive Summary (2-3 sentences)
2. Top 3 Priority Actions for hospital administrators
3. Risk Assessment (Low/Medium/High/Critical)

DATA:
- AQI: {state.get('aqi_data', {}).get('aqi', 'N/A')} ({state.get('aqi_data', {}).get('category', 'N/A')})
- Hospital Admissions Expected: {state.get('bed_capacity_plan', {}).get('daily_expected_admissions', 'N/A')}
- Bed Capacity Status: {state.get('bed_capacity_plan', {}).get('capacity_status', 'N/A')}
- Available Beds: {state.get('bed_capacity_plan', {}).get('available_beds', 'N/A')}
- Respiratory Cases: {state.get('health_predictions', {}).get('respiratory_cases', 'N/A')}
- Critical Alerts: {', '.join(state.get('critical_alerts', []))}

Festivals: {json.dumps(state.get('festival_predictions', {}).get('festivals', []), indent=2)}

Provide concise, actionable insights."""

    try:
        response = llm.generate_content(context)
        insights_text = response.text

        state['llm_insights'] = {
            'summary': insights_text,
            'generated_at': datetime.now().isoformat(),
            'model': 'gemini-pro'
        }
        print(f"   ✅ Generated insights ({len(insights_text)} chars)")
    except Exception as e:
        print(f"   ❌ LLM error: {e}")
        state['llm_insights'] = {'error': str(e)}

    return state

# Node 8: Conversational Query Handler
def handle_user_query(state: HospitalState) -> HospitalState:
    user_query = state.get('user_query', '')
    if not user_query or not LLM_AVAILABLE:
        return state

    print(f"\n💬 User Query: {user_query}")

    context = f"""You are a hospital resource management assistant. Answer based on current data.

HOSPITAL STATUS:
{json.dumps({
    'aqi': state.get('aqi_data'),
    'health_predictions': state.get('health_predictions'),
    'bed_capacity': state.get('bed_capacity_plan'),
    'resources': state.get('resource_allocation'),
    'alerts': state.get('critical_alerts')
}, indent=2)}

USER QUESTION: {user_query}

Provide a clear, concise answer with specific data and actions."""

    try:
        response = llm.generate_content(context)
        answer = response.text
        print(f"\n🤖 Assistant: {answer}")
    except Exception as e:
        print(f"\n❌ Query error: {e}")

    return state

# Node 9: Epidemic Scanner
def scrape_epidemic_news(state: HospitalState) -> HospitalState:
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
    except:
        alerts = []

    state['epidemic_alerts'] = alerts
    return state

# Router
def route_task(state: HospitalState) -> str:
    task = state.get('task_type', 'full_analysis')
    if task in ['full_analysis', 'user_query']:
        return 'daily_health'
    return END

# Build Workflow
workflow = StateGraph(HospitalState)

workflow.add_node("daily_health", predict_daily_health)
workflow.add_node("festival_surge", predict_festival_surge)
workflow.add_node("patient_count", predict_patient_count)
workflow.add_node("bed_capacity", calculate_bed_capacity)
workflow.add_node("resources", calculate_resources)
workflow.add_node("alerts", generate_alerts)
workflow.add_node("llm_insights", generate_llm_insights)
workflow.add_node("query_handler", handle_user_query)
workflow.add_node("epidemic_scan", scrape_epidemic_news)

workflow.add_conditional_edges(START, route_task)
workflow.add_edge("daily_health", "festival_surge")
workflow.add_edge("festival_surge", "patient_count")
workflow.add_edge("patient_count", "epidemic_scan")
workflow.add_edge("epidemic_scan", "bed_capacity")
workflow.add_edge("bed_capacity", "resources")
workflow.add_edge("resources", "alerts")
workflow.add_edge("alerts", "llm_insights")
workflow.add_edge("llm_insights", "query_handler")
workflow.add_edge("query_handler", END)

hospital_agent = workflow.compile()

# Test Function
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏥 LLM-ENHANCED HOSPITAL RESOURCE MANAGEMENT SYSTEM")
    print("="*70)

    # Full analysis
    print("\n[TEST 1] Full Analysis with LLM Insights")
    result = hospital_agent.invoke({"task_type": "full_analysis"})

    print("\n" + "="*70)
    print("📊 EXECUTIVE SUMMARY")
    print("="*70)

    print("\n🤖 AI INSIGHTS:")
    insights = result.get('llm_insights', {})
    if 'summary' in insights:
        print(insights['summary'])
    else:
        print("   LLM insights not available")

    print("\n⚠️ CRITICAL ALERTS:")
    alerts = result.get('critical_alerts', [])
    if alerts:
        for alert in alerts:
            print(f"   {alert}")
    else:
        print("   No critical alerts")

    print("\n🏥 BED CAPACITY:")
    bed_plan = result.get('bed_capacity_plan', {})
    print(f"   Status: {bed_plan.get('capacity_status', 'N/A')}")
    print(f"   Available: {bed_plan.get('available_beds', 'N/A')} beds")
    print(f"   Expected Admissions: {bed_plan.get('daily_expected_admissions', 'N/A')}/day")

    # Conversational query test
    print("\n" + "="*70)
    print("[TEST 2] Conversational Query")
    print("="*70)

    query_result = hospital_agent.invoke({
        "task_type": "user_query",
        "user_query": "Should we prepare for increased respiratory cases tomorrow?"
    })

    print("\n" + "="*70)
