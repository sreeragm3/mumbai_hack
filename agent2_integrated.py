# agent2_integrated.py - Integrated Health Surveillance Agent
import operator
from typing import Annotated, TypedDict, List
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

# Import Set 1 Hospital Surge modules from chinmay_mumabaihacks folder
try:
    from data_fetcher import (
        fetch_festival_dates_calendarific,
        get_events_for_alerting,
        get_forecast_inputs
    )
    from model_trainer import (
        load_trained_model,
        predict_surge_needs
    )
    from config import TARGET_YEAR, ALERT_LEAD_TIMES
    print("✅ Loaded Set 1 modules from chinmay_mumabaihacks/")
    SET1_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Could not load Set 1 modules: {e}")
    SET1_AVAILABLE = False

# Load all ML models (from root directory)
try:
    aqi_model = joblib.load('air_quality_model/aqi_model.pkl')
    city_encoder = joblib.load('air_quality_model/city_label_encoder.pkl')
    health_regressor = joblib.load('health_model/regressor_model.pkl')
    health_classifier = joblib.load('health_model/classifier_model.pkl')
    scaler = joblib.load('health_model/scaler.pkl')
    print("✅ Loaded AQI and Health models")
except Exception as e:
    print(f"❌ Error loading AQI/Health models: {e}")
    sys.exit(1)

# Load Hospital Surge model (from chinmay_mumabaihacks/data/)
surge_model = None
surge_features = None
if SET1_AVAILABLE:
    try:
        surge_model_path = os.path.join(hsr_dir, 'data', 'surge_predictor.joblib')
        surge_model = load_trained_model(surge_model_path)  # ✅ Correct - single return value
        surge_features = ['ed_volume_change_percent', 'trauma_cases', 'burn_cases', 
                         'respiratory_cases', 'festival_Diwali', 'festival_Holi']  # Define manually
        print("✅ Hospital Surge model loaded successfully")
    except Exception as e:
        print(f"⚠️ Hospital Surge model not available: {e}")

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    aqi_data: dict
    health_predictions: dict
    festival_predictions: dict
    surge_predictions: dict
    epidemic_alerts: List[str]
    task_type: str

# Helper Functions
def get_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_health_impact_class(score):
    if score == 0:
        return "Low"
    elif score == 1:
        return "Moderate"
    elif score == 2:
        return "High"
    else:
        return "Severe"

# Node 1: Predict Daily AQI
def predict_aqi(state: AgentState) -> AgentState:
    """Predict AQI for Delhi using current date features"""
    print("\n📊 Predicting AQI for Delhi...")
    
    now = datetime.now()
    
    # Use the exact same order and names as in your original agent2.py
    features = pd.DataFrame([[
        now.year,
        now.month,
        now.day,
        now.weekday(),
        city_encoder.transform(['Delhi'])[0]
    ]], columns=['Year', 'Month', 'Day', 'DayOfWeek', 'City_encoded'])
    
    # Check what features the model expects
    if hasattr(aqi_model, 'feature_names_in_'):
        expected_features = aqi_model.feature_names_in_
        print(f"Model expects features: {list(expected_features)}")
        print(f"We are providing: {list(features.columns)}")
        
        # Reorder columns to match expected order
        features = features[expected_features]
    
    aqi_value = aqi_model.predict(features)[0]
    category = get_category(aqi_value)
    
    state['aqi_data'] = {
        'aqi': round(aqi_value, 2),
        'category': category,
        'date': now.strftime('%Y-%m-%d'),
        'city': 'Delhi'
    }
    
    print(f"✅ Predicted AQI: {aqi_value:.2f} ({category})")
    return state


# Node 2: Predict Health Impact
def predict_health_impact(state: AgentState) -> AgentState:
    """Predict health impacts based on AQI"""
    print("\n🏥 Predicting Health Impacts...")
    
    aqi = state['aqi_data']['aqi']
    aqi_scaled = scaler.transform([[aqi]])
    
    reg_pred = health_regressor.predict(aqi_scaled)[0]
    class_pred = health_classifier.predict(aqi_scaled)[0]
    
    state['health_predictions'] = {
        'respiratory_cases': int(reg_pred[0]),
        'cardiovascular_cases': int(reg_pred[1]),
        'hospital_admissions': int(reg_pred[2]),
        'health_impact_score': round(reg_pred[3], 2),
        'health_impact_class': get_health_impact_class(class_pred)
    }
    
    print(f"✅ Health Impact: {state['health_predictions']['health_impact_class']}")
    print(f"   - Respiratory Cases: {state['health_predictions']['respiratory_cases']}")
    print(f"   - Hospital Admissions: {state['health_predictions']['hospital_admissions']}")
    
    return state

# Node 3: Enhanced Festival Forecast with Hospital Surge Prediction
def predict_festival_surge(state: AgentState) -> AgentState:
    """Fetch upcoming festivals and predict hospital resource needs"""
    print("\n🎉 Fetching Upcoming Festivals & Predicting Hospital Surge...")
    
    if not SET1_AVAILABLE:
        print("⚠️ Set 1 modules not available. Using fallback mode.")
        state['festival_predictions'] = {
            'festivals': [{
                'festival': 'Error',
                'recommendation': 'Set 1 modules not loaded'
            }],
            'total_festivals': 0
        }
        state['surge_predictions'] = {'model_available': False}
        return state
    
    try:
        # Fetch real festival data from Calendarific
        calendar_df = fetch_festival_dates_calendarific(TARGET_YEAR)
        
        # Get festivals needing alerts
        today = datetime.now().date()
        events_to_alert = get_events_for_alerting(calendar_df, today)

        
        if not events_to_alert:
            print("ℹ️ No festivals in alert window (30 days or 7 days)")
            state['festival_predictions'] = {
                'festivals': [],
                'total_festivals': 0,
                'message': 'No festivals in alert window'
            }
            state['surge_predictions'] = {'model_available': surge_model is not None}
            return state
        
        # Process each upcoming festival
        festival_data = []
        for event in events_to_alert:
            festival_name = event['festival_name']
            festival_date = event['festival_date']
            days_until = (festival_date - today).days
            
            print(f"\n📅 Processing: {festival_name} (in {days_until} days)")
            
            # Get forecast inputs for this festival
            forecast = get_forecast_inputs(festival_name)
            
            # Predict hospital surge needs using Set 1 model
            surge_pred = None
            if surge_model and surge_features:
                surge_pred = predict_surge_needs(
                    surge_model,
                    surge_features,
                    festival_name,
                    forecast['ed_volume_change'],
                    forecast['trauma_cases']
                )
                print(f"   🏥 Blood Units Needed: {surge_pred['predicted_blood_units']}")
                print(f"   ⚠️ Alert Level: {surge_pred['alert_level']}")
            
            festival_data.append({
                'festival': festival_name,
                'date': festival_date.strftime('%Y-%m-%d'),
                'days_until': days_until,
                'expected_aqi': forecast['aqi_level'],
                'ed_surge': f"{forecast['ed_volume_change']:.2f}x",
                'blood_units_needed': surge_pred['predicted_blood_units'] if surge_pred else 'N/A',
                'alert_level': surge_pred['alert_level'] if surge_pred else 'N/A',
                'recommendation': surge_pred['recommendation'] if surge_pred else 'Model not available'
            })
        
        state['festival_predictions'] = {
            'festivals': festival_data,
            'total_festivals': len(festival_data)
        }
        
        state['surge_predictions'] = {
            'model_available': surge_model is not None,
            'predictions_made': len(festival_data)
        }
        
        print(f"\n✅ Processed {len(festival_data)} festival(s) with surge predictions")
        
    except Exception as e:
        print(f"⚠️ Festival prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        state['festival_predictions'] = {
            'festivals': [],
            'total_festivals': 0
        }
        state['surge_predictions'] = {'model_available': False, 'error': str(e)}
    
    return state

# Node 4: Daily Epidemic Scraper
def scrape_epidemic_news(state: AgentState) -> AgentState:
    """Scrape WHO disease outbreak news"""
    print("\n🦠 Scraping Epidemic News...")
    
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
        
        print(f"✅ Found {len(alerts)} potential alerts")
        
    except Exception as e:
        print(f"⚠️ Scraping failed: {e}")
        alerts = [{'headline': 'Unable to fetch epidemic data', 'severity': 'ERROR'}]
    
    state['epidemic_alerts'] = alerts
    return state

# Router: Choose task path
def route_task(state: AgentState) -> str:
    """Route to appropriate task based on task_type"""
    task = state.get('task_type', 'daily_aqi')
    print(f"\n🔀 Routing to: {task}")
    
    if task == 'daily_aqi':
        return 'predict_aqi'
    elif task == 'weekly_festival':
        return 'predict_festival'
    elif task == 'daily_epidemic':
        return 'scrape_epidemic'
    else:
        return END

# Build Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("predict_aqi", predict_aqi)
workflow.add_node("predict_health", predict_health_impact)
workflow.add_node("predict_festival", predict_festival_surge)
workflow.add_node("scrape_epidemic", scrape_epidemic_news)

# Add edges
workflow.add_conditional_edges(START, route_task)
workflow.add_edge("predict_aqi", "predict_health")
workflow.add_edge("predict_health", END)
workflow.add_edge("predict_festival", END)
workflow.add_edge("scrape_epidemic", END)

# Compile
agent = workflow.compile()

# Test function
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Integrated Health Surveillance Agent")
    print("="*60)
    
    # Test daily AQI
    print("\n[TEST 1] Daily AQI Task")
    result = agent.invoke({"task_type": "daily_aqi", "messages": []})
    print("\n📋 Daily AQI Result:")
    print(json.dumps(result.get('aqi_data'), indent=2))
    print("\n📋 Health Predictions:")
    print(json.dumps(result.get('health_predictions'), indent=2))
    
    # Test festival prediction
    print("\n[TEST 2] Festival Surge Prediction Task")
    result = agent.invoke({"task_type": "weekly_festival", "messages": []})
    print("\n📋 Festival Predictions:")
    print(json.dumps(result.get('festival_predictions'), indent=2))
    
    # Test epidemic scanner
    print("\n[TEST 3] Epidemic Scanner Task")
    result = agent.invoke({"task_type": "daily_epidemic", "messages": []})
    print("\n📋 Epidemic Alerts:")
    if result.get('epidemic_alerts'):
        for alert in result['epidemic_alerts'][:3]:
            print(f"   - [{alert.get('severity')}] {alert.get('headline')}")
