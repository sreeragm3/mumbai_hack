# agent.py
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

# Load all ML models
aqi_model = joblib.load('air_quality_model/aqi_model.pkl')
city_encoder = joblib.load('air_quality_model/city_label_encoder.pkl')
health_regressor = joblib.load('health_model/regressor_model.pkl')
health_classifier = joblib.load('health_model/classifier_model.pkl')
scaler = joblib.load('health_model/scaler.pkl')

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    aqi_data: dict
    health_predictions: dict
    festival_predictions: dict
    epidemic_alerts: List[str]
    task_type: str  # 'daily_aqi', 'weekly_festival', 'daily_epidemic'

# Helper Functions
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

# Task 1: Fetch AQI and Predict Health Impact
def fetch_aqi_node(state: AgentState):
    """Fetches AQI data from external API"""
    try:
        # Replace with your actual AQI API endpoint
        city = "Delhi"
        date = datetime.now().strftime("%Y-%m-%d")
        
        # Example: Predict AQI using your model
        date_obj = pd.to_datetime(date)
        features = pd.DataFrame({
            "City_encoded": [city_encoder.transform([city])[0]],
            "Year": [date_obj.year],
            "Month": [date_obj.month],
            "Day": [date_obj.day],
            "DayOfWeek": [date_obj.dayofweek]
        })
        
        predicted_aqi = aqi_model.predict(features)[0]
        category = get_category(predicted_aqi)
        
        aqi_data = {
            "city": city,
            "date": date,
            "aqi": round(predicted_aqi, 2),
            "category": category
        }
        
        return {"aqi_data": aqi_data}
    except Exception as e:
        return {"aqi_data": {"error": str(e)}}

def predict_health_impact_node(state: AgentState):
    """Predicts health impacts from AQI data"""
    try:
        aqi_value = state["aqi_data"]["aqi"]
        
        input_df = pd.DataFrame({'AQI': [aqi_value]})
        aqi_scaled = scaler.transform(input_df)
        
        reg_preds = health_regressor.predict(aqi_scaled)[0]
        clf_pred = health_classifier.predict(aqi_scaled)[0]
        
        predictions = {
            'RespiratoryCases': float(reg_preds[0]),
            'CardiovascularCases': float(reg_preds[1]),
            'HospitalAdmissions': float(reg_preds[2]),
            'HealthImpactScore': float(reg_preds[3]),
            'HealthImpactClass': float(clf_pred)
        }
        
        print(f"✅ Daily AQI Health Prediction Complete:")
        print(f"   AQI: {aqi_value} ({state['aqi_data']['category']})")
        print(f"   Hospital Admissions: {predictions['HospitalAdmissions']:.0f}")
        
        return {"health_predictions": predictions}
    except Exception as e:
        return {"health_predictions": {"error": str(e)}}

# Task 2: Weekly Festival Prediction
def fetch_festivals_node(state: AgentState):
    """Fetches upcoming festivals in next week"""
    try:
        # Example: Mock festival data - replace with actual API
        next_week = datetime.now() + timedelta(days=7)
        festivals = [
            {"name": "Diwali", "date": next_week.strftime("%Y-%m-%d"), "expected_aqi_increase": 150}
        ]
        
        # Predict health impact for festival period
        predictions = []
        for festival in festivals:
            elevated_aqi = 200  # Example elevated AQI during festivals
            input_df = pd.DataFrame({'AQI': [elevated_aqi]})
            aqi_scaled = scaler.transform(input_df)
            
            reg_preds = health_regressor.predict(aqi_scaled)[0]
            
            predictions.append({
                "festival": festival["name"],
                "date": festival["date"],
                "expected_aqi": elevated_aqi,
                "hospital_admissions": float(reg_preds[2]),
                "respiratory_cases": float(reg_preds[0]),
                "medicine_requirement_multiplier": 2.5  # Example calculation
            })
        
        print(f"✅ Weekly Festival Prediction Complete:")
        for pred in predictions:
            print(f"   {pred['festival']}: {pred['hospital_admissions']:.0f} admissions expected")
        
        return {"festival_predictions": {"festivals": predictions}}
    except Exception as e:
        return {"festival_predictions": {"error": str(e)}}

# Task 3: Epidemic Scraper
def scrape_epidemic_news_node(state: AgentState):
    """Scrapes news for epidemic outbreaks"""
    try:
        alerts = []
        
        # Example: Scrape WHO or news sites
        keywords = ["outbreak", "epidemic", "disease", "pandemic"]
        
        # Mock scraping - replace with actual implementation
        url = "https://www.who.int/emergencies/disease-outbreak-news"
        
        # Simple example (in production, use proper scraping with error handling)
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Example: Find relevant keywords in headlines
            headlines = soup.find_all(['h2', 'h3', 'h4'])[:10]
            
            for headline in headlines:
                text = headline.get_text().lower()
                if any(keyword in text for keyword in keywords):
                    alerts.append({
                        "title": headline.get_text().strip(),
                        "timestamp": datetime.now().isoformat(),
                        "severity": "HIGH"
                    })
        except:
            # Fallback if scraping fails
            alerts.append({"info": "Scraping temporarily unavailable"})
        
        if len(alerts) > 0:
            print(f"🚨 EPIDEMIC ALERT: {len(alerts)} potential outbreaks detected")
            for alert in alerts[:3]:
                print(f"   - {alert.get('title', 'Alert')}")
        else:
            print("✅ No epidemic alerts detected")
        
        return {"epidemic_alerts": alerts}
    except Exception as e:
        return {"epidemic_alerts": [{"error": str(e)}]}

# Router function to decide which task to execute
def route_task(state: AgentState):
    """Routes to appropriate task based on task_type"""
    task_type = state.get("task_type", "daily_aqi")
    
    if task_type == "weekly_festival":
        return "festival"
    elif task_type == "daily_epidemic":
        return "epidemic"
    else:
        return "aqi"

# Build the Graph
def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes for each task
    workflow.add_node("fetch_aqi", fetch_aqi_node)
    workflow.add_node("predict_health", predict_health_impact_node)
    workflow.add_node("fetch_festivals", fetch_festivals_node)
    workflow.add_node("scrape_epidemic", scrape_epidemic_news_node)
    
    # Define edges with conditional routing
    workflow.add_conditional_edges(
        START,
        route_task,
        {
            "aqi": "fetch_aqi",
            "festival": "fetch_festivals",
            "epidemic": "scrape_epidemic"
        }
    )
    
    workflow.add_edge("fetch_aqi", "predict_health")
    workflow.add_edge("predict_health", END)
    workflow.add_edge("fetch_festivals", END)
    workflow.add_edge("scrape_epidemic", END)
    
    return workflow.compile()

# Create the agent
agent = create_agent_graph()

# Test individual tasks
if __name__ == "__main__":
    print("\n=== Testing Daily AQI Task ===")
    result1 = agent.invoke({"task_type": "daily_aqi", "messages": []})
    
    print("\n=== Testing Weekly Festival Task ===")
    result2 = agent.invoke({"task_type": "weekly_festival", "messages": []})
    
    print("\n=== Testing Daily Epidemic Scraper ===")
    result3 = agent.invoke({"task_type": "daily_epidemic", "messages": []})
