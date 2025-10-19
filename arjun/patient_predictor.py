# --- Flask Web App for Patient Count Prediction with AI Agent & Gemini ---
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeRegressor
from flask import Flask, render_template_string
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool

# --- 1️⃣ Load API Key ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Google API key not found in .env file.")

# --- 2️⃣ Dummy Data ---
data = {
    "Date": pd.date_range("2025-10-01", periods=10),
    "City": ["Kochi"] * 10,
    "PM2.5": [45, 50, 70, 80, 60, 55, 95, 40, 75, 65],
    "PM10": [80, 90, 110, 100, 85, 95, 120, 75, 105, 90],
    "NO2": [60, 65, 70, 85, 78, 72, 88, 55, 90, 82],
    "SO2": [20, 25, 35, 40, 30, 28, 42, 22, 45, 38],
    "CO": [0.8, 1.0, 1.2, 1.5, 1.0, 0.9, 2.5, 0.6, 1.8, 1.2],
    "NH3": [200, 250, 300, 400, 350, 280, 420, 180, 390, 310],
    "Total_Patients": [35, 36, 38, 40, 39, 37, 45, 33, 43, 41],
}
df = pd.DataFrame(data)

# --- 3️⃣ CPCB Thresholds with disease mapping ---
limits = {
    "PM2.5": (60, "Respiratory"),
    "PM10": (100, "Respiratory"),
    "NO2": (80, "Respiratory"),
    "SO2": (80, "Respiratory"),
    "CO": (2, "Cardiac"),
    "NH3": (400, "Respiratory")
}

# --- 4️⃣ Train Decision Tree (past patient data only) ---
df["Prev_Patient"] = df["Total_Patients"].shift(1).fillna(method="bfill")
features = ["Prev_Patient"]
target = "Total_Patients"
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(df[features], df[target])

# --- 5️⃣ Prediction Function ---
def predict_next_day(df, model):
    # Base prediction using previous day only
    latest_prev = df["Total_Patients"].iloc[-1]
    base_pred = model.predict([[latest_prev]])[0]

    # Pollution adjustment
    latest_pollutants = df.iloc[-1]
    pollution_impact = 0
    abnormal_pollutants = []
    for pollutant, (limit, disease) in limits.items():
        if latest_pollutants[pollutant] > limit:
            excess = latest_pollutants[pollutant] - limit
            pollution_impact += excess * 0.1  # factor for impact
            abnormal_pollutants.append(f"{pollutant} ({disease})")

    # Final prediction & range
    final_pred = base_pred + pollution_impact
    final_rounded = round(final_pred)
    lower = max(0, final_rounded - 2)
    upper = final_rounded + 2

    if abnormal_pollutants:
        abnormal_text = "Abnormal pollutant: " + ", ".join(abnormal_pollutants) + ", may increase related cases."
    else:
        abnormal_text = "No abnormal pollutants detected."

    return final_rounded, (lower, upper), abnormal_text

# --- 6️⃣ LangChain Tool ---
def get_patient_prediction(_):
    pred, rng, abnormal_text = predict_next_day(df, model)
    return f"Expected patients tomorrow: {pred} (Range: {rng[0]}–{rng[1]}). {abnormal_text}"

tools = [
    Tool(
        name="Patient Predictor",
        func=get_patient_prediction,
        description="Predicts next day's patient count using past patient data, with pollution adjustment."
    )
]

# --- 7️⃣ Initialize Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0
)

# --- 8️⃣ Initialize Agent ---
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

# --- 9️⃣ Flask App ---
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Patient Count Prediction</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 100px; }
        h1 { color: #2c3e50; }
        p { font-size: 1.5em; color: #34495e; }
    </style>
</head>
<body>
    <h1>Hospital Patient Count Prediction</h1>
    <p>{{ prediction }}</p>
</body>
</html>
"""

@app.route("/")
def home():
    prediction = agent.run("Predict expected number of patients for tomorrow in Kochi.")
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
