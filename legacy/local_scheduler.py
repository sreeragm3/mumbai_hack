# local_scheduler.py
import schedule
import time
from legacy.agent2 import agent

def run_daily_aqi():
    print("\n🔄 Running Daily AQI Task...")
    agent.invoke({"task_type": "daily_aqi", "messages": []})

def run_weekly_festival():
    print("\n🔄 Running Weekly Festival Task...")
    agent.invoke({"task_type": "weekly_festival", "messages": []})

def run_daily_epidemic():
    print("\n🔄 Running Daily Epidemic Scraper...")
    agent.invoke({"task_type": "daily_epidemic", "messages": []})

# Schedule tasks
schedule.every().day.at("19:24").do(run_daily_aqi)
schedule.every().monday.at("19:24").do(run_weekly_festival)
schedule.every().day.at("19:24").do(run_daily_epidemic)

print("✅ Scheduler started. Running tasks...")
while True:
    schedule.run_pending()
    time.sleep(60)
