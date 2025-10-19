# local_scheduler.py - Updated for Integrated Agent
import schedule
import time
from legacy.agent2_integrated import agent

def run_daily_aqi():
    """Run daily AQI and health impact prediction"""
    print("\n" + "="*60)
    print("🔄 Running Daily AQI & Health Task...")
    print("="*60)
    result = agent.invoke({"task_type": "daily_aqi", "messages": []})

    # Log results
    print("\n📊 AQI Summary:")
    print(f"   City: {result['aqi_data']['city']}")
    print(f"   AQI: {result['aqi_data']['aqi']} ({result['aqi_data']['category']})")
    print(f"\n🏥 Health Impact:")
    print(f"   Hospital Admissions: {result['health_predictions']['hospital_admissions']}")
    print(f"   Respiratory Cases: {result['health_predictions']['respiratory_cases']}")
    print(f"   Impact Level: {result['health_predictions']['health_impact_class']}")

def run_weekly_festival():
    """Run weekly festival and hospital surge prediction"""
    print("\n" + "="*60)
    print("🔄 Running Weekly Festival & Surge Prediction...")
    print("="*60)
    result = agent.invoke({"task_type": "weekly_festival", "messages": []})

    # Log results
    if result.get('festival_predictions', {}).get('festivals'):
        print(f"\n🎉 Found {result['festival_predictions']['total_festivals']} upcoming festival(s):")
        for fest in result['festival_predictions']['festivals']:
            print(f"\n   📅 {fest['festival']} - {fest['date']}")
            print(f"      Days Until: {fest['days_until']}")
            print(f"      Expected ED Surge: {fest['ed_surge']}")
            print(f"      Blood Units Needed: {fest['blood_units_needed']}")
            print(f"      Alert Level: {fest['alert_level']}")
            print(f"      Recommendation: {fest['recommendation']}")
    else:
        print("\nℹ️ No festivals in alert window (30 days or 7 days out)")

def run_daily_epidemic():
    """Run daily epidemic news scraper"""
    print("\n" + "="*60)
    print("🔄 Running Daily Epidemic Scraper...")
    print("="*60)
    result = agent.invoke({"task_type": "daily_epidemic", "messages": []})

    # Log results
    alerts = result.get('epidemic_alerts', [])
    if alerts and alerts[0].get('severity') != 'ERROR':
        print(f"\n🦠 Found {len(alerts)} potential epidemic alert(s):")
        for alert in alerts:
            print(f"   - [{alert['severity']}] {alert['headline']}")
    else:
        print("\nℹ️ No epidemic alerts detected or scraping failed")

# Schedule tasks
schedule.every().day.at("09:00").do(run_daily_aqi)          # 9 AM daily
schedule.every().monday.at("10:00").do(run_weekly_festival)  # 10 AM Monday
schedule.every().day.at("08:00").do(run_daily_epidemic)     # 8 AM daily

print("=" * 70)
print("✅ Integrated Health Surveillance Scheduler Started")
print("=" * 70)
print("\n📅 Scheduled Tasks:")
print("   🌫️  Daily AQI & Health Impact: Every day at 09:00")
print("   🎉 Festival & Hospital Surge: Every Monday at 10:00")
print("   🦠 Epidemic News Scanner: Every day at 08:00")
print("\n⏰ Waiting for scheduled tasks... (Press Ctrl+C to stop)")

while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
