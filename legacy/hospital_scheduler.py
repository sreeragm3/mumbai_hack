# hospital_scheduler.py - Automated Hospital Resource Management Scheduler
"""
Automated scheduling for Integrated Hospital Resource Management System

Runs comprehensive analysis combining:
- Set 1: Festival Surge Predictions
- Set 2: Daily AQI & Health Impact
- Set 3: Patient Count Forecasts

Schedule:
- Daily Full Analysis: 6:00 AM (before morning rounds)
- Capacity Check: 2:00 PM (mid-day update)
- Weekly Festival Review: Monday 8:00 AM
- Epidemic Scan: Every 6 hours
"""

import schedule
import time
import json
from datetime import datetime
from legacy.hospital_resource_manager import hospital_agent

def log_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def save_report_to_file(result, report_type):
    """Save analysis results to timestamped JSON file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"reports/{report_type}_{timestamp}.json"

    import os
    os.makedirs('reports', exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return filename

def run_full_analysis():
    """
    Complete hospital resource analysis
    Runs: AQI → Health Impact → Festival Surge → Patient Count → Bed Capacity → Resources
    """
    print("\n" + "="*70)
    print(f"🏥 FULL HOSPITAL RESOURCE ANALYSIS - {log_timestamp()}")
    print("="*70)

    try:
        result = hospital_agent.invoke({"task_type": "full_analysis", "messages": []})

        # Print Executive Summary
        print("\n" + "="*70)
        print("📊 EXECUTIVE SUMMARY")
        print("="*70)

        # Critical Alerts
        alerts = result.get('critical_alerts', [])
        if alerts:
            print("\n⚠️ CRITICAL ALERTS:")
            for alert in alerts:
                print(f"   {alert}")
        else:
            print("\n✅ No critical alerts at this time")

        # Bed Capacity
        bed_plan = result.get('bed_capacity_plan', {})
        print(f"\n🏥 BED CAPACITY:")
        print(f"   Status: {bed_plan.get('capacity_status', 'Unknown')}")
        print(f"   Available: {bed_plan.get('available_beds', 0)} beds")
        print(f"   Expected Admissions: {bed_plan.get('daily_expected_admissions', 0)}/day")
        print(f"   Weekly Demand: {bed_plan.get('beds_needed_weekly', 0)} beds")

        # Resource Allocation
        resources = result.get('resource_allocation', {})
        staffing = resources.get('staffing', {})
        equipment = resources.get('equipment', {})
        supplies = resources.get('supplies', {})

        print(f"\n📋 RESOURCE REQUIREMENTS:")
        print(f"   Staff: {staffing.get('nurses_required', 0)} nurses, {staffing.get('doctors_required', 0)} doctors")
        print(f"   Equipment: {equipment.get('ventilators', 0)} ventilators, {equipment.get('cardiac_monitors', 0)} monitors")
        if supplies.get('blood_units', 0) > 0:
            print(f"   Blood Supply: {supplies.get('blood_units', 0)} units")

        # AQI & Health
        aqi_data = result.get('aqi_data', {})
        health = result.get('health_predictions', {})
        print(f"\n🌫️ AIR QUALITY:")
        print(f"   AQI: {aqi_data.get('aqi', 0)} ({aqi_data.get('category', 'Unknown')})")
        print(f"   Health Impact: {health.get('health_impact_class', 'Unknown')}")
        print(f"   Respiratory Cases: {health.get('respiratory_cases', 0)}")

        # Festival Impact
        festivals = result.get('festival_predictions', {}).get('festivals', [])
        if festivals:
            print(f"\n🎉 UPCOMING FESTIVALS:")
            for fest in festivals[:3]:  # Show up to 3
                print(f"   {fest['festival']}: {fest['days_until']} days - {fest['alert_level']}")

        # Save report
        filename = save_report_to_file(result, 'full_analysis')
        print(f"\n💾 Report saved: {filename}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def run_capacity_check():
    """
    Quick capacity check (mid-day update)
    Focus on current bed availability and immediate needs
    """
    print("\n" + "="*70)
    print(f"🏥 MID-DAY CAPACITY CHECK - {log_timestamp()}")
    print("="*70)

    try:
        result = hospital_agent.invoke({"task_type": "full_analysis", "messages": []})

        bed_plan = result.get('bed_capacity_plan', {})
        resources = result.get('resource_allocation', {})

        print(f"\n   Capacity Status: {bed_plan.get('capacity_status', 'Unknown')}")
        print(f"   Available Beds: {bed_plan.get('available_beds', 0)}")
        print(f"   Expected Today: {bed_plan.get('daily_expected_admissions', 0)} admissions")

        alerts = result.get('critical_alerts', [])
        if alerts:
            print(f"\n   ⚠️ Alerts: {len(alerts)}")
            for alert in alerts[:2]:
                print(f"      {alert}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error during capacity check: {e}")

def run_epidemic_scan():
    """
    Quick epidemic monitoring scan
    Checks WHO for disease outbreak alerts
    """
    print("\n" + "="*70)
    print(f"🦠 EPIDEMIC SCAN - {log_timestamp()}")
    print("="*70)

    try:
        result = hospital_agent.invoke({"task_type": "epidemic_only", "messages": []})

        alerts = result.get('epidemic_alerts', [])
        if alerts and alerts[0].get('severity') != 'ERROR':
            print(f"\n   Found {len(alerts)} potential epidemic alert(s):")
            for alert in alerts[:3]:
                print(f"   🔴 {alert.get('headline', 'Unknown')}")
        else:
            print("\n   ✅ No epidemic alerts detected")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error during epidemic scan: {e}")

def run_weekly_festival_review():
    """
    Weekly detailed festival surge review
    Comprehensive planning for upcoming festivals
    """
    print("\n" + "="*70)
    print(f"🎉 WEEKLY FESTIVAL SURGE REVIEW - {log_timestamp()}")
    print("="*70)

    try:
        result = hospital_agent.invoke({"task_type": "full_analysis", "messages": []})

        festivals = result.get('festival_predictions', {}).get('festivals', [])

        if not festivals:
            print("\n   ℹ️ No festivals in alert window (next 30 days)")
        else:
            print(f"\n   📅 {len(festivals)} Festival(s) Requiring Preparation:")

            for fest in festivals:
                print(f"\n   🎊 {fest['festival']}")
                print(f"      Date: {fest['date']} ({fest['days_until']} days)")
                print(f"      ED Surge: {fest['ed_surge_multiplier']:.2f}x normal volume")
                print(f"      Blood Units: {fest['blood_units_needed']} units")
                print(f"      Alert Level: {fest['alert_level']}")

                # Recommendations based on days until
                if fest['days_until'] <= 7:
                    print(f"      🚨 ACTION: Finalize staffing, stock blood bank")
                elif fest['days_until'] <= 14:
                    print(f"      ⚠️ ACTION: Confirm vendor orders, schedule extra staff")
                else:
                    print(f"      📝 ACTION: Begin planning, send alerts to departments")

        # Save detailed festival report
        if festivals:
            filename = save_report_to_file(result, 'festival_review')
            print(f"\n   💾 Detailed report saved: {filename}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error during festival review: {e}")

# Schedule tasks
print("="*70)
print("🏥 HOSPITAL RESOURCE MANAGEMENT SCHEDULER")
print("="*70)
print("\n📅 Scheduled Tasks:\n")

# Daily full analysis - 6:00 AM (before morning rounds)
schedule.every().day.at("06:00").do(run_full_analysis)
print("   🌅 Daily Full Analysis: 6:00 AM")
print("      → Complete resource planning for the day")

# Mid-day capacity check - 2:00 PM
schedule.every().day.at("14:00").do(run_capacity_check)
print("\n   ☀️ Mid-Day Capacity Check: 2:00 PM")
print("      → Quick status update on bed availability")

# Weekly festival review - Monday 8:00 AM
schedule.every().monday.at("08:00").do(run_weekly_festival_review)
print("\n   🎉 Weekly Festival Review: Monday 8:00 AM")
print("      → Comprehensive festival surge planning")

# Epidemic scans - Every 6 hours (6 AM, 12 PM, 6 PM, 12 AM)
schedule.every().day.at("06:00").do(run_epidemic_scan)
schedule.every().day.at("12:00").do(run_epidemic_scan)
schedule.every().day.at("18:00").do(run_epidemic_scan)
schedule.every().day.at("00:00").do(run_epidemic_scan)
print("\n   🦠 Epidemic Scans: Every 6 hours (6 AM, 12 PM, 6 PM, 12 AM)")
print("      → WHO disease outbreak monitoring")

print("\n" + "="*70)
print("\n✅ Scheduler started. Reports saved to 'reports/' directory")
print("⏰ Waiting for scheduled tasks... (Press Ctrl+C to stop)\n")

# Run immediately on startup
print("🚀 Running initial analysis...")
run_full_analysis()

# Main loop
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
