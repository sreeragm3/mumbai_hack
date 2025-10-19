# hospital_scheduler_llm.py - Automated Scheduler for LLM-Enhanced Hospital Resource Management
"""
Automated scheduling with Gemini AI insights for hospital resource management.

Schedule:
- Daily Morning Briefing: 6:00 AM (before rounds) - Full AI analysis
- Mid-Day Check: 2:00 PM - Quick capacity update
- Evening Update: 8:00 PM - Shift change briefing with AI insights
- Weekly Festival Review: Monday 8:00 AM - Festival surge planning
- Epidemic Scans: Every 6 hours (24/7 monitoring)
"""

import schedule
import time
import json
from datetime import datetime
from hospital_resource_manager_llm import hospital_agent

def log_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def save_report_to_file(result, report_type):
    """Save analysis results to timestamped JSON and TXT files"""
    import os
    os.makedirs('reports', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_filename = f"reports/{report_type}_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:  # ADD encoding='utf-8'
        json.dump(result, f, indent=2, default=str)
    
    # Save human-readable TXT with AI insights
    txt_filename = f"reports/{report_type}_{timestamp}.txt"
    with open(txt_filename, 'w', encoding='utf-8') as f:  # ADD encoding='utf-8'
        f.write("="*70 + "\n")
        f.write(f"HOSPITAL RESOURCE MANAGEMENT REPORT\n")
        f.write(f"Report Type: {report_type}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")

        # AI Insights
        insights = result.get('llm_insights', {})
        if 'summary' in insights:
            f.write("🤖 AI INSIGHTS:\n")
            f.write("-"*70 + "\n")
            f.write(insights['summary'] + "\n\n")

        # Critical Alerts
        alerts = result.get('critical_alerts', [])
        f.write("⚠️ CRITICAL ALERTS:\n")
        f.write("-"*70 + "\n")
        if alerts:
            for alert in alerts:
                f.write(f"   {alert}\n")
        else:
            f.write("   ✅ No critical alerts\n")
        f.write("\n")

        # Bed Capacity
        bed_plan = result.get('bed_capacity_plan', {})
        f.write("🏥 BED CAPACITY PLAN:\n")
        f.write("-"*70 + "\n")
        f.write(f"   Status: {bed_plan.get('capacity_status', 'N/A')}\n")
        f.write(f"   Total Beds: {bed_plan.get('total_beds', 'N/A')}\n")
        f.write(f"   Available: {bed_plan.get('available_beds', 'N/A')} beds\n")
        f.write(f"   Expected Admissions: {bed_plan.get('daily_expected_admissions', 'N/A')}/day\n")
        f.write(f"   Weekly Demand: {bed_plan.get('beds_needed_weekly', 'N/A')} beds\n")
        f.write("\n")

        # Resource Allocation
        resources = result.get('resource_allocation', {})
        staffing = resources.get('staffing', {})
        equipment = resources.get('equipment', {})
        supplies = resources.get('supplies', {})

        f.write("📋 RESOURCE ALLOCATION:\n")
        f.write("-"*70 + "\n")
        f.write(f"   Staffing:\n")
        f.write(f"      Nurses Required: {staffing.get('nurses_required', 'N/A')}\n")
        f.write(f"      Doctors Required: {staffing.get('doctors_required', 'N/A')}\n")
        f.write(f"   Equipment:\n")
        f.write(f"      Ventilators: {equipment.get('ventilators', 'N/A')}\n")
        f.write(f"      Cardiac Monitors: {equipment.get('cardiac_monitors', 'N/A')}\n")
        if supplies.get('blood_units', 0) > 0:
            f.write(f"   Blood Supply: {supplies.get('blood_units', 0)} units needed\n")
        f.write("\n")

        # AQI & Health
        aqi_data = result.get('aqi_data', {})
        health = result.get('health_predictions', {})

        f.write("🌫️ AIR QUALITY & HEALTH IMPACT:\n")
        f.write("-"*70 + "\n")
        f.write(f"   AQI: {aqi_data.get('aqi', 'N/A')} ({aqi_data.get('category', 'N/A')})\n")
        f.write(f"   Health Impact: {health.get('health_impact_class', 'N/A')}\n")
        f.write(f"   Respiratory Cases: {health.get('respiratory_cases', 'N/A')}\n")
        f.write(f"   Cardiovascular Cases: {health.get('cardiovascular_cases', 'N/A')}\n")
        f.write(f"   Hospital Admissions: {health.get('hospital_admissions', 'N/A')}\n")
        f.write("\n")

        # Festivals
        festivals = result.get('festival_predictions', {}).get('festivals', [])
        if festivals:
            f.write("🎉 UPCOMING FESTIVALS:\n")
            f.write("-"*70 + "\n")
            for fest in festivals:
                f.write(f"   {fest['festival']}:\n")
                f.write(f"      Date: {fest['date']} ({fest['days_until']} days)\n")
                f.write(f"      ED Surge: {fest['ed_surge_multiplier']:.2f}x\n")
                f.write(f"      Blood Units: {fest['blood_units_needed']}\n")
                f.write(f"      Alert: {fest['alert_level']}\n")
            f.write("\n")

    return json_filename, txt_filename

def run_morning_briefing():
    """Complete morning briefing with AI insights (6:00 AM)"""
    print("\n" + "="*70)
    print(f"🌅 MORNING BRIEFING - {log_timestamp()}")
    print("="*70)

    try:
        result = hospital_agent.invoke({"task_type": "full_analysis"})

        # Print AI insights
        insights = result.get('llm_insights', {})
        if 'summary' in insights:
            print("\n🤖 AI ANALYSIS:")
            print(insights['summary'])

        # Print alerts
        alerts = result.get('critical_alerts', [])
        print("\n⚠️ ALERTS:")
        if alerts:
            for alert in alerts:
                print(f"   {alert}")
        else:
            print("   ✅ No critical alerts")

        # Print capacity
        bed_plan = result.get('bed_capacity_plan', {})
        print(f"\n🏥 CAPACITY: {bed_plan.get('capacity_status', 'N/A')}")
        print(f"   Available: {bed_plan.get('available_beds', 'N/A')} beds")
        print(f"   Expected: {bed_plan.get('daily_expected_admissions', 'N/A')} admissions")

        # Save reports
        json_file, txt_file = save_report_to_file(result, 'morning_briefing')
        print(f"\n💾 Reports saved:")
        print(f"   JSON: {json_file}")
        print(f"   TXT: {txt_file}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_midday_check():
    """Quick capacity check (2:00 PM)"""
    print("\n" + "="*70)
    print(f"☀️ MID-DAY CAPACITY CHECK - {log_timestamp()}")
    print("="*70)

    try:
        result = hospital_agent.invoke({"task_type": "full_analysis"})

        bed_plan = result.get('bed_capacity_plan', {})
        resources = result.get('resource_allocation', {})
        alerts = result.get('critical_alerts', [])

        print(f"\n   Status: {bed_plan.get('capacity_status', 'N/A')}")
        print(f"   Available: {bed_plan.get('available_beds', 'N/A')} beds")
        print(f"   Expected: {bed_plan.get('daily_expected_admissions', 'N/A')} admissions")

        if alerts:
            print(f"\n   ⚠️ {len(alerts)} Alert(s):")
            for alert in alerts[:2]:
                print(f"      {alert}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")

def run_evening_update():
    """Evening shift briefing with AI insights (8:00 PM)"""
    print("\n" + "="*70)
    print(f"🌙 EVENING SHIFT BRIEFING - {log_timestamp()}")
    print("="*70)

    try:
        result = hospital_agent.invoke({"task_type": "full_analysis"})

        # AI insights
        insights = result.get('llm_insights', {})
        if 'summary' in insights:
            print("\n🤖 AI SUMMARY FOR NIGHT SHIFT:")
            print(insights['summary'])

        # Key metrics
        bed_plan = result.get('bed_capacity_plan', {})
        resources = result.get('resource_allocation', {})

        print(f"\n📊 KEY METRICS:")
        print(f"   Capacity: {bed_plan.get('capacity_status', 'N/A')}")
        print(f"   Night Staff: {resources.get('staffing', {}).get('nurses_required', 'N/A')} nurses needed")

        # Save report
        json_file, txt_file = save_report_to_file(result, 'evening_update')
        print(f"\n💾 Shift report: {txt_file}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")

def run_weekly_festival_review():
    """Weekly festival review with AI recommendations (Monday 8:00 AM)"""
    print("\n" + "="*70)
    print(f"🎉 WEEKLY FESTIVAL SURGE REVIEW - {log_timestamp()}")
    print("="*70)

    try:
        result = hospital_agent.invoke({"task_type": "full_analysis"})

        festivals = result.get('festival_predictions', {}).get('festivals', [])

        if not festivals:
            print("\n   ℹ️ No festivals in alert window")
        else:
            print(f"\n   📅 {len(festivals)} Festival(s) Detected:")

            # Get AI insights on festivals
            insights = result.get('llm_insights', {})
            if 'summary' in insights:
                print("\n🤖 AI FESTIVAL ANALYSIS:")
                print(insights['summary'])

            print("\n   📋 FESTIVAL DETAILS:")
            for fest in festivals:
                print(f"\n   🎊 {fest['festival']}")
                print(f"      Date: {fest['date']} ({fest['days_until']} days)")
                print(f"      ED Surge: {fest['ed_surge_multiplier']:.2f}x")
                print(f"      Blood: {fest['blood_units_needed']} units")
                print(f"      Alert: {fest['alert_level']}")

                # Action timeline
                if fest['days_until'] <= 7:
                    print(f"      🚨 ACTION: Finalize staffing, stock blood bank")
                elif fest['days_until'] <= 14:
                    print(f"      ⚠️ ACTION: Confirm orders, schedule staff")
                else:
                    print(f"      📝 ACTION: Begin planning, alert departments")

            # Save detailed report
            json_file, txt_file = save_report_to_file(result, 'festival_review')
            print(f"\n   💾 Detailed report: {txt_file}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")

def run_epidemic_scan():
    """Quick epidemic monitoring scan"""
    print("\n" + "="*70)
    print(f"🦠 EPIDEMIC SCAN - {log_timestamp()}")
    print("="*70)

    try:
        result = hospital_agent.invoke({"task_type": "full_analysis"})

        alerts = result.get('epidemic_alerts', [])
        if alerts and alerts[0].get('severity') != 'ERROR':
            print(f"\n   ⚠️ {len(alerts)} Potential Alert(s):")
            for alert in alerts[:3]:
                print(f"      {alert.get('headline', 'Unknown')}")
        else:
            print("\n   ✅ No epidemic alerts detected")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}")

# Schedule tasks
print("="*70)
print("🏥 LLM-ENHANCED HOSPITAL SCHEDULER")
print("="*70)
print("\n📅 Scheduled Tasks:\n")

# Daily morning briefing with AI - 6:00 AM
schedule.every().day.at("06:00").do(run_morning_briefing)
print("   🌅 Morning Briefing: 6:00 AM (with AI insights)")

# Mid-day capacity check - 2:00 PM
schedule.every().day.at("14:00").do(run_midday_check)
print("   ☀️ Mid-Day Check: 2:00 PM")

# Evening shift briefing - 8:00 PM
schedule.every().day.at("20:00").do(run_evening_update)
print("   🌙 Evening Briefing: 8:00 PM (with AI insights)")

# Weekly festival review - Monday 8:00 AM
schedule.every().monday.at("08:00").do(run_weekly_festival_review)
print("   🎉 Festival Review: Monday 8:00 AM (AI recommendations)")

# Epidemic scans - Every 6 hours
schedule.every().day.at("00:00").do(run_epidemic_scan)
schedule.every().day.at("06:00").do(run_epidemic_scan)
schedule.every().day.at("12:00").do(run_epidemic_scan)
schedule.every().day.at("18:00").do(run_epidemic_scan)
print("   🦠 Epidemic Scans: Every 6 hours (24/7)")

print("\n" + "="*70)
print("\n✅ Scheduler started with AI-powered insights")
print("💾 Reports saved to 'reports/' directory (JSON + TXT)")
print("🤖 Gemini AI analysis included in all briefings")
print("⏰ Waiting for scheduled tasks... (Ctrl+C to stop)\n")

# Run initial morning briefing
print("🚀 Running initial briefing...")
run_morning_briefing()

# Main loop
while True:
    schedule.run_pending()
    time.sleep(60)
