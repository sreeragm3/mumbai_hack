"""
Main execution script for the Hospital Surge Readiness (HSR) Platform.
Orchestrates data fetching, processing, and alerting functionality.
"""

import os
import pandas as pd
import joblib
from datetime import date, datetime, timedelta

from config import TARGET_YEAR
from data_fetcher import (
    fetch_festival_dates_calendarific, 
    save_festival_calendar, 
    get_events_for_alerting,
    generate_historical_data,
    get_forecast_inputs
)
from model_trainer import (
    load_data,
    prepare_data,
    train_and_save_model,
    predict_surge_needs,
    load_trained_model
)


def generate_hsr_report(calendar_path: str, model_path: str, X_cols: list) -> None:
    """
    Generates the final Actionable Report for upcoming events.
    
    Args:
        calendar_path (str): Path to the festival calendar CSV file
        model_path (str): Path to the trained model file
        X_cols (list): List of feature column names from training
    """
    # Load resources
    calendar_df = pd.read_csv(calendar_path)
    model = joblib.load(model_path)
    
    # 1. Determine the window of upcoming events (e.g., events in the next 180 days)
    #    For simplicity, we will just target the two main 2026 festivals found: Holi and Diwali.
    
    report_data = []
    
    # Target 2026 events - only process festivals that exist in the calendar
    available_festivals = calendar_df['name'].unique()
    target_festivals = ['Holi', 'Diwali']
    
    for festival in target_festivals:
        # Check if festival exists in calendar data
        festival_matches = calendar_df[calendar_df['name'].str.contains(festival, case=False, na=False)]
        
        if festival_matches.empty:
            continue
            
        # 2. Get the date and lead time
        event_row = festival_matches.iloc[0]
        event_date = event_row['date']
        
        # 3. Get the forecasted inputs (from the new data_fetcher function)
        forecast_inputs = get_forecast_inputs(festival)
        
        # 4. Predict the critical resource (Blood Units)
        predicted_blood = predict_surge_needs(
            model, 
            X_cols,  # Pass the feature column list saved from training
            festival, 
            forecast_inputs['ed_surge_percent'], 
            forecast_inputs['trauma_cases_forecast']
        )
        
        # 5. Determine the action (Simple logic based on the prediction, e.g., using 25 as a baseline)
        baseline = 25
        increase_needed = max(0, predicted_blood - baseline)
        
        recommendation = ""
        if predicted_blood > baseline * 1.5:
            recommendation = "CRITICAL SURGE: Activate full disaster protocol (50%+ increase)"
        elif predicted_blood > baseline * 1.25:
            recommendation = "HIGH SURGE: Significant inventory increase (25-50%)"
        else:
            recommendation = "MODERATE SURGE: Review resource allocation (0-25% increase)"

        report_data.append({
            'Festival': festival,
            'Date': event_date,
            'Forecasted ED Surge': f"{forecast_inputs['ed_surge_percent']:.2f}x",
            'Predicted Blood Units': f"{predicted_blood:.1f}",
            'Increase Over Baseline': f"{increase_needed:.1f}",
            'HSR Recommendation': recommendation
        })

    # Generate the final report
    if not report_data:
        return
    
    report_df = pd.DataFrame(report_data)
    print("\n" + "="*70)
    print("                 HSR ACTIONABLE SURGE READINESS REPORT")
    print("="*70)
    print(report_df.to_markdown(index=False, numalign="left"))
    print("="*70)


def main():
    """
    Main execution function for the HSR Platform (Phase 1, 2 & 3).
    """
    print("Hospital Surge Readiness (HSR) Platform - Starting...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    try:
        # Fetch festival dates from Calendarific API
        print(f"\nFetching festival data for {TARGET_YEAR}...")
        festival_df = fetch_festival_dates_calendarific(TARGET_YEAR)
        
        if festival_df.empty:
            print("No high-risk festivals found for the specified year")
        else:
            print(f"Successfully fetched {len(festival_df)} high-risk festivals")
        
        # Save festival calendar to CSV
        save_festival_calendar(festival_df)
        
        # Check for current alerts
        if not festival_df.empty:
            today = date.today()
            current_alerts = get_events_for_alerting(festival_df, today)
            
            if current_alerts:
                print(f"\nFound {len(current_alerts)} alerts for today:")
                for alert in current_alerts:
                    print(f"  🚨 {alert['alert_type']}: {alert['festival_name']} on {alert['festival_date']}")
            else:
                print("\nNo alerts for today")
        
        print("\nPhase 1 Complete!")
        
        # Phase 2: Historical Surge Data and Prediction Model
        print("\nPhase 2: Historical Surge Data and Prediction Model")
        
        try:
            # Generate historical data
            generate_historical_data(num_years=5)
            
            # Load and prepare data for ML
            historical_df = load_data('data/historical_records.csv')
            X, y = prepare_data(historical_df)
            
            # Train and save the model
            model = train_and_save_model(X, y)
            
            print("\nPhase 2 Complete!")
            
            # Phase 3: Actionable Reporting and Integration
            print("\nPhase 3: Actionable Reporting and Integration")
            
            try:
                # Generate the final HSR actionable report
                generate_hsr_report('data/festival_calendar.csv', 'data/surge_predictor.joblib', list(X.columns))
                
                print("\nPhase 3 Complete!")
                print("\nHSR Platform execution completed successfully!")
                
            except Exception as e:
                print(f"\nError during Phase 3 execution: {str(e)}")
            
        except Exception as e:
            print(f"\nError during Phase 2 execution: {str(e)}")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")


if __name__ == "__main__":
    main()

