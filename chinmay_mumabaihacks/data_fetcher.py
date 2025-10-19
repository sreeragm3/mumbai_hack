"""
Data fetching and processing module for the Hospital Surge Readiness (HSR) Platform.
Handles Calendarific API integration and data processing.
"""

import requests
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any
import os
import random
import csv

from config import (
    CALENDARIFIC_API_KEY, 
    CALENDARIFIC_BASE_URL, 
    COUNTRY_CODE, 
    TARGET_YEAR,
    HIGH_RISK_FESTIVALS,
    ALERT_LEAD_TIMES
)
from data_classes import HistoricalSurgeRecord


def fetch_festival_dates_calendarific(year: int) -> pd.DataFrame:
    """
    Fetch festival dates from Calendarific API for the specified year.
    
    Args:
        year (int): The year to fetch festivals for
        
    Returns:
        pd.DataFrame: DataFrame containing filtered festival data
        
    Raises:
        requests.RequestException: If API request fails
        ValueError: If API key is not configured
    """
    if CALENDARIFIC_API_KEY == "YOUR_API_KEY_HERE":
        raise ValueError("Please configure your Calendarific API key in config.py")
    
    # Construct API URL
    url = f"{CALENDARIFIC_BASE_URL}/holidays"
    params = {
        'api_key': CALENDARIFIC_API_KEY,
        'country': COUNTRY_CODE,
        'year': year,
        'type': 'national'
    }
    
    try:
        # Make API request with timeout
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('meta', {}).get('code') != 200:
            raise requests.RequestException(f"API Error: {data.get('meta', {}).get('error_detail', 'Unknown error')}")
        
        holidays = data.get('response', {}).get('holidays', [])
        
        # Filter for high-risk festivals
        filtered_holidays = []
        for holiday in holidays:
            holiday_name = holiday.get('name', '').lower()
            
            # Check if this holiday matches any high-risk festival
            for risk_festival in HIGH_RISK_FESTIVALS:
                if risk_festival.lower() in holiday_name or holiday_name in risk_festival.lower():
                    filtered_holidays.append({
                        'name': holiday.get('name', ''),
                        'date': holiday.get('date', {}).get('iso', ''),
                        'type': holiday.get('type', ''),
                        'description': holiday.get('description', ''),
                        'is_high_risk': True
                    })
                    break
        
        
        # Convert to DataFrame
        if filtered_holidays:
            df = pd.DataFrame(filtered_holidays)
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df
        else:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['name', 'date', 'type', 'description', 'is_high_risk'])
            
    except requests.exceptions.Timeout:
        raise requests.RequestException("API request timed out. Please try again later.")
    except requests.exceptions.ConnectionError:
        raise requests.RequestException("Failed to connect to Calendarific API. Please check your internet connection.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise requests.RequestException("API endpoint not found. Please check the API URL.")
        elif e.response.status_code == 401:
            raise requests.RequestException("Invalid API key. Please check your Calendarific API key.")
        else:
            raise requests.RequestException(f"HTTP Error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise requests.RequestException(f"Unexpected error: {str(e)}")


def save_festival_calendar(df: pd.DataFrame) -> None:
    """
    Save festival calendar DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): Festival data to save
    """
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    file_path = 'data/festival_calendar.csv'
    df.to_csv(file_path, index=False)


def get_events_for_alerting(calendar_df: pd.DataFrame, today: date) -> List[Dict[str, Any]]:
    """
    Check which festivals are due for alerting based on lead times.
    
    Args:
        calendar_df (pd.DataFrame): Festival calendar data
        today (date): Current date to check against
        
    Returns:
        List[Dict[str, Any]]: List of festivals that need alerts
    """
    if calendar_df.empty:
        return []
    
    alerting_events = []
    
    for _, row in calendar_df.iterrows():
        festival_date = row['date']
        days_until = (festival_date - today).days
        
        # Check if this festival is within any alert lead time
        for lead_time in ALERT_LEAD_TIMES:
            if days_until == lead_time:
                alerting_events.append({
                    'festival_name': row['name'],
                    'festival_date': festival_date,
                    'days_until': days_until,
                    'alert_type': f"{lead_time}-day alert",
                    'description': row.get('description', ''),
                    'type': row.get('type', '')
                })
                break
    
    return alerting_events


def generate_historical_data(num_years: int = 5, file_path: str = 'data/historical_records.csv') -> None:
    """
    Generate synthetic historical surge data for Diwali and Holi festivals.
    
    This function creates realistic surge patterns based on real-world data insights:
    - Diwali: High burn cases (firecrackers/lamps) and respiratory cases (pollution)
    - Holi: High trauma cases (falls/accidents) and respiratory cases (chemical colors)
    
    Args:
        num_years (int): Number of years of historical data to generate (default: 5)
        file_path (str): Path to save the generated CSV file
    """
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Sample hospital and region data
    hospitals = [
        "Apollo Hospital", "Fortis Healthcare", "Max Hospital", "AIIMS Delhi",
        "KEM Hospital", "Seth GS Medical College", "Tata Memorial Hospital",
        "Narayana Health", "Manipal Hospital", "Columbia Asia"
    ]
    
    regions = [
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
        "Pune", "Ahmedabad", "Jaipur", "Kochi"
    ]
    
    # Generate data for each year
    current_year = datetime.now().year
    start_year = current_year - num_years
    
    records = []
    
    for year in range(start_year, current_year):
        
        # Generate Diwali data (typically in October/November)
        diwali_date = date(year, 10, 15) + pd.Timedelta(days=random.randint(-30, 30))
        
        # Diwali surge pattern: High burns, high respiratory, moderate trauma
        for hospital in random.sample(hospitals, 3):  # 3 hospitals per festival per year
            region = random.choice(regions)
            
            # Diwali characteristics
            ed_volume_change = random.uniform(1.60, 2.00)  # 60-100% surge
            burn_cases = random.randint(120, 180)  # High burn cases
            respiratory_cases = random.randint(80, 130)  # High respiratory due to pollution
            trauma_cases = random.randint(40, 80)  # Moderate trauma
            blood_units_used = random.randint(30, 50)  # High blood usage
            
            record = HistoricalSurgeRecord(
                festival_name="Diwali",
                festival_date=diwali_date,
                ed_volume_change_percent=ed_volume_change,
                burn_cases=burn_cases,
                respiratory_cases=respiratory_cases,
                trauma_cases=trauma_cases,
                blood_units_used=blood_units_used,
                hospital_name=hospital,
                region=region,
                year=year,
                notes=f"Firecracker injuries and pollution-related respiratory issues"
            )
            records.append(record)
        
        # Generate Holi data (typically in March)
        holi_date = date(year, 3, 1) + pd.Timedelta(days=random.randint(-15, 15))
        
        # Holi surge pattern: High trauma, high respiratory, low burns
        for hospital in random.sample(hospitals, 3):  # 3 hospitals per festival per year
            region = random.choice(regions)
            
            # Holi characteristics
            ed_volume_change = random.uniform(1.45, 1.75)  # 45-75% surge
            trauma_cases = random.randint(100, 160)  # High trauma cases
            respiratory_cases = random.randint(60, 110)  # High respiratory due to chemical colors
            burn_cases = random.randint(5, 15)  # Low burn cases
            blood_units_used = random.randint(20, 40)  # Moderate blood usage
            
            record = HistoricalSurgeRecord(
                festival_name="Holi",
                festival_date=holi_date,
                ed_volume_change_percent=ed_volume_change,
                burn_cases=burn_cases,
                respiratory_cases=respiratory_cases,
                trauma_cases=trauma_cases,
                blood_units_used=blood_units_used,
                hospital_name=hospital,
                region=region,
                year=year,
                notes=f"Trauma from falls/accidents and chemical color-related respiratory issues"
            )
            records.append(record)
    
    # Save records to CSV
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'festival_name', 'festival_date', 'ed_volume_change_percent',
            'burn_cases', 'respiratory_cases', 'trauma_cases', 'blood_units_used',
            'hospital_name', 'region', 'year', 'notes'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in records:
            writer.writerow({
                'festival_name': record.festival_name,
                'festival_date': record.festival_date.isoformat(),
                'ed_volume_change_percent': record.ed_volume_change_percent,
                'burn_cases': record.burn_cases,
                'respiratory_cases': record.respiratory_cases,
                'trauma_cases': record.trauma_cases,
                'blood_units_used': record.blood_units_used,
                'hospital_name': record.hospital_name,
                'region': record.region,
                'year': record.year,
                'notes': record.notes
            })
    
    print(f"Generated {len(records)} historical records for {num_years} years")


def get_forecast_inputs(festival_name: str) -> dict:
    """
    Simulates forecasting the independent variables needed for the ML model.
    In production, this would be a separate, complex forecasting model.
    
    Args:
        festival_name (str): Name of the festival to forecast
        
    Returns:
        dict: Dictionary containing forecasted inputs for the ML model
    """
    # Use a simple, hardcoded look-up table based on average synthetic data
    forecast_data = {
        'Diwali': {'ed_surge_percent': 1.85, 'trauma_cases_forecast': 65},
        'Diwali/Deepavali': {'ed_surge_percent': 1.85, 'trauma_cases_forecast': 65},
        'Holi': {'ed_surge_percent': 1.70, 'trauma_cases_forecast': 130},
        'Dussehra': {'ed_surge_percent': 1.50, 'trauma_cases_forecast': 80}  # Placeholder
    }
    
    return forecast_data.get(festival_name, {'ed_surge_percent': 1.0, 'trauma_cases_forecast': 0})


if __name__ == "__main__":
    # Main execution block for data generation
    print("=" * 60)
    print("HSR Platform - Historical Data Generation")
    print("=" * 60)
    
    try:
        generate_historical_data(num_years=5)
        print("\n✓ Data generation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during data generation: {str(e)}")

