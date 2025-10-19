"""
Configuration constants for the Hospital Surge Readiness (HSR) Platform.
Contains API settings, country codes, and alert configurations.
"""

# Calendarific API Configuration
import os
CALENDARIFIC_API_KEY = os.getenv("CALENDARIFIC_API_KEY", "YOUR_API_KEY_HERE")
CALENDARIFIC_BASE_URL = "https://calendarific.com/api/v2"
COUNTRY_CODE = "IN"  # India
TARGET_YEAR = 2026

# High-risk festivals that typically cause hospital surges
HIGH_RISK_FESTIVALS = [
    "Diwali",
    "Holi", 
    "Dussehra",
    "Navratri",
]

# Alert lead times in days before festival
ALERT_LEAD_TIMES = [30, 7]  # 30 days and 7 days before festival

