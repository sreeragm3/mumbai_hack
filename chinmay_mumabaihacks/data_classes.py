"""
Data classes for the Hospital Surge Readiness (HSR) Platform.
Defines core data structures for festival events and historical surge records.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class FestivalEvent:
    """
    Represents a festival event with its key information.
    """
    name: str
    date: date
    type: str
    description: Optional[str] = None
    is_high_risk: bool = True
    
    def __str__(self) -> str:
        return f"{self.name} on {self.date}"


@dataclass
class HistoricalSurgeRecord:
    """
    Represents historical hospital surge data for analysis.
    """
    festival_name: str
    festival_date: date
    ed_volume_change_percent: float
    burn_cases: int
    respiratory_cases: int
    trauma_cases: int
    blood_units_used: int
    hospital_name: str
    region: str
    year: int
    notes: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.festival_name} ({self.year}): {self.ed_volume_change_percent:.2f}x surge at {self.hospital_name}"

