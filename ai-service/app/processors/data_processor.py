"""
Layer 1 — Data Processing
Responsibilities:
  - Detect BASIC vs ADVANCED mode
  - Strictly handle NULL values (None != 0)
  - Assess data quality
  - Build cleaned, typed daily records
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from app.models.insight_schema import DailyDataPoint


@dataclass
class ProcessedDay:
    date: str
    steps: Optional[float]           # None = missing
    distance_meters: Optional[float]
    active_calories: Optional[int]
    total_calories: Optional[int]
    sleep_minutes: Optional[int]     # None = no wearable / not logged
    heart_rate: Optional[int]        # None = no wearable
    resting_heart_rate: Optional[int]


@dataclass
class ProcessedData:
    days: list[ProcessedDay] = field(default_factory=list)
    mode: str = "BASIC"              # "BASIC" | "ADVANCED"
    data_quality: str = "POOR"       # "POOR" | "FAIR" | "GOOD" | "EXCELLENT"
    days_with_steps: int = 0
    days_with_hr: int = 0
    days_with_sleep: int = 0
    total_days: int = 0


def _to_none_if_zero_or_missing(value, is_wearable_field: bool = False):
    """
    Core rule:
      - If value is already None → missing, keep as None
      - If value is 0 AND it's a wearable field (hr, sleep) → treat as missing (None)
        because 0 bpm or 0 minutes sleep is physiologically impossible
      - If value is 0 AND it's an activity field (steps, calories) → keep as 0 (valid)
    """
    if value is None:
        return None
    if is_wearable_field and value == 0:
        return None
    return value


def process_daily_data(raw_data: list[DailyDataPoint]) -> ProcessedData:
    """
    Transform raw daily data points into clean, mode-aware ProcessedData.
    This is the SINGLE source of truth for NULL handling.
    """
    days: list[ProcessedDay] = []

    for point in raw_data:
        day = ProcessedDay(
            date=point.date,
            steps=_to_none_if_zero_or_missing(point.steps, is_wearable_field=False),
            distance_meters=_to_none_if_zero_or_missing(point.distance_meters, is_wearable_field=False),
            active_calories=_to_none_if_zero_or_missing(point.active_calories, is_wearable_field=False),
            total_calories=_to_none_if_zero_or_missing(point.total_calories, is_wearable_field=False),
            sleep_minutes=_to_none_if_zero_or_missing(point.sleep_minutes, is_wearable_field=True),
            heart_rate=_to_none_if_zero_or_missing(point.heart_rate, is_wearable_field=True),
            resting_heart_rate=_to_none_if_zero_or_missing(point.resting_heart_rate, is_wearable_field=True),
        )
        days.append(day)

    days_with_steps = sum(1 for d in days if d.steps is not None and d.steps > 0)
    days_with_hr = sum(1 for d in days if d.heart_rate is not None)
    days_with_sleep = sum(1 for d in days if d.sleep_minutes is not None)
    total_days = len(days)

    # Determine mode
    mode = "ADVANCED" if (days_with_hr > 0 or days_with_sleep > 0) else "BASIC"

    # Assess data quality
    if total_days == 0:
        quality = "POOR"
    elif days_with_steps == 0:
        quality = "POOR"
    elif days_with_steps / total_days < 0.4:
        quality = "FAIR"
    elif days_with_steps / total_days < 0.7:
        quality = "GOOD"
    else:
        quality = "EXCELLENT"

    return ProcessedData(
        days=days,
        mode=mode,
        data_quality=quality,
        days_with_steps=days_with_steps,
        days_with_hr=days_with_hr,
        days_with_sleep=days_with_sleep,
        total_days=total_days,
    )
