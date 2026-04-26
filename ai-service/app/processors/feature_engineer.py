"""
Layer 3 — Feature Engineering
Responsibilities:
  - Compute derived features for BASIC and ADVANCED modes
  - Only uses data that is actually available (None-safe)
  - These features feed the Gemini prompt (NOT raw data)
"""
from __future__ import annotations
import statistics
import math
from typing import Optional
from app.processors.data_processor import ProcessedData
from app.models.insight_schema import AdvancedStatsData


def _safe_mean(values: list) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    return round(statistics.mean(filtered), 2) if filtered else None


def _compute_karvonen_ratio(hr_avg: Optional[float], resting_hr: Optional[float], age: int) -> Optional[float]:
    """
    Karvonen exercise intensity ratio.
    Formula: (HR_avg - Resting_HR) / (MaxHR - Resting_HR)
    MaxHR estimated as 220 - age.
    Returns a value in [0, 1], None if data unavailable.
    """
    if hr_avg is None or resting_hr is None:
        return None
    max_hr = 220 - age
    hrr = max_hr - resting_hr
    if hrr <= 0:
        return None
    ratio = (hr_avg - resting_hr) / hrr
    return round(max(0.0, min(1.0, ratio)), 3)


def _compute_recovery_score(resting_hr_avg: Optional[float], hr_avg: Optional[float]) -> Optional[float]:
    """
    Simple recovery proxy:
    Lower resting HR relative to active HR → better recovery.
    Score = (1 - resting_hr / hr_avg) * 100, clamped [0, 100]
    """
    if resting_hr_avg is None or hr_avg is None or hr_avg == 0:
        return None
    raw = (1.0 - resting_hr_avg / hr_avg) * 100
    return round(max(0.0, min(100.0, raw)), 1)


def _compute_hr_zones(hr_avg: Optional[float], age: int) -> Optional[dict]:
    """
    Classify average HR into training zones based on max HR %.
    Returns percentage time estimated in each zone (simplified heuristic).
    """
    if hr_avg is None:
        return None
    max_hr = 220 - age
    pct = hr_avg / max_hr
    if pct < 0.5:
        zone = "RECOVERY"
    elif pct < 0.6:
        zone = "AEROBIC_BASE"
    elif pct < 0.7:
        zone = "AEROBIC"
    elif pct < 0.8:
        zone = "THRESHOLD"
    else:
        zone = "ANAEROBIC"
    return {"primary_zone": zone, "hr_pct_of_max": round(pct * 100, 1)}


def extract_advanced_features(data: ProcessedData, age: int) -> Optional[AdvancedStatsData]:
    """
    Extract ADVANCED features only if wearable data is present.
    Returns None for BASIC mode.
    """
    if data.mode != "ADVANCED":
        return None

    days = data.days

    # Heart Rate
    hr_values = [d.heart_rate for d in days if d.heart_rate is not None]
    resting_hr_values = [d.resting_heart_rate for d in days if d.resting_heart_rate is not None]
    sleep_values = [d.sleep_minutes for d in days if d.sleep_minutes is not None]

    hr_avg = _safe_mean(hr_values)
    resting_hr_avg = _safe_mean(resting_hr_values)
    sleep_avg_hours = round(_safe_mean(sleep_values) / 60, 2) if _safe_mean(sleep_values) is not None else None

    karvonen_ratio = _compute_karvonen_ratio(hr_avg, resting_hr_avg, age)
    recovery_score = _compute_recovery_score(resting_hr_avg, hr_avg)
    hr_zones = _compute_hr_zones(hr_avg, age)

    return AdvancedStatsData(
        heart_rate_avg=hr_avg,
        resting_hr_avg=resting_hr_avg,
        karvonen_ratio=karvonen_ratio,
        sleep_avg_hours=sleep_avg_hours,
        recovery_score=recovery_score,
        hr_zones=hr_zones,
    )
