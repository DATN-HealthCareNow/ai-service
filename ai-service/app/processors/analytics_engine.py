"""
Layer 2 — Analytics Engine (PURE Rule-Based, NO ML)
Responsibilities:
  - BMI, BMR, TDEE calculations
  - Activity level classification
  - Trend detection (via simple linear regression)
  - Consistency and sedentary scoring
"""
from __future__ import annotations
import statistics
from typing import Optional
from app.processors.data_processor import ProcessedData, ProcessedDay
from app.models.insight_schema import AnalyticsBlock, TrendData, StatsData


def _safe_mean(values: list) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    return round(statistics.mean(filtered), 2) if filtered else None


def _safe_std(values: list) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    return round(statistics.pstdev(filtered), 2) if len(filtered) >= 2 else None


def _compute_trend(values: list[Optional[float]], min_points: int = 3) -> str:
    """
    Simple linear regression slope to classify trend.
    Returns: INCREASING | DECREASING | STABLE | INSUFFICIENT_DATA
    """
    filtered = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(filtered) < min_points:
        return "INSUFFICIENT_DATA"

    n = len(filtered)
    x_vals = [p[0] for p in filtered]
    y_vals = [p[1] for p in filtered]

    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)

    if denominator == 0:
        return "STABLE"

    slope = numerator / denominator
    rel_change = slope / y_mean if y_mean and y_mean != 0 else 0

    if rel_change > 0.05:
        return "INCREASING"
    elif rel_change < -0.05:
        return "DECREASING"
    else:
        return "STABLE"


def _calculate_bmi(weight_kg: float, height_cm: float) -> tuple[float, str]:
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)
    if bmi < 18.5:
        category = "UNDERWEIGHT"
    elif bmi < 25.0:
        category = "NORMAL"
    elif bmi < 30.0:
        category = "OVERWEIGHT"
    else:
        category = "OBESE"
    return bmi, category


def _calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: int) -> float:
    """Mifflin-St Jeor formula"""
    s = 5 if gender == 1 else -161
    return round((10 * weight_kg) + (6.25 * height_cm) - (5 * age) + s, 1)


def _classify_activity_level(steps_avg: Optional[float]) -> str:
    if steps_avg is None:
        return "UNKNOWN"
    if steps_avg < 3000:
        return "SEDENTARY"
    elif steps_avg < 7500:
        return "LIGHTLY_ACTIVE"
    elif steps_avg < 10000:
        return "MODERATELY_ACTIVE"
    else:
        return "VERY_ACTIVE"


def _get_pal(activity_level: str) -> float:
    return {
        "SEDENTARY": 1.2,
        "LIGHTLY_ACTIVE": 1.375,
        "MODERATELY_ACTIVE": 1.55,
        "VERY_ACTIVE": 1.725,
        "UNKNOWN": 1.2,
    }.get(activity_level, 1.2)


def run_analytics(
    data: ProcessedData,
    age: int,
    gender: int,
    height_cm: float,
    weight_kg: float,
) -> AnalyticsBlock:
    """
    Main analytics runner — purely rule-based.
    Computes all metrics from ProcessedData.
    """
    days = data.days

    # ── Biometric ────────────────────────────────────────────────────────────
    bmi, bmi_category = _calculate_bmi(weight_kg, height_cm)
    bmr = _calculate_bmr(weight_kg, height_cm, age, gender)

    # ── Step metrics ─────────────────────────────────────────────────────────
    steps_series = [d.steps for d in days]
    steps_avg = _safe_mean(steps_series)
    steps_std = _safe_std(steps_series)

    activity_level = _classify_activity_level(steps_avg)
    pal = _get_pal(activity_level)
    tdee = round(bmr * pal, 1)

    # ── Consistency & sedentary ───────────────────────────────────────────────
    active_days = sum(1 for d in days if d.steps is not None and d.steps > 500)
    total = data.total_days
    activity_consistency = round(active_days / total, 3) if total > 0 else None

    sedentary_days = sum(
        1 for d in days if d.steps is not None and d.steps < 5000
    )

    # ── Trends ────────────────────────────────────────────────────────────────
    calories_series = [d.active_calories for d in days]
    sleep_series = [d.sleep_minutes for d in days]
    hr_series = [d.heart_rate for d in days]

    trends = TrendData(
        steps=_compute_trend(steps_series),
        calories=_compute_trend(calories_series),
        sleep=_compute_trend(sleep_series),
        heart_rate=_compute_trend(hr_series),
    )

    stats = StatsData(
        steps_avg_7d=steps_avg,
        steps_std=steps_std,
        activity_consistency=activity_consistency,
        sedentary_days=sedentary_days,
        calories_avg=_safe_mean(calories_series),
    )

    return AnalyticsBlock(
        bmi=bmi,
        bmi_category=bmi_category,
        bmr=bmr,
        tdee=tdee,
        activity_level=activity_level,
        trends=trends,
        stats=stats,
    )
