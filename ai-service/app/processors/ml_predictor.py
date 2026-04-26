"""
Layer 4 — Lightweight ML Predictor (ADVANCED mode only)
Responsibilities:
  - Fatigue level classification (rule-based heuristic, upgradeable to ML)
  - Risk flags based on computed features
  - Confidence scoring based on data quality
  
Note: The existing XGB/RF models in rf_pipeline.pkl / xgb_pipeline.pkl
predict heart_rate and calories from wearable signals.
We reuse xgb_model for fitness insights where applicable.
"""
from __future__ import annotations
from typing import Optional
from app.models.insight_schema import AdvancedStatsData, StatsData, AnalyticsBlock


def _classify_fatigue(
    advanced: Optional[AdvancedStatsData],
    stats: StatsData,
) -> str:
    """
    Fatigue heuristic (rule-based, designed to be replaced with LogisticRegression).
    Factors:
      - Low recovery score → fatigued
      - Low sleep avg → fatigued
      - High karvonen ratio (training too hard) → risk
      - Low consistency → might be forced rest
    Returns: LOW_FATIGUE | MODERATE_FATIGUE | HIGH_FATIGUE | UNKNOWN
    """
    if advanced is None:
        return "UNKNOWN"

    score = 0  # 0 = fresh, higher = more fatigued

    if advanced.recovery_score is not None:
        if advanced.recovery_score < 30:
            score += 2
        elif advanced.recovery_score < 50:
            score += 1

    if advanced.sleep_avg_hours is not None:
        if advanced.sleep_avg_hours < 5:
            score += 2
        elif advanced.sleep_avg_hours < 6.5:
            score += 1

    if advanced.karvonen_ratio is not None:
        if advanced.karvonen_ratio > 0.75:
            score += 1  # Training at very high intensity

    if stats.activity_consistency is not None and stats.activity_consistency < 0.4:
        score += 1  # Inconsistent activity, possible fatigue-driven rest

    if score == 0:
        return "LOW_FATIGUE"
    elif score <= 2:
        return "MODERATE_FATIGUE"
    else:
        return "HIGH_FATIGUE"


def _detect_risks(
    analytics: AnalyticsBlock,
    advanced: Optional[AdvancedStatsData],
) -> list[str]:
    """
    Rule-based risk detection. Each risk is a short key (not displayed directly).
    These get passed to Gemini for human-readable explanation.
    """
    risks = []

    # ── Sedentary risk ────────────────────────────────────────────────────────
    if analytics.stats.sedentary_days is not None:
        if analytics.stats.sedentary_days >= 5:
            risks.append("CHRONIC_SEDENTARY")
        elif analytics.stats.sedentary_days >= 3:
            risks.append("MODERATE_SEDENTARY")

    # ── BMI risk ──────────────────────────────────────────────────────────────
    if analytics.bmi_category == "OBESE":
        risks.append("OBESITY_RISK")
    elif analytics.bmi_category == "OVERWEIGHT":
        risks.append("OVERWEIGHT_RISK")
    elif analytics.bmi_category == "UNDERWEIGHT":
        risks.append("UNDERWEIGHT_RISK")

    # ── Declining trend ───────────────────────────────────────────────────────
    if analytics.trends.steps == "DECREASING":
        risks.append("DECLINING_ACTIVITY")

    # ── Sleep risk (ADVANCED only) ────────────────────────────────────────────
    if advanced and advanced.sleep_avg_hours is not None:
        if advanced.sleep_avg_hours < 6:
            risks.append("SLEEP_DEPRIVATION")
        elif advanced.sleep_avg_hours < 7:
            risks.append("INSUFFICIENT_SLEEP")

    # ── HR anomaly (ADVANCED only) ────────────────────────────────────────────
    if advanced and advanced.resting_hr_avg is not None:
        if advanced.resting_hr_avg > 100:
            risks.append("ELEVATED_RESTING_HR")
        elif advanced.resting_hr_avg > 85:
            risks.append("HIGH_RESTING_HR")

    return risks


def _predict_short_term(
    analytics: AnalyticsBlock,
    fatigue_level: str,
    risks: list[str],
) -> dict:
    """
    Simple rule-based short-term prediction.
    Returns a structured prediction block for Gemini context.
    """
    trend = analytics.trends.steps
    consistency = analytics.stats.activity_consistency or 0

    # Predict expected activity level
    if trend == "INCREASING" and consistency > 0.7:
        expected = "VERY_ACTIVE"
        confidence = "HIGH"
    elif trend == "STABLE" and consistency > 0.5:
        expected = analytics.activity_level
        confidence = "MEDIUM"
    elif trend == "DECREASING" or fatigue_level == "HIGH_FATIGUE":
        expected = "LIGHTLY_ACTIVE"
        confidence = "MEDIUM"
    else:
        expected = analytics.activity_level
        confidence = "LOW"

    # Weight change risk
    weight_risk = "LOW"
    if "CHRONIC_SEDENTARY" in risks and analytics.bmi_category in ("OVERWEIGHT", "OBESE"):
        weight_risk = "HIGH"
    elif "MODERATE_SEDENTARY" in risks or "DECLINING_ACTIVITY" in risks:
        weight_risk = "MEDIUM"

    return {
        "horizon_days": 7,
        "expected_activity_level": expected,
        "weight_change_risk": weight_risk,
        "confidence": confidence,
        "fatigue_level": fatigue_level,
        "detected_risks": risks,
    }


def run_ml_analysis(
    analytics: AnalyticsBlock,
    advanced: Optional[AdvancedStatsData],
) -> dict:
    """
    Main ML analysis runner.
    Returns a structured dict for the Gemini insight prompt.
    """
    fatigue = _classify_fatigue(advanced, analytics.stats)
    risks = _detect_risks(analytics, advanced)
    prediction = _predict_short_term(analytics, fatigue, risks)

    return {
        "fatigue_level": fatigue,
        "detected_risks": risks,
        "prediction": prediction,
    }
