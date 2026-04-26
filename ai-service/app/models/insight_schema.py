"""
Pydantic schemas for the Health Insight API.
Strict NULL handling: None = missing data, 0 = real zero value.
"""
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ── Input Schemas ────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    age: int
    gender: int  # 0 = female, 1 = male
    height_cm: float
    weight_kg: float
    language: Optional[str] = "vi"  # "vi" | "en"


class DailyDataPoint(BaseModel):
    date: str  # YYYY-MM-DD
    steps: Optional[float] = None
    distance_meters: Optional[float] = None
    active_calories: Optional[int] = None
    total_calories: Optional[int] = None
    sleep_minutes: Optional[int] = None   # None = no wearable / not recorded
    heart_rate: Optional[int] = None      # None = no wearable
    resting_heart_rate: Optional[int] = None


class HealthInsightRequest(BaseModel):
    user_profile: UserProfile
    daily_data: list[DailyDataPoint] = Field(default_factory=list, min_length=1)
    window_days: int = Field(default=7, ge=1, le=30)


# ── Chat Schemas ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class HealthChatRequest(BaseModel):
    user_profile: UserProfile
    analytics_context: dict[str, Any]  # The analytics block from the insight response
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    message: str


class HealthChatResponse(BaseModel):
    reply: str
    suggested_questions: list[str] = Field(default_factory=list)


# ── Analytics sub-schemas ────────────────────────────────────────────────────

class TrendData(BaseModel):
    steps: str = "INSUFFICIENT_DATA"
    calories: str = "INSUFFICIENT_DATA"
    sleep: str = "INSUFFICIENT_DATA"
    heart_rate: str = "INSUFFICIENT_DATA"


class StatsData(BaseModel):
    steps_avg_7d: Optional[float] = None
    steps_std: Optional[float] = None
    activity_consistency: Optional[float] = None
    sedentary_days: Optional[int] = None
    calories_avg: Optional[float] = None


class AdvancedStatsData(BaseModel):
    heart_rate_avg: Optional[float] = None
    resting_hr_avg: Optional[float] = None
    karvonen_ratio: Optional[float] = None
    sleep_avg_hours: Optional[float] = None
    recovery_score: Optional[float] = None  # 0-100
    hr_zones: Optional[dict[str, float]] = None


class AnalyticsBlock(BaseModel):
    bmi: Optional[float] = None
    bmi_category: Optional[str] = None
    bmr: Optional[float] = None
    tdee: Optional[float] = None
    activity_level: str = "UNKNOWN"
    trends: TrendData = Field(default_factory=TrendData)
    stats: StatsData = Field(default_factory=StatsData)
    advanced: Optional[AdvancedStatsData] = None


# ── Insight sub-schemas ──────────────────────────────────────────────────────

class PredictionBlock(BaseModel):
    horizon_days: int = 7
    expected_activity_level: str
    weight_change_risk: str  # LOW | MEDIUM | HIGH
    confidence: str          # LOW | MEDIUM | HIGH
    notes: Optional[str] = None


class InsightBlock(BaseModel):
    summary: str
    insights: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    prediction: Optional[PredictionBlock] = None
    recommendations: list[str] = Field(default_factory=list)


# ── Full Response ────────────────────────────────────────────────────────────

class HealthInsightResponse(BaseModel):
    mode: Literal["BASIC", "ADVANCED"]
    data_quality: str  # POOR | FAIR | GOOD | EXCELLENT
    analytics: AnalyticsBlock
    insight: Optional[InsightBlock] = None
    error: Optional[str] = None  # Populated if Gemini fails (partial response)
