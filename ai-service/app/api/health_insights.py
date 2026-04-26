"""
Health Insight API — Main entry points
POST /ai/v1/health-insights   → Full analysis pipeline
POST /ai/v1/health-chat       → Contextual AI chat
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from app.models.insight_schema import (
    HealthInsightRequest, HealthInsightResponse,
    HealthChatRequest, HealthChatResponse,
    AnalyticsBlock,
)
from app.processors.data_processor import process_daily_data
from app.processors.analytics_engine import run_analytics
from app.processors.feature_engineer import extract_advanced_features
from app.processors.ml_predictor import run_ml_analysis
from app.services.insight_service import generate_health_insight, generate_health_chat_reply

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ai/v1/health-insights", response_model=HealthInsightResponse)
async def analyze_health_insights(request: HealthInsightRequest) -> HealthInsightResponse:
    """
    Full AI health analysis pipeline:
    1. Process & clean raw daily data (NULL handling)
    2. Run rule-based analytics (BMI, BMR, TDEE, trends)
    3. Extract engineered features (ADVANCED mode only)
    4. Run ML/heuristic analysis (fatigue, risks, prediction)
    5. Generate Gemini AI insight (summary, recommendations)
    """
    profile = request.user_profile
    logger.info(
        f"[health-insights] Starting analysis: mode=TBD, days={len(request.daily_data)}, "
        f"language={profile.language}"
    )

    try:
        # Step 1 — Data Processing
        processed = process_daily_data(request.daily_data)
        logger.info(f"[health-insights] Processed: mode={processed.mode}, quality={processed.data_quality}")

        # Fail fast for completely empty data
        if processed.data_quality == "POOR" and processed.days_with_steps == 0:
            return HealthInsightResponse(
                mode=processed.mode,
                data_quality="POOR",
                analytics=AnalyticsBlock(activity_level="UNKNOWN"),
                insight=None,
                error="Insufficient data: no step data available for analysis. Please sync your health data.",
            )

        # Step 2 — Rule-Based Analytics
        analytics = run_analytics(
            data=processed,
            age=profile.age,
            gender=profile.gender,
            height_cm=profile.height_cm,
            weight_kg=profile.weight_kg,
        )

        # Step 3 — Feature Engineering (ADVANCED only)
        advanced_features = extract_advanced_features(processed, age=profile.age)
        if advanced_features:
            analytics.advanced = advanced_features

        # Step 4 — ML / Heuristic Analysis
        ml_result = run_ml_analysis(analytics, advanced_features)

        # Step 5 — Gemini AI Insight (may fail gracefully)
        insight = generate_health_insight(
            analytics=analytics,
            advanced=advanced_features,
            ml_result=ml_result,
            mode=processed.mode,
            language=profile.language or "vi",
        )

        error_msg = None
        if insight is None:
            error_msg = "AI insight generation failed. Analytics data is still available."
            logger.warning("[health-insights] Gemini failed — returning partial response")

        return HealthInsightResponse(
            mode=processed.mode,
            data_quality=processed.data_quality,
            analytics=analytics,
            insight=insight,
            error=error_msg,
        )

    except Exception as e:
        logger.error(f"[health-insights] Unexpected error in pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Health insights pipeline error: {str(e)}"
        )


@router.post("/ai/v1/health-chat", response_model=HealthChatResponse)
async def health_chat(request: HealthChatRequest) -> HealthChatResponse:
    """
    Contextual health AI chat.
    Uses the analytics_context from a previous /health-insights call
    to answer user questions with full data awareness.
    """
    language = request.user_profile.language or "vi"
    logger.info(f"[health-chat] Chat request received, language={language}")

    # Build user profile dict for context
    profile_dict = {
        "age": request.user_profile.age,
        "gender": "male" if request.user_profile.gender == 1 else "female",
        "height_cm": request.user_profile.height_cm,
        "weight_kg": request.user_profile.weight_kg,
    }

    # Build conversation history in expected format
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in request.conversation_history
    ]

    return generate_health_chat_reply(
        user_profile=profile_dict,
        analytics_context=request.analytics_context,
        conversation_history=history,
        user_message=request.message,
        language=language,
    )
