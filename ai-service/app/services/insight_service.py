"""
Layer 5 — Gemini AI Insight & Chat Orchestration
Responsibilities:
  - Build structured prompts from FEATURES (never raw data)
  - Call Gemini API with fallback
  - Parse and validate structured JSON response
  - Handle chat conversations with full analytics context
"""
from __future__ import annotations
import json
import logging
from typing import Any, Optional
from app.models.insight_schema import (
    AnalyticsBlock, AdvancedStatsData, InsightBlock,
    PredictionBlock, HealthChatResponse,
)
from app.services.gemini_service import _generate_with_model_fallback, ANALYSIS_MODELS, ARTICLE_MODELS

logger = logging.getLogger(__name__)

# ── Prompt Builders ───────────────────────────────────────────────────────────

def _build_basic_context(analytics: AnalyticsBlock) -> str:
    s = analytics.stats
    t = analytics.trends
    return f"""
BIOMETRIC:
  - BMI: {analytics.bmi} ({analytics.bmi_category})
  - BMR: {analytics.bmr} kcal/day
  - TDEE (estimated): {analytics.tdee} kcal/day

ACTIVITY (7-day window):
  - Activity level: {analytics.activity_level}
  - Average steps/day: {s.steps_avg_7d or 'N/A'}
  - Steps variability (std): {s.steps_std or 'N/A'}
  - Active days ratio: {f"{s.activity_consistency:.0%}" if s.activity_consistency is not None else 'N/A'}
  - Sedentary days (< 5,000 steps): {s.sedentary_days or 'N/A'}
  - Avg active calories/day: {s.calories_avg or 'N/A'}

TRENDS (7-day):
  - Steps trend: {t.steps}
  - Calories trend: {t.calories}
""".strip()


def _build_advanced_context(advanced: AdvancedStatsData) -> str:
    hr_zone = advanced.hr_zones.get("primary_zone", "N/A") if advanced.hr_zones else "N/A"
    hr_pct = advanced.hr_zones.get("hr_pct_of_max", "N/A") if advanced.hr_zones else "N/A"
    return f"""
WEARABLE DATA:
  - Avg active heart rate: {advanced.heart_rate_avg or 'N/A'} bpm
  - Avg resting heart rate: {advanced.resting_hr_avg or 'N/A'} bpm
  - Karvonen intensity ratio: {advanced.karvonen_ratio or 'N/A'} (0 = rest, 1 = max effort)
  - Primary HR zone: {hr_zone} ({hr_pct}% of max HR)
  - Recovery score: {advanced.recovery_score or 'N/A'}/100
  - Avg sleep: {advanced.sleep_avg_hours or 'N/A'} hours/night

TRENDS:
  - Sleep trend: {'{trends_sleep}'}
  - Heart rate trend: {'{trends_hr}'}
""".strip()


def _build_risk_context(ml_result: dict) -> str:
    risks = ml_result.get("detected_risks", [])
    fatigue = ml_result.get("fatigue_level", "UNKNOWN")
    pred = ml_result.get("prediction", {})

    risks_str = ", ".join(risks) if risks else "No significant risks detected"

    return f"""
ANALYSIS RESULTS:
  - Detected risks: {risks_str}
  - Fatigue level: {fatigue}
  - Predicted activity next 7 days: {pred.get('expected_activity_level', 'N/A')}
  - Weight change risk: {pred.get('weight_change_risk', 'N/A')}
  - Prediction confidence: {pred.get('confidence', 'N/A')}
""".strip()


def _build_insight_prompt(
    analytics: AnalyticsBlock,
    advanced: Optional[AdvancedStatsData],
    ml_result: dict,
    mode: str,
    language: str,
) -> str:
    lang_instruction = (
        "Respond ENTIRELY in Vietnamese. Use friendly, motivational tone."
        if language == "vi"
        else "Respond ENTIRELY in English. Use friendly, motivational tone."
    )

    basic_ctx = _build_basic_context(analytics)

    advanced_ctx = ""
    if mode == "ADVANCED" and advanced:
        raw_adv = _build_advanced_context(advanced)
        advanced_ctx = raw_adv.replace(
            "{trends_sleep}", analytics.trends.sleep
        ).replace(
            "{trends_hr}", analytics.trends.heart_rate
        )

    risk_ctx = _build_risk_context(ml_result)

    return f"""You are a certified health coach AI embedded in a mobile health app.
Your job is to analyze the user's health metrics and provide CLEAR, ACTIONABLE, EVIDENCE-BASED insights.

LANGUAGE: {lang_instruction}

--- USER HEALTH DATA ---

{basic_ctx}

{advanced_ctx}

{risk_ctx}

--- TASK ---
Return a JSON object with EXACTLY these keys:
{{
  "summary": "2-3 sentence overview of the user's health status this week",
  "insights": ["insight 1", "insight 2", "insight 3"],
  "risks": ["risk description 1"],
  "prediction": {{
    "horizon_days": 7,
    "expected_activity_level": "<value from analysis>",
    "weight_change_risk": "<LOW|MEDIUM|HIGH>",
    "confidence": "<LOW|MEDIUM|HIGH>",
    "notes": "1 sentence explaining the prediction"
  }},
  "recommendations": ["specific action 1", "specific action 2", "specific action 3"]
}}

STRICT RULES:
1. Base EVERY statement on the provided data. Do NOT invent facts.
2. Do NOT mention any metric that has "N/A" or is missing.
3. insights: 2-4 items, each starting with a specific observation.
4. risks: 0-3 items. Leave empty array [] if no real risks exist.
5. recommendations: 2-4 SPECIFIC, immediately actionable steps.
6. Return ONLY the JSON object, no markdown, no extra text.
"""


def _build_chat_prompt(
    user_profile: dict,
    analytics_context: dict,
    conversation_history: list[dict],
    user_message: str,
    language: str,
) -> str:
    lang_instruction = (
        "Toàn bộ câu trả lời bằng tiếng Việt."
        if language == "vi"
        else "Reply entirely in English."
    )

    history_text = ""
    for msg in conversation_history[-6:]:  # Keep last 3 exchanges
        role = "User" if msg["role"] == "user" else "Health Coach"
        history_text += f"{role}: {msg['content']}\n"

    ctx_str = json.dumps(analytics_context, ensure_ascii=False, indent=2)

    return f"""You are a personal AI health coach. You have access to the user's weekly health analytics.
{lang_instruction}

--- USER HEALTH ANALYTICS CONTEXT ---
{ctx_str}

--- CONVERSATION HISTORY ---
{history_text if history_text else "(new conversation)"}

--- USER MESSAGE ---
User: {user_message}

--- INSTRUCTIONS ---
1. Answer based STRICTLY on the user's analytics context above.
2. Be conversational, empathetic, and encouraging.
3. If asked about something not in the data (e.g., specific medical diagnosis), 
   politely clarify you can only discuss the measured metrics.
4. Keep replies concise (2-4 sentences unless a detailed explanation is needed).
5. After your reply, suggest 2-3 follow-up questions the user might ask.

Return a JSON object:
{{
  "reply": "your response here",
  "suggested_questions": ["question 1", "question 2", "question 3"]
}}

Return ONLY the JSON, no markdown.
"""


# ── Response Parsers ──────────────────────────────────────────────────────────

def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_insight_response(raw_text: str, ml_result: dict) -> InsightBlock:
    """Parse Gemini insight response into InsightBlock."""
    cleaned = _clean_json_text(raw_text)
    data = json.loads(cleaned)

    pred_data = data.get("prediction", {})
    prediction = PredictionBlock(
        horizon_days=pred_data.get("horizon_days", 7),
        expected_activity_level=pred_data.get(
            "expected_activity_level",
            ml_result.get("prediction", {}).get("expected_activity_level", "UNKNOWN")
        ),
        weight_change_risk=pred_data.get(
            "weight_change_risk",
            ml_result.get("prediction", {}).get("weight_change_risk", "LOW")
        ),
        confidence=pred_data.get(
            "confidence",
            ml_result.get("prediction", {}).get("confidence", "LOW")
        ),
        notes=pred_data.get("notes"),
    )

    return InsightBlock(
        summary=data.get("summary", ""),
        insights=data.get("insights", []),
        risks=data.get("risks", []),
        prediction=prediction,
        recommendations=data.get("recommendations", []),
    )


def _parse_chat_response(raw_text: str) -> HealthChatResponse:
    """Parse Gemini chat response into HealthChatResponse."""
    cleaned = _clean_json_text(raw_text)
    data = json.loads(cleaned)
    return HealthChatResponse(
        reply=data.get("reply", ""),
        suggested_questions=data.get("suggested_questions", []),
    )


# ── Main Public Functions ─────────────────────────────────────────────────────

def generate_health_insight(
    analytics: AnalyticsBlock,
    advanced: Optional[AdvancedStatsData],
    ml_result: dict,
    mode: str,
    language: str = "vi",
) -> Optional[InsightBlock]:
    """
    Calls Gemini to generate structured health insights.
    Returns None on failure (caller should return partial response).
    """
    try:
        prompt = _build_insight_prompt(analytics, advanced, ml_result, mode, language)
        response = _generate_with_model_fallback(
            model_candidates=ANALYSIS_MODELS,
            contents=prompt,
            temperature=0.3,
        )
        return _parse_insight_response(response.text, ml_result)
    except Exception as e:
        logger.error(f"[insight_service] Gemini insight generation failed: {e}")
        return None


def generate_health_chat_reply(
    user_profile: dict,
    analytics_context: dict,
    conversation_history: list[dict],
    user_message: str,
    language: str = "vi",
) -> HealthChatResponse:
    """
    Calls Gemini for a contextual chat reply.
    Always returns a HealthChatResponse (with error fallback).
    """
    try:
        prompt = _build_chat_prompt(
            user_profile, analytics_context,
            conversation_history, user_message, language,
        )
        response = _generate_with_model_fallback(
            model_candidates=ARTICLE_MODELS,  # Flash is faster for chat
            contents=prompt,
            temperature=0.5,
        )
        return _parse_chat_response(response.text)
    except Exception as e:
        logger.error(f"[insight_service] Gemini chat failed: {e}")
        error_msg = (
            "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại."
            if language == "vi"
            else "Sorry, I'm experiencing a technical issue. Please try again."
        )
        return HealthChatResponse(
            reply=error_msg,
            suggested_questions=[],
        )
