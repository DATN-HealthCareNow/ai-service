"""
Layer 5 — Gemini AI Insight & Chat Orchestration
Responsibilities:
  - Build structured prompts from FEATURES (never raw data)
  - Call Gemini API with fallback
  - Parse and validate structured JSON response
  - Handle chat conversations with full analytics context
  - RAG: retrieve relevant vector context before answering
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
from app.services.rag_service import search_relevant_context, sync_health_record_to_vector_db
from app.core.prompts import (
    INTENT_CLASSIFICATION_PROMPT, BASE_SYSTEM_PROMPT, INTENT_INSTRUCTIONS,
    MEMORY_EXTRACTION_PROMPT, PROACTIVE_COACHING_PROMPT
)
import asyncio

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


async def extract_and_save_memory(user_id: str, user_message: str):
    """
    Background task: Analyzes user message for long-term facts/preferences and saves to Vector DB.
    """
    if not user_id or len(user_message) < 5:
        return
        
    try:
        prompt = f"{MEMORY_EXTRACTION_PROMPT}\n\nUser Message: {user_message}"
        response = _generate_with_model_fallback(
            model_candidates=ARTICLE_MODELS, # Fast model
            contents=prompt,
            temperature=0.0,
        )
        cleaned = _clean_json_text(response.text)
        data = json.loads(cleaned)
        facts = data.get("facts", [])
        
        for f in facts:
            fact_str = f.get("fact")
            category = f.get("category", "preference")
            if fact_str:
                logger.info(f"[Memory] Extracted new fact for user {user_id}: {fact_str} ({category})")
                await sync_health_record_to_vector_db(
                    user_id=user_id,
                    record_type="user_preference",
                    record_data={
                        "fact": fact_str,
                        "category": category,
                        "date": "Memory Extraction"
                    }
                )
    except Exception as e:
        logger.warning(f"[Memory] Failed to extract memory: {e}")


async def _detect_intent(user_message: str, conversation_history: list[dict]) -> str:
    """Uses a fast model to classify the intent of the user's message."""
    # Build a brief history context (last 3 messages)
    history_text = ""
    for msg in conversation_history[-3:]:
        role = "User" if msg["role"] == "user" else "AI"
        history_text += f"{role}: {msg['content']}\n"
        
    prompt = f"{INTENT_CLASSIFICATION_PROMPT}\n\nRecent History:\n{history_text}\nUser Message: {user_message}"
    
    try:
        response = _generate_with_model_fallback(
            model_candidates=ARTICLE_MODELS, # Use fast model
            contents=prompt,
            temperature=0.0, # Zero temperature for deterministic classification
        )
        cleaned = _clean_json_text(response.text)
        data = json.loads(cleaned)
        intent = data.get("intent", "casual_chat")
        # Ensure it's one of the known intents
        if intent not in INTENT_INSTRUCTIONS:
            return "casual_chat"
        return intent
    except Exception as e:
        logger.error(f"[insight_service] Intent detection failed: {e}")
        return "casual_chat"


def _build_chat_prompt(
    user_profile: dict,
    analytics_context: dict,
    conversation_history: list[dict],
    user_message: str,
    language: str,
    intent: str,
    rag_context: str = "",
) -> str:
    lang_instruction = (
        "Respond ENTIRELY in Vietnamese. Use friendly, empathetic tone."
        if language == "vi"
        else "Respond ENTIRELY in English. Use friendly, empathetic tone."
    )

    history_text = ""
    # Optimize conversation memory: keep last 5 messages
    for msg in conversation_history[-5:]:
        role = "User" if msg["role"] == "user" else "Health Coach"
        history_text += f"{role}: {msg['content']}\n"

    # Context inclusion depends on intent
    ctx_str = ""
    if intent in ["health_analysis", "risk_analysis", "medication"]:
        # Include full analytics
        ctx_str = json.dumps(analytics_context, ensure_ascii=False, indent=2)
    elif intent in ["emotional_support", "motivation"]:
        # Only include sleep, HR, and trends
        filtered_ctx = {
            "sleep_avg_hours": analytics_context.get("advanced", {}).get("sleep_avg_hours") if analytics_context.get("advanced") else None,
            "resting_hr_avg": analytics_context.get("advanced", {}).get("resting_hr_avg") if analytics_context.get("advanced") else None,
            "trends": analytics_context.get("trends", {})
        }
        ctx_str = json.dumps(filtered_ctx, ensure_ascii=False, indent=2)
    # casual_chat and emergency get no/minimal context to save tokens and prevent overfitting

    # RAG context section
    rag_section = ""
    if rag_context and intent != "emergency":
        rag_section = f"""
## RETRIEVED HEALTH HISTORY (from Vector DB - highly relevant)
This is specific historical data retrieved from Vector Search that is most relevant to the user's question.
Prioritize this information when it directly answers the user's question, especially regarding their preferences, allergies, or past medical records.
{rag_context}
"""

    system_prompt = BASE_SYSTEM_PROMPT.format(lang_instruction=lang_instruction)
    intent_instruction = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS["casual_chat"])

    return f"""{system_prompt}

{intent_instruction}

## CONTEXT
User Profile: Age {user_profile.get('age')}, Gender {user_profile.get('gender')}, Height {user_profile.get('height_cm')}cm, Weight {user_profile.get('weight_kg')}kg

Recent Analytics Context:
{ctx_str if ctx_str else "(No specific analytics needed for this intent)"}
{rag_section}
Conversation History:
{history_text if history_text else "(new conversation)"}

User Question:
{user_message}

---
## OUTPUT FORMAT (STRICT JSON)
Return ONLY valid JSON with exactly these keys:

{{
  "reply": "string (main answer)",
  "emotional_tone": "string (e.g., empathetic, encouraging, informative, urgent)",
  "risk_level": "low | medium | high",
  "recommendations": ["short bullet 1", "short bullet 2"],
  "suggested_actions": ["Log water", "Start workout", "Sleep early"],
  "suggested_questions": ["next question 1", "next question 2"],
  "detected_health_topics": ["sleep", "stress"],
  "requires_doctor_consultation": boolean,
  "requires_emergency_attention": boolean,
  "confidence_score": float (0.0 to 1.0)
}}

IMPORTANT:
- No markdown block around JSON. Return ONLY the raw JSON string.
- No explanation outside JSON.
"""


# ── Response Parsers ──────────────────────────────────────────────────────────

def _clean_json_text(text: str) -> str:
    if not text:
        return "{}"
    text = text.strip()
    
    # Extract JSON from markdown code block if present
    import re
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # Otherwise, extract from first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end >= start:
        return text[start:end+1]
        
    return text


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


def _parse_chat_response(raw_text: str, intent: str) -> HealthChatResponse:
    """Parse Gemini chat response into HealthChatResponse."""
    cleaned = _clean_json_text(raw_text)
    data = json.loads(cleaned)
    return HealthChatResponse(
        reply=data.get("reply", ""),
        intent=intent,
        emotional_tone=data.get("emotional_tone", "neutral"),
        risk_level=data.get("risk_level", "low"),
        recommendations=data.get("recommendations", []),
        suggested_actions=data.get("suggested_actions", []),
        suggested_questions=data.get("suggested_questions", []),
        detected_health_topics=data.get("detected_health_topics", []),
        requires_doctor_consultation=data.get("requires_doctor_consultation", False),
        requires_emergency_attention=(data.get("requires_emergency_attention", False) or intent == "emergency"),
        confidence_score=data.get("confidence_score", 1.0),
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


async def generate_proactive_coaching(
    user_id: str,
    user_profile: dict,
    analytics_context: dict,
    language: str = "vi",
) -> Optional[dict]:
    """
    Analyzes daily health data to proactively generate notifications (praise, warning, suggestion).
    Called by cron jobs.
    """
    lang_instruction = (
        "Respond ENTIRELY in Vietnamese."
        if language == "vi"
        else "Respond ENTIRELY in English."
    )
    
    ctx_str = json.dumps(analytics_context, ensure_ascii=False, indent=2)
    prompt = f"{PROACTIVE_COACHING_PROMPT.format(lang_instruction=lang_instruction)}\n\nUser Profile:\n{json.dumps(user_profile)}\n\nRecent Analytics Context:\n{ctx_str}"
    
    try:
        response = _generate_with_model_fallback(
            model_candidates=ARTICLE_MODELS, # Fast model is sufficient
            contents=prompt,
            temperature=0.4,
        )
        cleaned = _clean_json_text(response.text)
        data = json.loads(cleaned)
        
        # We return a dict that matches ProactiveCoachingResponse schema
        return data
    except Exception as e:
        logger.error(f"[insight_service] Proactive coaching generation failed: {e}")
        return None


async def generate_health_chat_reply(
    user_id: Optional[str],
    user_profile: dict,
    analytics_context: dict,
    conversation_history: list[dict],
    user_message: str,
    language: str = "vi",
) -> HealthChatResponse:
    """
    Multi-Intent RAG-enhanced chat reply pipeline:
    1. Detect Intent using a fast model.
    2. Search Vector DB ONLY if intent requires it.
    3. Build intent-specific prompt with tailored context.
    4. Call Gemini for structured response.
    """
    
    # ── Step 1: Detect Intent & Extract Memory ──────────────────────────────────
    intent = await _detect_intent(user_message, conversation_history)
    logger.info(f"[health-chat] Detected Intent: {intent}")
    
    if user_id:
        asyncio.create_task(extract_and_save_memory(user_id, user_message))

    # ── Step 2: RAG (Memory Retrieval) ──────────────────────────────────────────
    rag_context = ""
    # We search memory for ALL intents except emergency, so the AI can remember preferences 
    # even when having a casual chat or building a meal plan (health_analysis).
    if user_id and intent != "emergency":
        try:
            rag_context = await search_relevant_context(
                user_id=user_id,
                user_question=user_message,
                limit=3,
            )
            if rag_context:
                logger.info(f"[RAG] Found context for user={user_id}, intent={intent}")
        except Exception as rag_err:
            logger.warning(f"[RAG] Vector search failed (non-fatal): {rag_err}")

    # ── Step 3: Build & Send Prompt ─────────────────────────────────────────────
    try:
        from app.core.tools import AVAILABLE_TOOLS, set_analytics_context
        # Inject live analytics so tools can return real user data
        set_analytics_context(analytics_context)
        
        prompt = _build_chat_prompt(
            user_profile=user_profile,
            analytics_context=analytics_context,
            conversation_history=conversation_history,
            user_message=user_message,
            language=language,
            intent=intent,
            rag_context=rag_context,
        )
        response = _generate_with_model_fallback(
            model_candidates=ARTICLE_MODELS,  # Flash is faster for chat
            contents=prompt,
            temperature=0.5,
            tools=AVAILABLE_TOOLS,
        )
        
        # ── Step 4: Multi-Agent Tool Calling Loop ───────────────────────────────
        if getattr(response, "function_calls", None):
            logger.info(f"[Agent] AI decided to use {len(response.function_calls)} tool(s).")
            fn_call = response.function_calls[0]
            fn_name = fn_call.name
            args = dict(fn_call.args) if fn_call.args else {}
            
            tool_result = ""
            for tool in AVAILABLE_TOOLS:
                if tool.__name__ == fn_name:
                    logger.info(f"[Agent] Executing {fn_name} with args {args}")
                    try:
                        tool_result = tool(**args)
                    except Exception as e:
                        tool_result = f"Error executing tool: {e}"
                    break
            
            if not tool_result:
                tool_result = f"Tool {fn_name} not found."
                
            # Re-prompt Gemini with tool results
            new_prompt = prompt + f"\n\n## SYSTEM INFO - TOOL RESULT ({fn_name})\n{tool_result}\n\nNow, generate the final JSON response answering the user."
            logger.info(f"[Agent] Re-prompting with tool result.")
            response = _generate_with_model_fallback(
                model_candidates=ARTICLE_MODELS,
                contents=new_prompt,
                temperature=0.5,
            )

        return _parse_chat_response(response.text, intent)
    except Exception as e:
        logger.error(f"[insight_service] Gemini chat failed: {e}")
        error_msg = (
            "Xin lỗi, tôi đang gặp sự cố kỹ thuật. Vui lòng thử lại."
            if language == "vi"
            else "Sorry, I'm experiencing a technical issue. Please try again."
        )
        return HealthChatResponse(
            reply=error_msg,
            intent=intent,
            emotional_tone="apologetic",
            risk_level="low",
            recommendations=[],
            suggested_actions=[],
            suggested_questions=[],
        )
