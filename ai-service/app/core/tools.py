"""
Tool Registry for HealthCare Now AI Agent (Phase 3).

These Python functions are passed to Gemini's Function Calling API.
The AI will autonomously decide WHEN and HOW to call them.
Add new tools here — the AI picks them up automatically.
"""
import logging

logger = logging.getLogger(__name__)

# Shared analytics context — injected by insight_service before each chat turn
_current_analytics: dict = {}

def set_analytics_context(ctx: dict):
    """Called by insight_service to inject live data before tool calls."""
    global _current_analytics
    _current_analytics = ctx or {}


def search_hospital(location: str) -> str:
    """Finds nearby hospitals, clinics, and pharmacies based on user location.

    Args:
        location: The location or district to search in (e.g. "Quận 5", "Bình Thạnh", "Ho Chi Minh City").
    """
    logger.info(f"[Tool] search_hospital called with location: {location}")
    loc = location.lower()
    if "quận 5" in loc or "quan 5" in loc or "district 5" in loc:
        return (
            "Các cơ sở y tế uy tín gần Quận 5:\n"
            "1. Bệnh viện Đại học Y Dược TP.HCM – 215 Hồng Bàng, Q5 (Hạng đặc biệt)\n"
            "2. Bệnh viện Chợ Rẫy – 201B Nguyễn Chí Thanh, Q5 (Hạng đặc biệt)\n"
            "3. Phòng khám Đa khoa Medic – 254 Hòa Hảo, Q10 (Gần Q5)\n"
            "4. Bệnh viện Nhân Dân 115 – 527 Sư Vạn Hạnh, Q10 (Gần Q5)"
        )
    if "bình thạnh" in loc or "binh thanh" in loc:
        return (
            "Các cơ sở y tế gần Bình Thạnh:\n"
            "1. Bệnh viện Bình Thạnh – 136 Đinh Tiên Hoàng\n"
            "2. Bệnh viện Gia An 115 – 97 Nguyễn Cửu Vân\n"
            "3. Phòng khám FV Hospital Satellite – Quận Bình Thạnh"
        )
    return (
        f"Các cơ sở y tế uy tín tại {location}:\n"
        "1. Bệnh viện Tâm Anh TP.HCM – 2B Phổ Quang, Tân Bình\n"
        "2. Bệnh viện Vinmec Central Park – 208 Nguyễn Hữu Cảnh, Bình Thạnh\n"
        "3. Bệnh viện FV – 6 Nguyễn Lương Bằng, Phú Mỹ Hưng\n"
        "Gợi ý: Gọi trước để đặt lịch hẹn tránh chờ lâu nhé."
    )


def get_health_metrics(metric_type: str) -> str:
    """Gets the user's current or recent health metrics from their tracking data.

    Args:
        metric_type: The type of metric to retrieve. Options: "heart_rate", "steps", "sleep", "calories", "bmi", "summary".
    """
    logger.info(f"[Tool] get_health_metrics called: {metric_type}, context keys: {list(_current_analytics.keys())}")
    ctx = _current_analytics
    mt = metric_type.lower()

    if mt in ("heart_rate", "nhip tim", "nhịp tim"):
        hr = ctx.get("heart_rate") or ctx.get("avg_heart_rate") or ctx.get("resting_heart_rate")
        if hr:
            return f"Nhịp tim gần nhất của bạn là {hr} bpm."
        return "Không có dữ liệu nhịp tim trong hồ sơ hiện tại."

    if mt in ("steps", "bước chân", "buoc chan"):
        stats = ctx.get("stats") or {}
        steps = ctx.get("steps_today") or stats.get("steps_avg_7d") or ctx.get("steps")
        if steps:
            return f"Số bước chân hôm nay (hoặc trung bình 7 ngày) của bạn là {int(steps):,} bước."
        return "Không có dữ liệu bước chân trong hồ sơ hiện tại."

    if mt in ("sleep", "giấc ngủ", "giac ngu"):
        sleep = ctx.get("sleep_hours") or ctx.get("avg_sleep") or (ctx.get("stats") or {}).get("sleep_avg")
        if sleep:
            return f"Thời gian ngủ trung bình gần đây của bạn là {sleep:.1f} tiếng."
        return "Không có dữ liệu giấc ngủ trong hồ sơ hiện tại."

    if mt in ("calories", "calo"):
        cal = ctx.get("calories_burned") or (ctx.get("stats") or {}).get("calories_avg")
        if cal:
            return f"Lượng calo tiêu thụ gần nhất của bạn là {int(cal)} kcal/ngày."
        return "Không có dữ liệu calories trong hồ sơ hiện tại."

    if mt in ("bmi",):
        bmi = ctx.get("bmi")
        cat = ctx.get("bmi_category", "")
        if bmi:
            return f"BMI hiện tại của bạn là {bmi:.1f} ({cat})."
        return "Không có dữ liệu BMI trong hồ sơ hiện tại."

    if mt in ("summary", "tổng quan", "all"):
        parts = []
        if ctx.get("bmi"):
            parts.append(f"BMI: {ctx['bmi']:.1f} ({ctx.get('bmi_category','')})")
        stats = ctx.get("stats") or {}
        if stats.get("steps_avg_7d"):
            parts.append(f"Bước chân TB 7 ngày: {int(stats['steps_avg_7d']):,}")
        if stats.get("calories_avg"):
            parts.append(f"Calo TB: {int(stats['calories_avg'])} kcal")
        if ctx.get("heart_rate"):
            parts.append(f"Nhịp tim: {ctx['heart_rate']} bpm")
        if parts:
            return "Tóm tắt sức khỏe của bạn:\n" + "\n".join(f"• {p}" for p in parts)
        return "Chưa có đủ dữ liệu để tóm tắt."

    return f"Không nhận dạng được loại chỉ số '{metric_type}'. Hãy thử: heart_rate, steps, sleep, calories, bmi."


AVAILABLE_TOOLS = [search_hospital, get_health_metrics]
