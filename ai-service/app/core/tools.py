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
        metric_type: The type of metric to retrieve. Options:
            - "heart_rate": Today's heart rate (bpm)
            - "heart_rate_weekly": Weekly average heart rate (bpm)
            - "steps": Today's step count
            - "steps_weekly": Weekly average step count
            - "sleep": Recent sleep hours
            - "calories": Today's calories burned
            - "calories_weekly": Weekly average calories
            - "bmi": Current BMI
            - "summary": Full today's health summary
    """
    logger.info(f"[Tool] get_health_metrics called: {metric_type}, context keys: {list(_current_analytics.keys())}")
    ctx = _current_analytics
    mt = metric_type.lower()
    weekly = ctx.get("weekly_data") or []

    # ── Helper: compute weekly avg from weekly_data ────────────────────────────
    def _weekly_avg(field: str) -> float | None:
        vals = [d.get(field) for d in weekly if d.get(field) is not None and d.get(field) > 0]
        return round(sum(vals) / len(vals), 1) if vals else None

    # ── Heart rate ─────────────────────────────────────────────────────────────
    if mt in ("heart_rate", "nhip tim", "nhịp tim", "nhịp tim hôm nay"):
        hr = ctx.get("heart_rate") or ctx.get("avg_heart_rate") or ctx.get("resting_heart_rate")
        if hr:
            return f"Nhịp tim hôm nay của bạn là {hr} bpm."
        return "Không có dữ liệu nhịp tim hôm nay."

    if mt in ("heart_rate_weekly", "nhịp tim trung bình", "nhịp tim tuần", "heart_rate_avg"):
        avg = _weekly_avg("heart_rate")
        if avg:
            return f"Nhịp tim trung bình 7 ngày qua của bạn là {avg} bpm."
        # Fallback to today's if no weekly
        hr = ctx.get("heart_rate")
        if hr:
            return f"Chỉ có dữ liệu hôm nay: nhịp tim {hr} bpm (chưa đủ 7 ngày để tính trung bình)."
        return "Không có dữ liệu nhịp tim trong tuần này."

    # ── Steps ──────────────────────────────────────────────────────────────────
    if mt in ("steps", "bước chân", "buoc chan", "bước chân hôm nay"):
        stats = ctx.get("stats") or {}
        steps = ctx.get("steps_today") or stats.get("steps_avg_7d") or ctx.get("steps")
        if steps:
            label = "hôm nay" if ctx.get("steps_today") else "trung bình 7 ngày"
            return f"Số bước chân {label} của bạn là {int(steps):,} bước."
        return "Không có dữ liệu bước chân hôm nay."

    if mt in ("steps_weekly", "bước chân trung bình", "bước chân tuần", "steps_avg"):
        avg = _weekly_avg("steps")
        stats = ctx.get("stats") or {}
        if avg or stats.get("steps_avg_7d"):
            val = avg or stats.get("steps_avg_7d")
            return f"Số bước chân trung bình 7 ngày qua của bạn là {int(val):,} bước/ngày."
        return "Không có đủ dữ liệu bước chân trong tuần này."

    # ── Sleep ──────────────────────────────────────────────────────────────────
    if mt in ("sleep", "giấc ngủ", "giac ngu"):
        sleep_min = ctx.get("sleep_minutes_today")
        if sleep_min:
            return f"Tối qua bạn ngủ {sleep_min // 60} tiếng {sleep_min % 60} phút."
        avg_min = _weekly_avg("sleep_minutes")
        if avg_min:
            return f"Thời gian ngủ trung bình 7 ngày qua của bạn là {avg_min/60:.1f} tiếng."
        sleep_h = ctx.get("sleep_hours") or (ctx.get("stats") or {}).get("sleep_avg")
        if sleep_h:
            return f"Thời gian ngủ trung bình gần đây là {float(sleep_h):.1f} tiếng."
        return "Không có dữ liệu giấc ngủ trong hồ sơ hiện tại."

    # ── Calories ───────────────────────────────────────────────────────────────
    if mt in ("calories", "calo", "calories hôm nay"):
        cal = ctx.get("calories_today") or ctx.get("calories_burned") or (ctx.get("stats") or {}).get("calories_avg")
        if cal:
            label = "hôm nay" if ctx.get("calories_today") else "trung bình"
            return f"Lượng calo tiêu thụ {label} của bạn là {int(cal)} kcal."
        return "Không có dữ liệu calories trong hồ sơ hiện tại."

    if mt in ("calories_weekly", "calo trung bình", "calo tuần", "calories_avg"):
        avg = _weekly_avg("calories")
        stats = ctx.get("stats") or {}
        if avg or stats.get("calories_avg"):
            val = avg or stats.get("calories_avg")
            return f"Lượng calo tiêu thụ trung bình 7 ngày qua là {int(val)} kcal/ngày."
        return "Không có đủ dữ liệu calories trong tuần này."

    # ── BMI ────────────────────────────────────────────────────────────────────
    if mt in ("bmi",):
        bmi = ctx.get("bmi")
        cat = ctx.get("bmi_category", "")
        if bmi:
            return f"BMI hiện tại của bạn là {bmi:.1f} ({cat})."
        return "Không có dữ liệu BMI trong hồ sơ hiện tại."

    # ── Summary ────────────────────────────────────────────────────────────────
    if mt in ("summary", "tổng quan", "all"):
        parts = []
        if ctx.get("bmi"):
            parts.append(f"BMI: {ctx['bmi']:.1f} ({ctx.get('bmi_category','')})")
        steps = ctx.get("steps_today") or (ctx.get("stats") or {}).get("steps_avg_7d")
        if steps:
            label = "hôm nay" if ctx.get("steps_today") else "TB 7 ngày"
            parts.append(f"Bước chân {label}: {int(steps):,}")
        cal = ctx.get("calories_today") or (ctx.get("stats") or {}).get("calories_avg")
        if cal:
            parts.append(f"Calo: {int(cal)} kcal")
        if ctx.get("heart_rate"):
            parts.append(f"Nhịp tim: {ctx['heart_rate']} bpm")
        if parts:
            return "Tóm tắt sức khỏe hôm nay của bạn:\n" + "\n".join(f"• {p}" for p in parts)
        return "Chưa có đủ dữ liệu để tóm tắt."

    return f"Không nhận dạng được loại chỉ số '{metric_type}'. Hãy thử: heart_rate, heart_rate_weekly, steps, steps_weekly, sleep, calories, bmi."


AVAILABLE_TOOLS = [search_hospital, get_health_metrics]
