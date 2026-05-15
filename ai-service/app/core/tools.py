import logging

logger = logging.getLogger(__name__)

def search_hospital(location: str) -> str:
    """Finds nearby hospitals and clinics based on user location.
    
    Args:
        location: The location to search in (e.g. "Ho Chi Minh City", "District 1")
    """
    logger.info(f"[Tool] search_hospital called with location: {location}")
    return f"Tìm thấy các bệnh viện uy tín gần {location}: 1. Bệnh viện Đại học Y Dược (Quận 5), 2. Bệnh viện Chợ Rẫy (Quận 5), 3. Bệnh viện Tâm Anh."

def get_health_metrics(metric_type: str) -> str:
    """Gets the current user's real-time health metrics.
    
    Args:
        metric_type: The type of metric (e.g. "heart_rate", "sleep", "steps")
    """
    logger.info(f"[Tool] get_health_metrics called with metric_type: {metric_type}")
    if metric_type == "heart_rate":
        return "Nhịp tim hiện tại là 75 bpm (Bình thường)."
    elif metric_type == "steps":
        return "Số bước chân hôm nay là 4500 bước."
    elif metric_type == "sleep":
        return "Đêm qua ngủ 6 tiếng 30 phút."
    return "Không có dữ liệu cho loại này."

AVAILABLE_TOOLS = [search_hospital, get_health_metrics]
