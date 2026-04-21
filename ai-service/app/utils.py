# filepath: d:\ki8\khoaluan\AI\ai-service\app\utils.py
import json
import joblib
import pandas as pd
import os
from typing import Dict, Any, List

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FOOD_DB_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "food_db.json"))
rf_model = None
xgb_model = None
FOOD_DB: List[Dict[str, Any]] = []


def load_food_db() -> List[Dict[str, Any]]:
    try:
        with open(FOOD_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading food_db.json: {e}")
        return []


FOOD_DB = load_food_db()


def load_models():
    global rf_model, xgb_model
    try:
        rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_pipeline.pkl"))
        xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


def get_food_recommendations(calories: float, count: int = 5) -> List[Dict[str, Any]]:
    if not FOOD_DB:
        return []

    # Tìm các nguyên liệu có lượng calo/100g gần với mục tiêu nhất
    sorted_foods = sorted(
        FOOD_DB,
        key=lambda item: abs(item.get("nutrition_per_100g", {}).get("calories", 0) - calories)
    )

    return sorted_foods[:count]


def choose_food_by_category(category: str, target_calories: float) -> Dict[str, Any]:
    candidates = [item for item in FOOD_DB if item["category"] == category]
    if not candidates:
        return {}
    return min(candidates, key=lambda item: abs(item.get("nutrition_per_100g", {}).get("calories", 0) - target_calories))


def build_detailed_meal_suggestions(calories: float) -> List[Dict[str, Any]]:
    if not FOOD_DB:
        return []

    # Gợi ý mỗi nhóm một nguyên liệu chi tiết
    protein = choose_food_by_category("PROTEIN", calories * 0.4)
    carb = choose_food_by_category("CARB", calories * 0.4)
    fiber = choose_food_by_category("FIBER", calories * 0.1)
    fat = choose_food_by_category("FAT", calories * 0.1)

    # Trả về danh sách các object chi tiết
    ingredients = [item for item in [protein, carb, fiber, fat] if item]
    return ingredients


# Hệ số điều chỉnh để calo thực tế hơn (do dữ liệu gốc có thể tính theo interval ngắn)
CALORIE_SCALING_FACTOR = 2.8

# Danh sách các nhãn hoạt động hợp lệ trong dataset huấn luyện
VALID_DATASET_ACTIVITIES = [
    "Lying", "Sitting", "Self Pace walk", 
    "Running 3 METs", "Running 5 METs", "Running 7 METs"
]

# Ánh xạ nhãn từ mobile/user sang nhãn trong dataset huấn luyện
ACTIVITY_MAPPING = {
    "RUN": "Running 7 METs",
    "CYCLING": "Running 5 METs",
    "GYM": "Running 3 METs",
    "YOGA": "Self Pace walk"
}


def predict_heart_calories(steps: float, age: int, weight: float, height: float, gender: int, distance: float, activity: str) -> Dict[str, Any]:
    # Logic xử lý nhãn hoạt động:
    # 1. Nếu rỗng -> mặc định "Self Pace walk"
    # 2. Nếu đã là nhãn chuẩn trong dataset -> giữ nguyên
    # 3. Nếu là nhãn mobile (RUN, GYM...) -> ánh xạ sang nhãn chuẩn
    # 4. Nếu không khớp gì -> mặc định "Self Pace walk"
    
    act_upper = activity.strip().upper() if activity else ""
    
    # Tìm kiếm trong bảng ánh xạ trước
    mapped_activity = ACTIVITY_MAPPING.get(act_upper)
    
    if not mapped_activity:
        # Nếu không có trong bảng ánh xạ, kiểm tra xem có phải nhãn chuẩn không (không phân biệt hoa thường)
        for valid_act in VALID_DATASET_ACTIVITIES:
            if valid_act.upper() == act_upper:
                mapped_activity = valid_act
                break
    
    # Cuối cùng nếu vẫn không thấy thì mặc định
    if not mapped_activity:
        mapped_activity = "Self Pace walk"
    
    BMI = weight / ((height / 100) ** 2)
    sample = pd.DataFrame([{
        "steps": steps,
        "BMI": BMI,
        "age": age,
        "gender": gender,
        "distance": distance,
        "activity": mapped_activity
    }])
    pred = xgb_model.predict(sample)[0]
    
    # Áp dụng hệ số nhân để Calo sát với thực tế vận động hàng ngày hơn
    raw_calories = float(pred[1])
    adjusted_calories = raw_calories * CALORIE_SCALING_FACTOR
    
    return {
        "heart_rate": float(pred[0]),
        "calories": adjusted_calories,
        "recommended_foods": build_detailed_meal_suggestions(adjusted_calories)
    }