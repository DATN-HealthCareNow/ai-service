import json
import joblib
import pandas as pd
from typing import Dict, Any, List, Union
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

# === TOÁN HỌC HÓA DINH DƯỠNG (RULE-BASED) ===

def calculate_bmr(weight: float, height: float, age: int, gender: int) -> float:
    """Tính BMR theo công thức Mifflin-St Jeor"""
    s = 5 if gender == 1 else -161
    return (10 * weight) + (6.25 * height) - (5 * age) + s

def get_met_value(activity: str) -> float:
    met_map = {
        "RUN": 8.0,
        "WALK": 3.5,
        "YOGA": 2.5,
        "GYM": 5.0,
        "CYCLING": 7.5,
        "SLEEP": 1.0,
        "SITTING": 1.3
    }

    def normalize_activities(activity_input: Union[str, List[str], None]) -> List[str]:
        if isinstance(activity_input, list):
            normalized = [str(a).strip().upper() for a in activity_input if str(a).strip()]
        elif isinstance(activity_input, str):
            normalized = [activity_input.strip().upper()] if activity_input.strip() else []
        else:
            normalized = []

        if not normalized:
            return ["WALK"]

        # Loại duplicate nhưng giữ thứ tự chọn ban đầu
        return list(dict.fromkeys(normalized))

    def pick_dominant_activity(activities: List[str]) -> str:
        return max(activities, key=get_met_value)
    return met_map.get(activity.upper(), 3.0)
def calculate_activity_calories(activities: List[str], distance: float, weight: float, steps: int) -> float:
def calculate_activity_calories(activity: str, distance: float, weight: float, steps: int) -> float:
    """Tính Calo vận động dựa trên MET và Physics"""
    # Ưu tiên tính theo quãng đường chạy/đi bộ (Công thức: km * kg * 1.036)
    if distance > 0:
        return distance * weight * 1.036
    
    # Tính theo bước chân nếu không có quãng đường
    if steps > 0:
        dist_km = (steps * 0.7) / 1000 # 1 bước ~ 0.7m
        return dist_km * weight * 1.036
    
    dominant_activity = pick_dominant_activity(activities)
    met = get_met_value(dominant_activity)
    met = get_met_value(activity)
    return (met * 3.5 * weight / 200) * 30 

# === LOGIC LỌC VÀ CHỌN MÓN ĂN ===

def is_forbidden(food_item: Dict[str, Any], forbidden_foods: List[str]) -> bool:
    """Kiểm tra nghiêm ngặt thực phẩm cấm qua tên, nhãn dị ứng và loại"""
    food_name = food_item["name"].lower()
    allergens = [a.lower() for a in food_item.get("allergens", [])]
    
    for f in forbidden_foods:
        f_lower = f.lower()
        # 1. Khớp tên trực tiếp hoặc chứa từ khóa cấm (e.g., "Trứng" cấm luôn "Lòng trắng trứng")
        if f_lower in food_name:
            return True
        # 2. Khớp nhãn dị ứng (e.g., cấm "Trứng" lọc qua allergen "egg")
        if f_lower == "trứng" and "egg" in allergens:
            return True
        if f_lower == "sữa" and ("milk" in allergens or "dairy" in allergens):
            return True
        if f_lower in allergens:
            return True
    return False

def choose_food_with_quantity(category: str, target_calories: float, forbidden_foods: List[str]) -> Dict[str, Any]:
    """Chọn món và tính toán số lượng Gram cần thiết"""
    candidates = [
        item for item in FOOD_DB 
        if item["category"] == category and not is_forbidden(item, forbidden_foods)
    ]
    
    if not candidates:
        return {}
    
    # Chọn món có mật độ dinh dưỡng phù hợp (ở đây lấy món đầu tiên hoặc ngẫu nhiên để đa dạng)
    import random
    food = random.choice(candidates)
    
    # Tính toán số Gram: (Target Calo / Calo mỗi 100g) * 100
    cal_per_100g = food["nutrition_per_100g"]["calories"]
    quantity_g = (target_calories / cal_per_100g) * 100
    
    # Làm tròn số Gram cho đẹp
    quantity_g = round(quantity_g / 5) * 5 
    
    # Tính tổng dinh dưỡng cho phần ăn này
    total_metrics = {
        "calories": round(target_calories),
        "protein": round(food["nutrition_per_100g"]["protein_g"] * quantity_g / 100, 1),
        "fat": round(food["nutrition_per_100g"]["fat_g"] * quantity_g / 100, 1),
        "carb": round(food["nutrition_per_100g"]["carbs_g"] * quantity_g / 100, 1)
    }
    
    return {
        "name": food["name"],
        "quantity_g": quantity_g,
        "unit": "gram",
        "category": food["category"],
        "total_metrics": total_metrics,
        "note": f"Dựa trên mục tiêu {target_calories} kcal cho nhóm {category}"
    }

def build_meal_plan(total_calories: float, forbidden_foods: List[str]) -> List[Dict[str, Any]]:
    """Xây dựng thực đơn 3 bữa dựa trên tổng Calo mục tiêu"""
    meal_types = [
        {"type": "BREAKFAST", "ratio": 0.25},
        {"type": "LUNCH", "ratio": 0.40},
        {"type": "DINNER", "ratio": 0.35}
    ]
    
    plan = []
    for m in meal_types:
        meal_cal = total_calories * m["ratio"]
        
        # Mỗi bữa gồm Protein + Carb + Fiber/Fat
        foods = []
        foods.append(choose_food_with_quantity("PROTEIN", meal_cal * 0.4, forbidden_foods))
        foods.append(choose_food_with_quantity("CARB", meal_cal * 0.4, forbidden_foods))
        foods.append(choose_food_with_quantity("FIBER", meal_cal * 0.2, forbidden_foods))
        
        # Lọc bỏ các object rỗng
        foods = [f for f in foods if f]
        
        plan.append({
            "meal_type": m["type"],
            "total_meal_calories": round(meal_cal),
            "foods": foods
        })
        
    return plan
def predict_heart_calories(
    steps: float,
    age: int,
    weight: float,
    height: float,
    gender: int,
    distance: float,
    activity: Union[str, List[str], None],
    forbidden_foods: List[str] = []
) -> Dict[str, Any]:
    activities = normalize_activities(activity)
    dominant_activity = pick_dominant_activity(activities)

def predict_heart_calories(steps: float, age: int, weight: float, height: float, gender: int, distance: float, activity: str, forbidden_foods: List[str] = []) -> Dict[str, Any]:
    # 1. TÍNH TOÁN CỨNG (RULE-BASED)
    activity_kcal = calculate_activity_calories(activities, distance, weight, int(steps))
    activity_kcal = calculate_activity_calories(activity, distance, weight, int(steps))
    
    # TDEE = BMR * PAL (Physical Activity Level) + Activity_Kcal
    # PAL mặc định cho người ít vận động là 1.2
    tdee = (bmr * 1.2) + activity_kcal
    
    # 2. AI INSIGHT (Dự đoán nhịp tim & Calo tham khảo từ XGBoost)
    # Giữ lại để tham khảo hoặc dùng cho UI "AI Prediction"
    BMI = weight / ((height / 100) ** 2)
    sample = pd.DataFrame([{
        "steps": steps,
        "BMI": BMI,
        "age": age,
        "gender": gender,
        "activity": "Running 7 METs" if dominant_activity == "RUN" else "Self Pace walk"
        "activity": "Running 7 METs" if activity.upper() == "RUN" else "Self Pace walk"
    }])
    
    ai_heart_rate = 90.0 # Default
    if xgb_model:
        pred = xgb_model.predict(sample)[0]
        ai_heart_rate = float(pred[0])

    # 3. PHÂN BỔ MACROS (Protein 30%, Carb 40%, Fat 30%)
    target_protein_g = (tdee * 0.30) / 4
    target_carb_g = (tdee * 0.40) / 4
    target_fat_g = (tdee * 0.30) / 9

    return {
        "summary": {
            "bmr": round(bmr),
            "activities": activities,
            "dominant_activity": dominant_activity,
            "activity_calories": round(activity_kcal),
            "total_tdee": round(tdee),
            "macros_target": {
                "protein_g": round(target_protein_g),
                "carb_g": round(target_carb_g),
                "fat_g": round(target_fat_g)
            }
        },
        "heart_rate_insight": round(ai_heart_rate),
        "meals": build_meal_plan(tdee, forbidden_foods)
    }