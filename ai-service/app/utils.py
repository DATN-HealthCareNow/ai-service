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

    sorted_foods = sorted(
        FOOD_DB,
        key=lambda item: abs(item.get("nutrition", {}).get("calories", 0) - calories)
    )

    return sorted_foods[:count]


def build_full_meal_suggestions(calories: float, count: int = 3) -> List[str]:
    foods = get_food_recommendations(calories, count)
    return [item["name"] for item in foods]


def predict_heart_calories(steps: float, age: int, weight: float, height: float, gender: int, distance: float, activity: str) -> Dict[str, Any]:
    BMI = weight / ((height / 100) ** 2)
    sample = pd.DataFrame([{
        "steps": steps,
        "BMI": BMI,
        "age": age,
        "gender": gender,
        "distance": distance,
        "activity": activity
    }])
    pred = xgb_model.predict(sample)[0]
    calories = float(pred[1])
    return {
        "heart_rate": float(pred[0]),
        "calories": calories,
        "recommended_meals": build_full_meal_suggestions(calories),
        "recommended_foods": get_food_recommendations(calories)
    }