# filepath: d:\ki8\khoaluan\AI\ai-service\app\api\predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils import predict_heart_calories

router = APIRouter()

class PredictionRequest(BaseModel):
    steps: float
    age: int
    weight: float
    height: float
    gender: int  # 0 or 1
    distance: float
    activity: str
    forbidden_foods: list[str] = []

@router.post("/predict")
def predict_wearable(request: PredictionRequest):
    try:
        result = predict_heart_calories(
            request.steps, request.age, request.weight, request.height,
            request.gender, request.distance, request.activity,
            request.forbidden_foods
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))