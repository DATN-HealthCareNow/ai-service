from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    steps: float 
    weight: float 
    height: float 
    age: int 
    gender: int 
    distance: float 
    activity: Optional[str] = None
    activities: Optional[list[str]] = None


class PredictResponse(BaseModel):
    heart_rate: float
    calories: float