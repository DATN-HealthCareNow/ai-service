from pydantic import BaseModel


class PredictRequest(BaseModel):
    steps: float 
    weight: float 
    height: float 
    age: int 
    gender: int 
    distance: float 
    activity: str 


class PredictResponse(BaseModel):
    heart_rate: float
    calories: float