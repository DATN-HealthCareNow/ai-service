import joblib
import pandas as pd

# load model
rf = joblib.load("app/models/rf_pipeline.pkl")

def predict(data):

    BMI = data.weight / ((data.height / 100) ** 2)

    sample = pd.DataFrame([{
        "steps": data.steps,
        "BMI": BMI,
        "age": data.age,
        "gender": data.gender,
        "distance": data.distance,
        "activity": data.activity
    }])

    pred = rf.predict(sample)[0]

    return {
        "heart_rate": float(pred[0]),
        "calories": float(pred[1])
    }