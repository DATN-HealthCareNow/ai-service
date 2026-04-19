import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib

def main():
    # 1. Đọc dữ liệu
    df = pd.read_csv("wearables_monitoring_data.csv")

    # 2. Tính BMI
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)

    # 3. Làm sạch
    df = df[df["steps"] > 10]
    df = df[df["heart_rate"] < 200]
    df = df[df["calories"] > 0]
    df = df.dropna()

    # 4. Chọn feature
    feature_cols = ["steps", "BMI", "age", "gender", "distance"]
    cat_cols = ["activity"]

    X = df[feature_cols + cat_cols]
    y = df[["heart_rate", "calories"]]

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    print("Starting Random Forest optimization...")
    rf_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", MultiOutputRegressor(RandomForestRegressor(random_state=42)))
    ])

    rf_params = {
        'model__estimator__n_estimators': [100, 200, 300, 500],
        'model__estimator__max_depth': [None, 10, 20, 30],
        'model__estimator__min_samples_split': [2, 5, 10],
        'model__estimator__min_samples_leaf': [1, 2, 4]
    }

    rf_search = RandomizedSearchCV(rf_pipeline, param_distributions=rf_params, 
                                   n_iter=10, cv=3, scoring='neg_mean_absolute_error', 
                                   random_state=42, n_jobs=-1, verbose=1)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    print("Starting XGBoost optimization...")
    xgb_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", MultiOutputRegressor(XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0)))
    ])

    xgb_params = {
        'model__estimator__n_estimators': [100, 200, 300, 500],
        'model__estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__estimator__max_depth': [3, 4, 6, 8, 10],
        'model__estimator__subsample': [0.6, 0.8, 1.0],
        'model__estimator__colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb_search = RandomizedSearchCV(xgb_pipeline, param_distributions=xgb_params, 
                                    n_iter=15, cv=3, scoring='neg_mean_absolute_error', 
                                    random_state=42, n_jobs=-1, verbose=1)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    # Hàm đánh giá
    def evaluate_multi(model, X_eval, y_eval, name="Model"):
        y_pred = pd.DataFrame(model.predict(X_eval), columns=y_eval.columns)
        print(f"\n===== {name} Evaluation =====")
        for col in y_eval.columns:
            mae = mean_absolute_error(y_eval[col], y_pred[col])
            rmse = np.sqrt(mean_squared_error(y_eval[col], y_pred[col]))
            r2 = r2_score(y_eval[col], y_pred[col])
            print(f"{col}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

    evaluate_multi(best_rf, X_test, y_test, "Optimized Random Forest")
    evaluate_multi(best_xgb, X_test, y_test, "Optimized XGBoost")

    # Lưu pipeline
    os.makedirs("ai-service/app/models", exist_ok=True)
    joblib.dump(best_rf, "ai-service/app/models/rf_pipeline.pkl")
    joblib.dump(best_xgb, "ai-service/app/models/xgb_pipeline.pkl")
    print("Optimized models have been saved.")

if __name__ == "__main__":
    main()
