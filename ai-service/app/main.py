# filepath: d:\ki8\khoaluan\AI\ai-service\app\main.py
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
load_dotenv(dotenv_path=".env")

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("Thiếu GEMINI_API_KEY")

from fastapi import FastAPI
from app.api.article import router as article_router
from app.api.predict import router as predict_router
from app.api.analysis import router as analysis_router
from app.utils import load_models  # 👈 Thêm import

origins = [
    "http://localhost:3000",
]
app = FastAPI(
    title="AI Service",
    description="Generate article using Gemini AI and predict wearable data",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👈 Thêm load models khi startup
@app.on_event("startup")
async def startup_event():
    load_models()

app.include_router(article_router)
app.include_router(predict_router, prefix="/ai", tags=["AI"])
app.include_router(analysis_router, tags=["Analysis"])