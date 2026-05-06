# filepath: ai-service/app/main.py
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
from app.api.health_insights import router as health_insights_router
from app.api.rag_sync import router as rag_sync_router
from app.core.database import get_mongo_client, close_mongo_connection
from app.utils import load_models

import os

# CORS_ALLOW_ORIGINS từ env (danh sách cách nhau bởi dấu phẩy)
# Mặc định "*" để các service nội bộ (container, EC2) gọi được nhau.
# Trên production, Nginx đã chặn từ bên ngoài — chỉ traffic qua reverse proxy mới vào được.
_raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app = FastAPI(
    title="AI Service",
    description="Health AI with RAG-enhanced chat, insight analysis, and vector sync",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Load ML models
    load_models()
    # Initialize MongoDB connection
    try:
        get_mongo_client()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[Startup] MongoDB not available: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(article_router)
app.include_router(predict_router, prefix="/ai", tags=["AI"])
app.include_router(analysis_router, tags=["Analysis"])
app.include_router(health_insights_router, tags=["Health Insights"])
app.include_router(rag_sync_router, tags=["RAG Vector Sync"])