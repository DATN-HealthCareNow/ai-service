from fastapi import APIRouter, HTTPException
from app.services.gemini_service import generate_article
from app.models.article_model import ArticleRequest
from app.services.database_service import db_service

router = APIRouter(prefix="/ai", tags=["AI Article"])

@router.post("/generate-article")
async def create_article(data: ArticleRequest):
    try:
        content = generate_article(data.title, data.category)
        
        # Lưu vào database
        article_id = db_service.save_article(data.title, data.category, content)
        
        return {
            "id": article_id,
            "title": data.title,
            "category": data.category,
            "content": content,
            "status": "PUBLISHED"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))