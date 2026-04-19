from fastapi import APIRouter, HTTPException
from app.services.gemini_service import generate_article
from app.models.article_model import ArticleRequest

router = APIRouter(prefix="/ai", tags=["AI Article"])

@router.post("/generate-article")
async def create_article(data: ArticleRequest):
    try:
        content = generate_article(data.title, data.category)
        return {
            "title": data.title,
            "category": data.category,
            "content": content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))