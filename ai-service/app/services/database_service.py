from pymongo import MongoClient
from app.core.config import settings
from datetime import datetime
import re

class DatabaseService:
    def __init__(self):
        self.client = MongoClient(settings.MONGO_URI)
        # Tự động lấy db name từ URI (thường là healthcare_core)
        self.db = self.client.get_default_database()
        self.collection = self.db["articles"]

    def to_slug(self, text: str):
        if not text:
            return f"article-{int(datetime.now().timestamp())}"
        # Chuyển về chữ thường, bỏ dấu và ký tự đặc biệt
        slug = text.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'\s+', '-', slug).strip('-')
        return slug if slug else f"article-{int(datetime.now().timestamp())}"

    def save_article(self, title: str, category: str, content: str):
        now = datetime.now()
        article_data = {
            "title": title,
            "slug": self.to_slug(title),
            "summary": content[:200] + "..." if len(content) > 200 else content,
            "content": content,
            "category": category,
            "status": "PUBLISHED", # Cho phép hiển thị ngay trên mobile
            "seoKeywords": [],
            "metaTitle": title,
            "metaDescription": content[:150],
            "coverImageUrl": "https://img.freepik.com/free-vector/healthy-lifestyle-concept-illustration_114360-6003.jpg", # Placeholder
            "aiGenerated": True,
            "authorId": "ai-service",
            "authorName": "HealthCareNow AI",
            "views": 0,
            "publishedAt": now,
            "createdAt": now,
            "updatedAt": now
        }
        result = self.collection.insert_one(article_data)
        return str(result.inserted_id)

db_service = DatabaseService()
