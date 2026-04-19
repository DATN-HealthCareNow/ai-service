from pydantic import BaseModel

class ArticleRequest(BaseModel):
    title: str
    category: str