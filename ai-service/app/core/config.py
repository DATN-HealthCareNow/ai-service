import os

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    MONGO_URI: str = os.getenv("MONGO_CORE_URI")

settings = Settings()