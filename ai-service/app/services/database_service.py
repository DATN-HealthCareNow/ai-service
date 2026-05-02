from pymongo import MongoClient
from app.core.config import settings
from datetime import datetime

class DatabaseService:
    def __init__(self):
        self.client = MongoClient(settings.MONGO_URI)
        # Tự động lấy db name từ URI (thường là healthcare_core)
        self.db = self.client.get_default_database()
        self.collection = self.db["articles"]

    def save_medical_record(self, record_data: dict, user_id: str = "anonymous"):
        medical_records_col = self.db["medical_records"]
        now = datetime.now()
        
        document = {
            "userId": user_id,
            "diagnosis": record_data.get("diagnosis", ""),
            "medications": record_data.get("medications", []),
            "forbiddenFoods": record_data.get("forbidden_foods", []),
            "createdAt": now,
            "updatedAt": now
        }
        
        result = medical_records_col.insert_one(document)
        return str(result.inserted_id)

db_service = DatabaseService()
