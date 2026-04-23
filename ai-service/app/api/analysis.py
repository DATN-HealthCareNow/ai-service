from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from app.services.gemini_service import analyze_medical_record_image
from app.services.database_service import db_service
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/v1/analysis/medical-record")
async def analyze_medical_record(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        json_str = analyze_medical_record_image(contents, file.content_type)
        
        # Try to parse string to dict to return as JSONResponse directly
        try:
            data = json.loads(json_str)
            
            # Lưu vào database nếu hồ sơ đọc được
            if data.get("is_readable", False):
                user_id = request.headers.get("x-user-id", "anonymous")
                record_id = db_service.save_medical_record(data, user_id)
                data["id"] = record_id
                
            return JSONResponse(content=data)
        except json.JSONDecodeError:
            # If not pure JSON, return as string or wrapped in dict
            return JSONResponse(content={"result": json_str})
            
    except Exception as e:
        logger.error(f"Error processing medical record: {e}")
        raise HTTPException(status_code=500, detail=str(e))
