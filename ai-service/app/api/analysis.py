from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.services.gemini_service import analyze_medical_record_image
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/v1/analysis/medical-record")
async def analyze_medical_record(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        json_str = analyze_medical_record_image(contents, file.content_type)
        
        # Try to parse string to dict to return as JSONResponse directly
        try:
            data = json.loads(json_str)
            return JSONResponse(content=data)
        except json.JSONDecodeError:
            # If not pure JSON, return as string or wrapped in dict
            return JSONResponse(content={"result": json_str})
            
    except Exception as e:
        logger.error(f"Error processing medical record: {e}")
        raise HTTPException(status_code=500, detail=str(e))
