"""
RAG Service — Embedding + Vector DB operations.

Pipeline:
  [Data] → generate_embedding() → store in MongoDB health_vectors
  [User Question] → generate_embedding() → $vectorSearch → context string
"""
import logging
import json
from typing import Any, List, Optional
from google import genai
from app.services.gemini_service import client
from app.core.database import get_database

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-004"   # 768-dim, Google native, no extra deps
VECTOR_COLLECTION_NAME = "health_vectors"


# ── Embedding ─────────────────────────────────────────────────────────────────

def generate_embedding(text: str) -> List[float]:
    """Convert text to vector using Google text-embedding-004."""
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        return response.embeddings[0].values
    except Exception as e:
        logger.error(f"[RAG] Embedding failed: {e}")
        return []


# ── Text Builder (ETL) ────────────────────────────────────────────────────────

def _build_text_for_record(record_type: str, record_data: dict) -> str:
    """
    Converts raw data dict to a human-readable text block for embedding.
    Better text = better semantic search.
    """
    if record_type == "medical_record":
        meds = record_data.get("medications", [])
        med_names = ", ".join(
            m.get("name", str(m)) if isinstance(m, dict) else str(m)
            for m in meds
        )
        return (
            f"Hồ sơ y tế:\n"
            f"  Chẩn đoán: {record_data.get('diagnosis', 'Không rõ')}\n"
            f"  Trạng thái: {record_data.get('status', 'ACTIVE')}\n"
            f"  Thuốc: {med_names or 'Không có'}\n"
            f"  Ngày khám: {record_data.get('visit_date', record_data.get('date', 'Không rõ'))}\n"
            f"  Ghi chú: {record_data.get('notes', record_data.get('note', ''))}"
        )

    elif record_type == "meal":
        foods = record_data.get("food_items", record_data.get("foods", []))
        food_names = ", ".join(
            f.get("name", str(f)) if isinstance(f, dict) else str(f)
            for f in foods
        )
        return (
            f"Bữa ăn:\n"
            f"  Thời điểm: {record_data.get('meal_type', record_data.get('type', 'Bữa ăn'))}\n"
            f"  Ngày: {record_data.get('date', 'Không rõ')}\n"
            f"  Thực phẩm: {food_names or 'Không rõ'}\n"
            f"  Tổng calo: {record_data.get('total_calories', record_data.get('calories', 'N/A'))} kcal\n"
            f"  Protein: {record_data.get('protein', 'N/A')}g  "
            f"Carbs: {record_data.get('carbs', 'N/A')}g  "
            f"Fat: {record_data.get('fat', 'N/A')}g"
        )

    elif record_type == "workout":
        return (
            f"Buổi tập luyện:\n"
            f"  Loại: {record_data.get('activity_type', record_data.get('type', 'Không rõ'))}\n"
            f"  Ngày: {record_data.get('date', 'Không rõ')}\n"
            f"  Thời gian: {record_data.get('duration_minutes', record_data.get('duration_secs', 0))} phút\n"
            f"  Calo đốt: {record_data.get('calories_burned', record_data.get('active_calories', 'N/A'))} kcal\n"
            f"  Cường độ: {record_data.get('intensity', 'N/A')}"
        )

    elif record_type == "daily_health":
        return (
            f"Số liệu sức khỏe hằng ngày:\n"
            f"  Ngày: {record_data.get('date', 'Không rõ')}\n"
            f"  Bước chân: {record_data.get('steps', 'N/A')}\n"
            f"  Nhịp tim TB: {record_data.get('heart_rate', 'N/A')} bpm\n"
            f"  Giấc ngủ: {record_data.get('sleep_minutes', 'N/A')} phút\n"
            f"  Calo tiêu thụ: {record_data.get('active_calories', 'N/A')} kcal"
        )

    elif record_type == "health_insight":
        return (
            f"Phân tích AI sức khỏe:\n"
            f"  Tóm tắt: {record_data.get('summary', 'N/A')}\n"
            f"  Điểm số sức khỏe: {record_data.get('health_score', 'N/A')}\n"
            f"  Rủi ro: {', '.join(record_data.get('risks', [])) or 'Không có'}\n"
            f"  Gợi ý: {', '.join(record_data.get('recommendations', [])) or 'N/A'}"
        )

    # fallback: generic JSON
    return f"Loại: {record_type}\nDữ liệu: {json.dumps(record_data, ensure_ascii=False)}"


# ── Write: Sync to Vector DB ──────────────────────────────────────────────────

async def sync_health_record_to_vector_db(
    user_id: str,
    record_type: str,
    record_data: dict[str, Any],
    record_id: Optional[str] = None,
):
    """
    ETL: Build readable text → embed → store in MongoDB health_vectors.
    Call this whenever a user saves a medical record, meal, or workout.
    """
    db = await get_database()
    collection = db[VECTOR_COLLECTION_NAME]

    # 1. Build searchable text
    text_content = _build_text_for_record(record_type, record_data)

    # 2. Generate vector
    vector = generate_embedding(text_content)
    if not vector:
        logger.warning(f"[RAG] Skipping sync for {record_type}: embedding failed")
        return

    # 3. Upsert into Vector DB
    doc = {
        "user_id": user_id,
        "record_id": record_id or record_data.get("id") or record_data.get("_id"),
        "record_type": record_type,
        "content": text_content,
        "embedding": vector,
        "metadata": {
            "date": record_data.get("date") or record_data.get("visit_date"),
            "diagnosis": record_data.get("diagnosis"),
            "meal_type": record_data.get("meal_type") or record_data.get("type"),
        }
    }

    # If record_id exists, upsert (update existing vector)
    if doc["record_id"]:
        await collection.update_one(
            {"user_id": user_id, "record_id": doc["record_id"]},
            {"$set": doc},
            upsert=True,
        )
    else:
        await collection.insert_one(doc)

    logger.info(f"[RAG] Synced {record_type} for user={user_id} | preview: {text_content[:80]}...")


# ── Read: Search Vector DB ────────────────────────────────────────────────────

async def search_relevant_context(
    user_id: str,
    user_question: str,
    limit: int = 3,
    min_score: float = 0.6,
) -> str:
    """
    Semantic search: embed question → $vectorSearch in MongoDB Atlas.
    Returns a formatted context string for LLM injection.

    Requires:
    - MongoDB Atlas (not local MongoDB)
    - Vector Search index named 'vector_index' on health_vectors.embedding
    """
    db = await get_database()
    collection = db[VECTOR_COLLECTION_NAME]

    query_vector = generate_embedding(user_question)
    if not query_vector:
        return ""

    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 50,
                    "limit": limit,
                    "filter": {"user_id": user_id},
                }
            },
            {
                "$project": {
                    "content": 1,
                    "record_type": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "_id": 0,
                }
            },
        ]

        results = await collection.aggregate(pipeline).to_list(length=limit)

        if not results:
            return ""

        context_parts = []
        for res in results:
            score = res.get("score", 0)
            if score >= min_score:
                context_parts.append(res.get("content", ""))

        if not context_parts:
            return ""

        return "\n\n".join(context_parts)

    except Exception as e:
        logger.error(f"[RAG] Vector search error for user={user_id}: {e}")
        return ""


# ── Delete: Remove user data ──────────────────────────────────────────────────

async def delete_user_vectors(user_id: str) -> int:
    """Delete all vector records belonging to a user (GDPR / account deletion)."""
    db = await get_database()
    collection = db[VECTOR_COLLECTION_NAME]
    result = await collection.delete_many({"user_id": user_id})
    logger.info(f"[RAG] Deleted {result.deleted_count} vectors for user={user_id}")
    return result.deleted_count
