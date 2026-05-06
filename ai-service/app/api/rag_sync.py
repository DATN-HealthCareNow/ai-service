"""
RAG Sync API — Webhook endpoints to index health data into Vector DB.
Called by BFF/Core service whenever health records change.

POST /ai/v1/rag/sync          → Sync a single health record to Vector DB
POST /ai/v1/rag/sync-batch    → Sync multiple records at once
DELETE /ai/v1/rag/clear/{user_id} → Clear all vectors for a user
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional
from app.services.rag_service import (
    sync_health_record_to_vector_db,
    delete_user_vectors,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request Schemas ───────────────────────────────────────────────────────────

class RagSyncRequest(BaseModel):
    user_id: str
    record_type: Literal[
        "medical_record",
        "meal",
        "workout",
        "daily_health",
        "health_insight",
    ]
    record_data: dict[str, Any]


class RagSyncBatchRequest(BaseModel):
    user_id: str
    records: list[dict[str, Any]] = Field(min_length=1, max_length=50)


class RagSyncResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    record_type: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ai/v1/rag/sync", response_model=RagSyncResponse)
async def sync_record(request: RagSyncRequest, background_tasks: BackgroundTasks):
    """
    Sync a single health record to MongoDB Vector DB.
    Runs embedding + upsert in background so API returns immediately.
    """
    logger.info(f"[RAG Sync] Received sync request: user={request.user_id}, type={request.record_type}")

    # Run sync in background to avoid blocking the response
    background_tasks.add_task(
        sync_health_record_to_vector_db,
        user_id=request.user_id,
        record_type=request.record_type,
        record_data=request.record_data,
    )

    return RagSyncResponse(
        success=True,
        message=f"Sync queued for {request.record_type}. Embedding will complete shortly.",
        user_id=request.user_id,
        record_type=request.record_type,
    )


@router.post("/ai/v1/rag/sync-batch", response_model=RagSyncResponse)
async def sync_batch(request: RagSyncBatchRequest, background_tasks: BackgroundTasks):
    """
    Sync multiple records at once (e.g., initial data migration).
    Each record must have 'record_type' and 'record_data' fields.
    """
    logger.info(f"[RAG Sync] Batch sync: user={request.user_id}, count={len(request.records)}")

    for record in request.records:
        record_type = record.get("record_type")
        record_data = record.get("record_data", {})
        if not record_type:
            continue
        background_tasks.add_task(
            sync_health_record_to_vector_db,
            user_id=request.user_id,
            record_type=record_type,
            record_data=record_data,
        )

    return RagSyncResponse(
        success=True,
        message=f"{len(request.records)} records queued for background embedding.",
        user_id=request.user_id,
    )


@router.delete("/ai/v1/rag/clear/{user_id}", response_model=RagSyncResponse)
async def clear_user_vectors(user_id: str):
    """
    Delete all vector records for a specific user.
    Use when a user deletes their account or requests data erasure (GDPR).
    """
    try:
        deleted_count = await delete_user_vectors(user_id)
        return RagSyncResponse(
            success=True,
            message=f"Deleted {deleted_count} vectors for user {user_id}.",
            user_id=user_id,
        )
    except Exception as e:
        logger.error(f"[RAG Sync] Failed to clear vectors for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
