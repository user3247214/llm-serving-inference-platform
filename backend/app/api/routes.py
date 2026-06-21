from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.services.llm_service import llm_service

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": settings.model_name,
        "environment": settings.environment,
        "backend": llm_service.get_runtime_backend(),
    }


@router.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest):
    if payload.stream:
        return StreamingResponse(llm_service.stream_chat(payload), media_type="text/event-stream")
    return await llm_service.generate_chat(payload)
