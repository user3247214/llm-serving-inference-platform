from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.services.llm_service import llm_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    await llm_service.startup()
    yield
    await llm_service.shutdown()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

allow_origins = [origin.strip() for origin in settings.cors_origins.split(",")]
allow_credentials = True
if len(allow_origins) == 1 and allow_origins[0] == "*":
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
