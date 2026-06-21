from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "LLM Serving API"
    app_version: str = "0.1.0"
    environment: str = "development"

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"

    model_name: str = "distilgpt2"
    quantization: str | None = "awq"
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    max_num_batched_tokens: int = 8192
    max_concurrent_requests: int = 64
    trust_remote_code: bool = False

    use_mock_model: bool = True
    groq_api_key: str = ""


settings = Settings()
