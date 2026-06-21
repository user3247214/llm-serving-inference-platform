from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from importlib import import_module
from typing import Any, AsyncIterator

from fastapi import HTTPException

from app.core.config import settings
from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, Usage

try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMService:
    def __init__(self) -> None:
        self.engine: AsyncLLMEngine | None = None
        self.tokenizer = None
        self.hf_model = None
        self.auto_tokenizer_cls = None
        self.auto_model_cls = None
        self.runtime_backend = "uninitialized"
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def startup(self) -> None:
        if settings.use_mock_model:
            self.runtime_backend = "mock"
            return

        if settings.groq_api_key:
            self.runtime_backend = "groq"
            return

        try:
            transformers = import_module("transformers")
            self.auto_tokenizer_cls = transformers.AutoTokenizer
            self.auto_model_cls = transformers.AutoModelForCausalLM
        except Exception:
            self.runtime_backend = "mock"
            return

        self.tokenizer = self.auto_tokenizer_cls.from_pretrained(
            settings.model_name,
            trust_remote_code=settings.trust_remote_code,
        )

        if VLLM_AVAILABLE:
            engine_args = AsyncEngineArgs(
                model=settings.model_name,
                dtype=settings.dtype,
                quantization=settings.quantization,
                tensor_parallel_size=settings.tensor_parallel_size,
                gpu_memory_utilization=settings.gpu_memory_utilization,
                max_model_len=settings.max_model_len,
                max_num_batched_tokens=settings.max_num_batched_tokens,
                trust_remote_code=settings.trust_remote_code,
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.runtime_backend = "vllm"
            return

        try:
            import torch

            self.hf_model = self.auto_model_cls.from_pretrained(
                settings.model_name,
                trust_remote_code=settings.trust_remote_code,
            )
            self.hf_model.to(torch.device("cpu"))
            self.hf_model.eval()
            self.runtime_backend = "transformers"
        except Exception:
            self.runtime_backend = "mock"

    async def shutdown(self) -> None:
        self.engine = None
        self.hf_model = None
        self.tokenizer = None
        self.runtime_backend = "uninitialized"

    def _build_prompt(self, request: ChatCompletionRequest) -> str:
        assert self.tokenizer is not None
        messages = [message.model_dump() for message in request.messages]
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Plain causal models (e.g. distilgpt2) perform better with a short single-turn instruction.
        latest_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        return (
            "You are a concise helpful assistant. Reply in one short sentence.\n"
            f"User: {latest_user}\n"
            "Assistant:"
        )

    @staticmethod
    def _clean_transformers_output(text: str) -> str:
        cleaned = text.replace("\r", "").strip()
        for marker in ["\nUser:", "\nuser:", "\nAssistant:", "\nassistant:"]:
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        return cleaned.strip('" ').strip()

    @staticmethod
    def _token_estimate(text: str) -> int:
        return max(1, len(text.split()))

    @staticmethod
    def _sse(data: dict[str, Any] | str) -> str:
        payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        return f"data: {payload}\n\n"

    async def _generate_with_groq(self, request: ChatCompletionRequest) -> GenerationResult:
        import urllib.request

        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        model = settings.model_name if settings.model_name != "distilgpt2" else "llama-3.1-8b-instant"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        def _call() -> dict:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())

        data = await asyncio.to_thread(_call)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return GenerationResult(
            text=content,
            prompt_tokens=usage.get("prompt_tokens", self._token_estimate(" ".join(m.content for m in request.messages))),
            completion_tokens=usage.get("completion_tokens", self._token_estimate(content)),
        )

    async def _generate_with_vllm(self, request: ChatCompletionRequest) -> GenerationResult:
        if self.engine is None:
            raise HTTPException(status_code=503, detail="Model engine is not initialized")

        prompt = self._build_prompt(request)
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        request_id = str(uuid.uuid4())
        final_output = ""

        async for output in self.engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                final_output = output.outputs[0].text

        prompt_tokens = self._token_estimate(prompt)
        completion_tokens = self._token_estimate(final_output)
        return GenerationResult(final_output, prompt_tokens, completion_tokens)

    async def _generate_with_transformers(self, request: ChatCompletionRequest) -> GenerationResult:
        if self.hf_model is None or self.tokenizer is None:
            raise HTTPException(status_code=503, detail="Transformers backend is not initialized")

        prompt = self._build_prompt(request)

        def _run_generation() -> str:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output_ids = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    do_sample=request.temperature > 0,
                    temperature=max(0.01, request.temperature),
                    top_p=max(0.01, request.top_p),
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        final_output = await asyncio.to_thread(_run_generation)
        final_output = self._clean_transformers_output(final_output)
        if not final_output:
            final_output = "(empty model output)"

        prompt_tokens = self._token_estimate(prompt)
        completion_tokens = self._token_estimate(final_output)
        return GenerationResult(final_output, prompt_tokens, completion_tokens)

    async def _generate_with_mock(self, request: ChatCompletionRequest) -> GenerationResult:
        user_content = " ".join([msg.content for msg in request.messages if msg.role == "user"]).strip()
        answer = (
            "[mock-response] "
            f"Model '{settings.model_name}' received: {user_content[:200]}"
        )
        prompt_text = " ".join([m.content for m in request.messages])
        return GenerationResult(answer, self._token_estimate(prompt_text), self._token_estimate(answer))

    async def generate_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(status_code=400, detail="Use stream_chat for stream=true requests")

        async with self.semaphore:
            start = time.time()
            if self.runtime_backend == "mock":
                generation = await self._generate_with_mock(request)
            elif self.runtime_backend == "groq":
                generation = await self._generate_with_groq(request)
            elif self.runtime_backend == "vllm":
                generation = await self._generate_with_vllm(request)
            else:
                generation = await self._generate_with_transformers(request)

            _latency_ms = int((time.time() - start) * 1000)

        model_name = request.model or settings.model_name
        usage = Usage(
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.prompt_tokens + generation.completion_tokens,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:20]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generation.text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        model_name = request.model or settings.model_name
        response_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
        created = int(time.time())

        yield self._sse(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
        )

        final_text = ""
        async with self.semaphore:
            if self.runtime_backend == "mock":
                user_content = " ".join([m.content for m in request.messages if m.role == "user"]).strip()
                final_text = f"[mock-response] Model '{settings.model_name}' received: {user_content[:200]}"
            elif self.runtime_backend == "groq":
                generation = await self._generate_with_groq(request)
                final_text = generation.text
                parts = final_text.split(" ")
                for index, part in enumerate(parts):
                    token = part if index == len(parts) - 1 else f"{part} "
                    if token:
                        yield self._sse({"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]})
                parts = final_text.split(" ")
                for index, part in enumerate(parts):
                    token = part if index == len(parts) - 1 else f"{part} "
                    if token:
                        yield self._sse(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                            }
                        )
            elif self.runtime_backend == "vllm":
                if self.engine is None:
                    raise HTTPException(status_code=503, detail="Model engine is not initialized")

                prompt = self._build_prompt(request)
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                )

                request_id = str(uuid.uuid4())
                final_text = ""
                async for output in self.engine.generate(prompt, sampling_params, request_id):
                    if output.outputs:
                        text = output.outputs[0].text
                        delta = text[len(final_text) :]
                        if delta:
                            final_text = text
                            yield self._sse(
                                {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                                }
                            )
            else:
                generation = await self._generate_with_transformers(request)
                final_text = generation.text
                parts = final_text.split(" ")
                for index, part in enumerate(parts):
                    token = part if index == len(parts) - 1 else f"{part} "
                    if token:
                        yield self._sse(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                            }
                        )

        prompt_text = " ".join([m.content for m in request.messages])
        usage = {
            "prompt_tokens": self._token_estimate(prompt_text),
            "completion_tokens": self._token_estimate(final_text),
            "total_tokens": self._token_estimate(prompt_text) + self._token_estimate(final_text),
        }

        yield self._sse(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            }
        )
        yield self._sse("[DONE]")

    def get_runtime_backend(self) -> str:
        return self.runtime_backend


llm_service = LLMService()
