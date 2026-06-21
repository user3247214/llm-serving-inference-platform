# Architecture

## Overview

This platform serves open-source LLMs with low-latency inference using `vLLM` as the execution engine, `FastAPI` as the API layer, and `React` as an operator-facing chat console.

## Components

1. Backend API (`backend/app/main.py`)
- FastAPI app with CORS and lifecycle hooks.
- Exposes `GET /health` and `POST /v1/chat/completions`.

2. Inference Service (`backend/app/services/llm_service.py`)
- Initializes `AsyncLLMEngine` from vLLM.
- Uses internal vLLM scheduling for dynamic batching.
- Supports quantization options via env (`QUANTIZATION=awq|gptq|None`).
- Supports mock mode (`USE_MOCK_MODEL=true`) for local development without GPUs.

3. Frontend Console (`frontend/src/App.jsx`)
- React chat client that sends OpenAI-style payloads.
- Displays last-request latency and token usage.

## Request Flow

1. User submits prompt in React UI.
2. Frontend sends request to `POST /v1/chat/completions`.
3. FastAPI validates payload and delegates to `LLMService`.
4. vLLM generates response (batched/scheduled internally).
5. API returns OpenAI-compatible response with usage stats (standard or SSE stream).

## Scaling Notes

- Increase `TENSOR_PARALLEL_SIZE` for multi-GPU serving.
- Tune `MAX_NUM_BATCHED_TOKENS` based on latency/throughput SLOs.
- Run multiple backend replicas behind a load balancer for horizontal scale.
