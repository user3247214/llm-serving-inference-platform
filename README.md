# LLM Serving & Inference Platform

High-performance LLM serving platform for efficient deployment of open-source models with quantization and dynamic batching.

## What Is Implemented

- FastAPI backend with OpenAI-style endpoint: `POST /v1/chat/completions`
- vLLM async engine integration for efficient generation and scheduler-driven batching
- Quantization-ready model runtime config (AWQ/GPTQ/none via env)
- React inference console for interactive chat, latency, and token visibility
- Dockerfiles and `docker-compose.yml` for one-command local startup
- Mock model mode for development on machines without a GPU

## Repository Layout

```text
backend/
	app/
		api/routes.py
		core/config.py
		models/schemas.py
		services/llm_service.py
		main.py
	requirements.txt
	Dockerfile
	.env.example

frontend/
	src/App.jsx
	src/main.jsx
	src/styles.css
	package.json
	Dockerfile

docs/
	architecture.md

docker-compose.yml
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- (Optional for real inference) NVIDIA GPU + CUDA-compatible environment

## Backend: Run Locally

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will start at `http://localhost:8000`.

## Frontend: Run Locally

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

The UI will start at `http://localhost:5173`.

## Run with Docker Compose

```bash
docker compose up --build
```

Services:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

## API Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d "{\"model\":\"distilgpt2\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence\"}],\"temperature\":0.7,\"top_p\":0.95,\"max_tokens\":64,\"stream\":false}"
```

## Configuration

Set in `backend/.env`:

- `MODEL_NAME`: Hugging Face model ID (default `distilgpt2` for CPU-friendly local runs)
- `QUANTIZATION`: `awq`, `gptq`, or empty for non-quantized
- `TENSOR_PARALLEL_SIZE`: Number of GPUs for tensor parallelism
- `GPU_MEMORY_UTILIZATION`: Fraction of GPU memory reserved by vLLM
- `MAX_MODEL_LEN`: Maximum context length
- `MAX_NUM_BATCHED_TOKENS`: Controls dynamic batching capacity
- `MAX_CONCURRENT_REQUESTS`: API-level concurrency guard
- `USE_MOCK_MODEL`: `true` for mock responses, `false` for real inference

## Deploy (GitHub + Vercel + Render)

1. Push to GitHub
- Install Git on your machine if needed.
- From repo root:

```bash
git add .
git commit -m "LLM serving platform: backend, streaming, frontend, deploy config"
git push origin main
```

2. Deploy backend on Render
- Create a new Render Blueprint or Web Service from this repo.
- Use `render.yaml` at repo root.
- After deploy, copy your backend URL (for example `https://your-backend.onrender.com`).
- In Render environment variables, set `CORS_ORIGINS` to your final Vercel URL.

3. Deploy frontend on Vercel
- Import the GitHub repo into Vercel.
- Set project root to `frontend`.
- Set environment variables:
	- `VITE_API_BASE_URL=https://your-backend.onrender.com`
	- `VITE_MODEL_NAME=distilgpt2`
- Redeploy and open your Vercel URL.

4. Validate public demo
- `GET https://your-backend.onrender.com/health`
- Open your Vercel URL and run a chat prompt.

## Notes

- Dynamic batching is handled internally by vLLM's scheduler.
- Both non-streaming and SSE streaming responses are supported (`stream=false|true`).
- For production, add auth, rate limiting, observability, and autoscaling.
