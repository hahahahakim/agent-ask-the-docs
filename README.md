# Ask the Docs — AI Agent for 0G Labs

An AI-powered documentation agent that searches and synthesizes answers from 0G Labs official resources. Ask questions in plain English and get direct, cited answers — no documentation digging required.

**What you can ask:**

| Topic | Example |
|---|---|
| Storage | *"How do I upload a file to 0G Storage?"* |
| Inference & AI Models | *"What AI models are available on mainnet?"* |
| Data Availability | *"How do I submit a blob to 0G DA?"* |
| Compute Network | *"How do I access the inference API?"* |
| Staking & Delegation | *"How do I delegate my tokens?"* |
| Smart Contracts & EVM | *"What is the 0G chain ID and RPC URL?"* |
| Network Info | *"What are the testnet endpoints?"* |
| Private Computer | *"What is verifiable inference on 0G?"* |

---

## Setup

**Requirements:** Python 3.9+

```bash
git clone https://github.com/YOUR_USERNAME/agent-ask-the-docs.git
cd agent-ask-the-docs
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your values:

```env
# Required
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-0g-inference-endpoint/v1
MODEL_NAME=your_model_name

# Required for API server only
API_KEYS=your-generated-key
CORS_ALLOWED_ORIGINS=https://yoursite.com
```

Generate an API key:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## Running

### CLI

```bash
# Interactive mode
python3 agent.py

# Single query
python3 agent.py "What is 0G storage?"
```

**Interactive commands:** `exit` to quit · `clear` to reset conversation · `recache` to refresh docs cache

### API Server

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

> **Note:** Always run with a single worker. Conversation history is in-process and not shared across workers.

---

## API Reference

All endpoints except `/health` require the `X-API-Key` header.

### GET /health

```bash
curl http://localhost:8000/health
```

### POST /chat

Returns a complete answer.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "What is 0G storage?", "thread_id": "user-123"}'
```

**Response:**
```json
{
  "answer": "...",
  "thread_id": "user-123",
  "model": "your-model",
  "duration_s": 2.34
}
```

`thread_id` is optional — a UUID is generated if omitted. Pass the same `thread_id` in follow-up messages to continue a conversation.

### POST /chat/stream

Streams the answer token-by-token using Server-Sent Events.

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -N \
  -d '{"query": "What is 0G storage?", "thread_id": "user-123"}'
```

**SSE frames:**
```
data: {"token": "0G Storage is..."}
data: {"model": "your-model", "duration_s": 3.12}
data: [DONE]

event: error
data: {"message": "Stream interrupted. Retry via POST /chat with the same query and thread_id."}
```

**JavaScript example:**
```javascript
const res = await fetch("/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
  body: JSON.stringify({ query, thread_id }),
});
const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
    const { token } = JSON.parse(line.slice(6));
    if (token) process.stdout.write(token);
  }
}
```

### POST /chat — Cache Invalidation

Send `"recache"` as the query to flush the full documentation cache and vector index. The agent will re-fetch all pages on the next query.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "recache", "thread_id": "admin"}'
```

### Error Responses

| Status | Meaning |
|---|---|
| `401` | Missing or invalid `X-API-Key` |
| `422` | Request validation failed |
| `429` | Rate limit exceeded (20 req/min per key) |
| `500` | Internal server error |

---

## Docker

```bash
docker build -t agent-ask-the-docs .

docker run -d \
  --name agent-docs \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-key" \
  -e OPENAI_BASE_URL="https://your-endpoint/v1" \
  -e MODEL_NAME="your-model" \
  -e API_KEYS="your-generated-key" \
  -e CORS_ALLOWED_ORIGINS="https://yoursite.com" \
  -v agent-data:/app/data \
  agent-ask-the-docs
```

**Docker Compose:**
```yaml
services:
  agent:
    image: agent-ask-the-docs:latest
    ports:
      - "8000:8000"
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      OPENAI_BASE_URL: ${OPENAI_BASE_URL}
      MODEL_NAME: ${MODEL_NAME}
      API_KEYS: ${API_KEYS}
      CORS_ALLOWED_ORIGINS: https://yoursite.com
    volumes:
      - agent-data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  agent-data:
```

> **Note:** Always run with a single worker. Conversation history is stored in-process and is not shared across workers.

---

## Tests

```bash
python3 -m pytest tests/ -v
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | LLM API key (required) |
| `OPENAI_BASE_URL` | OpenAI | OpenAI-compatible endpoint URL |
| `MODEL_NAME` | `gpt-4o` | Model name |
| `API_KEYS` | — | Comma-separated API keys (required for server) |
| `CORS_ALLOWED_ORIGINS` | — | Comma-separated allowed origins (required for server) |
| `RATE_LIMIT_PER_MINUTE` | `20` | Requests per minute per API key |
| `THREAD_TTL_HOURS` | `24` | Hours before inactive threads are wiped |
| `DATA_DIR` | `./data` | Directory for SQLite caches |
| `VERBOSE` | `false` | Show LangGraph debug traces |
