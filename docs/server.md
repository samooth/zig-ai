# Server Mode — HTTP API

zig-ai-engine includes a built-in HTTP server compatible with the OpenAI, Anthropic, and Ollama APIs. All three sets of endpoints are exposed simultaneously on the same server instance. There is no flag to select one API over another — the client chooses which endpoint to call.

## Quick start

```bash
# Start the server with a GGUF model
./zig-out/bin/zig-ai-engine \
  --model modelo.gguf \
  --server \
  --server-port 8080 \
  --server-host 0.0.0.0
```

## Supported APIs

| API | Endpoint | Notes |
|---|---|---|
| OpenAI | `/v1/chat/completions` | `stream: true/false`, function calling not yet supported |
| Anthropic | `/v1/messages` | SSE streaming, system prompt, multi-turn |
| Ollama | `/api/generate`, `/api/chat` | Model listing not yet supported |

## Flags

| Flag | Default | Description |
|---|---|---|
| `--server` | off | Enable HTTP server mode |
| `--server-host <addr>` | `127.0.0.1` | Bind address |
| `--server-port <n>` | `8080` | Bind port |
| `--server-auth <keys>` | — | Comma-separated API keys (Bearer token auth) |
| `--server-max-queue <n>` | `256` | Max concurrent requests in queue |
| `--server-timeout <ms>` | `60000` | Request timeout in milliseconds |

## Request format (OpenAI-compatible)

The `/v1/chat/completions` endpoint accepts the standard OpenAI schema:

```json
{
  "model": "model-name",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": true
}
```

Anthropic (`/v1/messages`) and Ollama (`/api/chat`, `/api/generate`) use their own schemas; see the [official docs](https://docs.anthropic.com/en/api/messages) and [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) respectively.

## Streaming

When `stream: true`, the server returns SSE events compatible with the OpenAI streaming format:

```
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: [DONE]
```

## Chat templates

The server supports Jinja2 chat templates embedded in GGUF metadata (`-jinja` flag). Templates are automatically selected based on the model’s `tokenizer_chat_template` field.

## Authentication

If `--server-auth` is provided, the server expects a Bearer token:

```
Authorization: Bearer <api-key>
```

Requests without a valid key receive `401 Unauthorized`.

## Batching

The server batches incoming requests to maximize GPU utilization. Requests are grouped by similar sequence length and processed in parallel. The `--server-max-queue` flag controls backpressure.

## Environment variables

| Variable | Description |
|---|---|
| `NOGPU_PREFILL=1` | Force CPU prefill |
| `NOQ4=1` | Disable Q4 GEMM |
| `NOQ4SSM=1` | Disable Q4 in SSM |
| `NOQ4ATTN=1` | Disable Q4 in attention |
| `NOQ4FFN=1` | Disable Q4 in FFN |

## Examples

### curl (OpenAI)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
```

### Python (OpenAI client)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
resp = client.chat.completions.create(
  model="model",
  messages=[{"role": "user", "content": "Hello"}],
  max_tokens=64,
)
print(resp.choices[0].message.content)
```

## See also

- [`../README.md`](../README.md) — build, CLI, flags
- [`../ROADMAP.md`](../ROADMAP.md) — roadmap
- [`../CHANGELOG.md`](../CHANGELOG.md) — changelog
