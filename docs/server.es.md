# Modo servidor — API HTTP

zig-ai-engine incluye un servidor HTTP incorporado compatible con las APIs de OpenAI, Anthropic y Ollama. Los tres conjuntos de endpoints se exponen simultáneamente en la misma instancia del servidor. No existe un flag para seleccionar una API sobre otra — el cliente elige qué endpoint llamar.

## Inicio rápido

```bash
# Iniciar el servidor con un modelo GGUF
./zig-out/bin/zig-ai-engine \
  --model modelo.gguf \
  --server \
  --server-port 8080 \
  --server-host 0.0.0.0
```

## APIs soportadas

| API | Endpoint | Notas |
|---|---|---|
| OpenAI | `/v1/chat/completions` | `stream: true/false`, function calling no soportado aún |
| Anthropic | `/v1/messages` | Streaming SSE, system prompt, multi-turn |
| Ollama | `/api/generate`, `/api/chat` | Listado de modelos no soportado aún |

## Flags

| Flag | Default | Descripción |
|---|---|---|
| `--server` | off | Activa modo servidor HTTP |
| `--server-host <addr>` | `127.0.0.1` | Dirección de bind |
| `--server-port <n>` | `8080` | Puerto de bind |
| `--server-auth <keys>` | — | API keys separadas por coma (auth Bearer token) |
| `--server-max-queue <n>` | `256` | Requests máximas en cola |
| `--server-timeout <ms>` | `60000` | Timeout de request en milisegundos |

## Formato de request (compatible OpenAI)

El endpoint `/v1/chat/completions` acepta el schema estándar de OpenAI:

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

Anthropic (`/v1/messages`) y Ollama (`/api/chat`, `/api/generate`) usan sus propios schemas; ver la [documentación oficial de Anthropic](https://docs.anthropic.com/en/api/messages) y la [API de Ollama](https://github.com/ollama/ollama/blob/main/docs/api.md) respectivamente.

## Streaming

Cuando `stream: true`, el servidor devuelve eventos SSE compatibles con el formato de streaming de OpenAI:

```
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: [DONE]
```

## Chat templates

El servidor soporta chat templates Jinja2 embebidos en la metadata de GGUF (flag `-jinja`). Los templates se seleccionan automáticamente según el campo `tokenizer_chat_template` del modelo.

## Autenticación

Si se provee `--server-auth`, el servidor espera un Bearer token:

```
Authorization: Bearer <api-key>
```

Requests sin una key válida reciben `401 Unauthorized`.

## Batching

El servidor agrupa requests entrantes para maximizar la utilización de GPU. Los requests se agrupan por longitud de secuencia similar y se procesan en paralelo. El flag `--server-max-queue` controla backpressure.

## Variables de entorno

| Variable | Descripción |
|---|---|
| `NOGPU_PREFILL=1` | Forzar prefill CPU |
| `NOQ4=1` | Desactivar Q4 GEMM |
| `NOQ4SSM=1` | Desactivar Q4 en SSM |
| `NOQ4ATTN=1` | Desactivar Q4 en attention |
| `NOQ4FFN=1` | Desactivar Q4 en FFN |

## Ejemplos

### curl (OpenAI)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
```

### Python (cliente OpenAI)

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

## Ver también

- [`../README.md`](../README.md) — build, CLI, flags
- [`../ROADMAP.md`](../ROADMAP.md) — roadmap
- [`../CHANGELOG.md`](../CHANGELOG.md) — changelog
