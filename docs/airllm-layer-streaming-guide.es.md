# Guía de usuario: AirLLM Layer Streaming

## Visión general

**AirLLM Layer Streaming** permite ejecutar modelos grandes que no caben en VRAM
mediante streaming de pesos capa a capa desde la RAM del host a la GPU. El motor
mantiene solo `N` capas residentes en VRAM (configurable vía `--layer-stream-max`),
cargando la siguiente capa de forma asíncrona mientras la GPU calcula la capa actual,
y expulsando la capa menos recientemente usada cuando se excede el presupuesto.

Esto permite ejecutar modelos mayores que la VRAM (p. ej. Qwen3.5-0.8B Q4_0 en una GPU de 8GB).

## Inicio rápido

```bash
# Activar layer streaming con máximo 2 capas residentes
./zig-out/bin/zig-ai-engine \
  --model /ruta/al/modelo.gguf \
  --prompt "Tu prompt aquí" \
  --max-tokens 256 \
  --layer-stream \
  --layer-stream-max 2
```

## Flags

| Flag | Default | Descripción |
|---|---|---|
| `--layer-stream` | off | Activa layer streaming |
| `--layer-stream-max <n>` | 2 | Máximo de capas residentes simultáneamente en VRAM |

## Presupuesto de VRAM

El motor calcula el presupuesto de VRAM automáticamente basándose en la GPU detectada.
El presupuesto se divide en: 60% pesos residentes, 20% KV cache, 20% activaciones, 5% overhead.

## Tuning

- Aumentar `--layer-stream-max` mejora throughput a costa de VRAM
- Reducirlo permite modelos más grandes a costa de velocidad
- El valor óptimo depende del tamaño del modelo y la VRAM disponible

## Troubleshooting

- **OOM**: reducir `--layer-stream-max` o usar un modelo más pequeño
- **Lento**: aumentar `--layer-stream-max` si hay VRAM disponible
- **CUDA graphs**: el warm-up de caches antes de capture es automático

## Ver también

- [`../README.md`](../README.md) — build, CLI, estructura del repo
- [`../ROADMAP.md`](../ROADMAP.md) — roadmap público
- [`../CHANGELOG.md`](../CHANGELOG.md) — historial de cambios
- [This guide in English](airllm-layer-streaming-guide.md)
