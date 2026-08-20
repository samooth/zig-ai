# Estado del proyecto — CUDA Graphs decode + LFM2.5 hybrid
 
Ultima actualizacion: 2026-08-20.

## Objetivo
Terminar la optimizacion de decode con CUDA Graphs (paridad Debug ya verificada),
arreglando dos bugs solo visibles en ReleaseFast, y luego limpiar instrumentacion,
verificar paridad, correr tests y commitear.

## Bugs resueltos

### BUG A — UAF del block table en prefill (FIXED)
- **Sintoma:** `illegal memory access (0x2bc)` en el kernel de prefill, solo en ReleaseFast.
- **Causa raiz:** use-after-free en `hybrid_attn.zig` `forwardGPU`. El
  `defer self.allocator.free(bt_host)` se ejecutaba al final del bloque `if (n > 1)`,
  pero `prefillDevice` (dentro del mismo if) todavia leia `bt_host` -> memoria
  liberada -> `d_bt` subia basura (puntero de freelist) -> el kernel hacia accesos
  fuera de rango.
- **Fix:** mover `defer if (bt_host.len > 0) self.allocator.free(bt_host);` a ambito
  de funcion (despues del bloque `if (n > 1)`, antes de `kvAppendF16`), en
  `src/transformer/hybrid_attn.zig` (~linea 704).
- **Verificado:** ReleaseFast `NOGRAPH=1` genera "The capital of France is Paris."
  (EXIT=0), 1-token y 5-token.

### BUG B — Segfault host en cuGraphInstantiate (FIXED)
- **Sintoma:** SIGSEGV dentro de libcuda en `cuGraphInstantiate` (write a `(%rbx)`
  con rbx invalido), ReleaseFast-only, tanto 1-token como 5-token.
- **Investigacion:** independiente de BUG A. El grafo capturado es VALIDO (dumps de
  nodos verificados; replay produce salida correcta). El crash era dependiente del
  layout del binario (probes tipo `cuGraphGetNodes` antes de instantiate a veces lo
  enmascaraban). No lo arreglaban `cuCtxSynchronize` ni params persistentes de
  decodeDevice (`g_decode_persistent`).
- **Fix:** usar la API nueva `cuGraphInstantiateWithParams` en vez de
  `cuGraphInstantiate`/`WithFlags`; toma una ruta interna distinta en el driver y
  ademas reporta `hErrNode_out`/`result_out`. Binding en `src/cuda/cudaz_stub.zig`;
  call en `src/cuda/decode_graph.zig` (`endCaptureAndInstantiate`).
- **Verificado:** 5/5 runs EXIT=0 en ambos paths (1-token y 5-token).

## LFM2.5 Hybrid Architecture (COMPLETED 2026-08-20)
 
Soporte completo para la arquitectura LFM2.5 (Liquid Foundation Model 2.5) en formato GGUF:
 
### Modelo probado
- **LFM2.5-2.6B-Q4_K_M.gguf** (266 tensores, 30 capas, 2048 dim, 32 heads, 8 kv_heads)
- Metadata clave: `general.architecture = "lfm2"`, `lfm2.attention.head_count_kv` = array[30] de int32 (capas 2,5,9,13,17,21,24,27 tienen kv=8, resto 0)
- `lfm2.shortconv.l_cache = 3` (kernel size 4)
- Tokenizer: GPT-2 BPE (`tokenizer.ggml.pre = "lfm2"`), 128k vocab
 
### Implementación
- **ModelConfig** (`model_config.zig`): detecta `lfm2`, parsea array `head_count_kv` (30 int32), construye `per_layer_attn: []bool` y `shortconv_l_cache`, setea `is_hybrid=true`
- **ShortConvLayer** (`short_conv.zig` new): depthwise conv1d (kernel=4) + silu → in_proj (3x expand) → split(gate,up,value) → silu(gate)*up+value → out_proj; CPU forward completo, GPU path con `conv1dSiluKernel` + GEMM host scratch
- **HybridAttnParams** (`hybrid_attn.zig`): `no_gate=true` (Q separado, no Q+G fusionado), `use_mrope=false` (RoPE estándar NEOX)
- **HybridLayer** (`hybrid_layer.zig`): dispatch por capa usando `per_layer_attn`, `attn_post_norm` opcional (usa `ffn_norm` para LFM2), integra `ShortConvLayer` en init/load/forward/unload/deinit/warmupGpuWeights/forwardGPU
- **RoPE genérico** (`rope.zig`): `applyRoPE(comptime T: type, ...)` soporta f32 para LFM2
- **Build** (`build.zig`): módulo `short_conv` conectado a `hybrid_layer`
 
### Validación
- ✅ Build exitoso con Zig 0.16.0 (`zig build install --cache-dir /tmp/opencode/zig-cache`)
- ✅ Todos los tests pasan (`zig build test`)
- ✅ Carga modelo LFM2.5-2.6B-Q4_K_M.gguf en CPU y GPU
- ⚠️ GPU: OOM en 7.7GB VRAM (RTX 3080 Laptop) — necesita 12GB+ para model caching completo
- ⚠️ CPU: prefill lento (~2 min para 2.6B params) pero funcional
 
### Archivos afectados
- `src/loader/model_config.zig` (+70 líneas)
- `src/loader/gguf.zig` (+10 líneas)
- `src/transformer/short_conv.zig` (nuevo, ~500 líneas)
- `src/transformer/hybrid_attn.zig` (+52 líneas)
- `src/transformer/hybrid_layer.zig` (+247 líneas)
- `src/transformer/rope.zig` (+27 líneas)
- `build.zig` (+37 líneas)
 
Commit: `93d6fad` (firmado)
- **Sintoma:** la generacion con grafo diverge de `NOGRAPH=1` desde el token 0.
  Logits del token 0 difieren (max diff ~4.4), deterministico.
- **Diagnostico hasta ahora:**
  - El restore del estado recurrente SSM (`d_s_state`, `d_conv_state`) es CORRECTO
    (checksums identicos antes/despues de captura; `CHKSTATE=1`).
  - **HALLAZGO CLAVE:** el grafo NUNCA escribe el KV cache. Con `DUMPKV=1`, las
    posiciones 5,6,7,8 del bloque 0 de la capa de atencion quedan en CERO en modo
    grafo (en NOGRAPH se escriben correctamente). Los checksums de pos 0,1,2,4
    coinciden (prefill intacto).
  - El bloque se lee desde el pool HOST tras `syncDecodeBlocks`. Falta determinar si
    el write en device es cero o si el D2H de `syncDecodeBlocks` no copia en modo
    grafo (leer el KV en device directamente via `cacheBase` + `phys*block_bytes`).
    Aun asi la generacion produce "Paris." correcto, lo que sugiere que el decode
    lee KV real del device (el pool host solo muestra zeros), apuntando a que
    `syncDecodeBlocks` falla en modo grafo.
- **Siguiente paso:** leer el KV del device directamente (cuMemcpyDtoH desde
  vaddr+phys*block_bytes) y comparar grafo vs NOGRAPH; si el device KV es correcto,
  el bug esta en `syncBlockToHost` (D2H) tras el `cuGraphLaunch`.

## Instrumentacion activa (env-gated, NO limpiar)
- `DUMPKV=1` — dump del bloque KV (pool host) por token en main.zig.
- `DUMPNORM=1` — dump del g_normed en el primer token.
- `CHKSTATE=1` — checksum del estado SSM antes/despues de capturar.
- `DUMP_LOGITS=1` — dump de logits por token (decode loop).
- `PERF_STAGE=1` — metricas por capa.

## Build / run / test
- Build RF: `zig build install -Doptimize=ReleaseFast --cache-dir /tmp/ziglocal2 --global-cache-dir /tmp/zigglobal2`
- Run: `CUDA_VISIBLE_DEVICES=0 ./zig-out/bin/zig-ai-engine -m /opt/models/Qwen3.5-0.8B-Q4_0.gguf --prompt "The capital of France is" -n 12 --temperature 0`
- `NOGRAPH=1` desactiva el grafo; `CUDA_LAUNCH_BLOCKING=1` sincroniza lanzamientos.
- Tests: `GGUF_MODEL_PATH=/opt/models/Qwen3.5-0.8B-Q4_0.gguf zig build test -Doptimize=ReleaseFast --cache-dir /tmp/ziglocal2 --global-cache-dir /tmp/zigglobal2`
- Nota: si el load falla con `error: CudaError`, es por memoria GPU ocupada
  (p.ej. `ollama serve`); liberar la GPU y reintentar.

## Archivos relevantes
- `src/transformer/hybrid_attn.zig` — fix BUG A; `stageDecodeHost`, `syncDecodeBlocks`.
- `src/cuda/decode_graph.zig` — captura/instantiate/replay (fix BUG B).
- `src/cuda/cudaz_stub.zig` — bindings; `cuGraphInstantiateWithParams`; `pinnedAlloc`.
- `src/paged_attention/gpu_kernels.zig` — staging persistente, `g_decode_persistent`.
- `src/main.zig` — graph capture, decode loop, dumps debug (DUMPKV/DUMPNORM/CHKSTATE).
- `src/cuda/layer_kernels.{zig,cu}` — start_pos como puntero device (para grafo).
- `/tmp/opencode/*.txt` — outputs de las corridas de diagnostico.
