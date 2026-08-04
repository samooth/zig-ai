# PagedAttention para zig-ai

Implementación de PagedAttention (vLLM, Kwon et al. 2023) adaptada al motor de inferencia zig-ai.

## Características

- **Bloques fijos**: KV cache dividido en bloques de `block_size` tokens
- **Memoria no contigua**: Los bloques se asignan bajo demanda desde un pool
- **Copy-on-Write**: Compartición eficiente para beam search y parallel sampling
- **Prefix Caching**: Reutilización de bloques KV entre requests con prefijos comunes
- **CPU Offloading**: Swap de bloques preempted a memoria del host
- **Referencia CPU**: Kernel de atención paginada con online softmax
- **Stubs CUDA**: Kernels GPU listos para compilación con nvcc

## Estructura

```
src/paged_attention/
├── root.zig              # Exports y configuración
├── block.zig             # Bloque físico con refcount
├── allocator.zig         # Pool allocator con COW y swap
├── block_table.zig       # Mapeo lógico->físico por secuencia
├── paged_kv_cache.zig    # Gestor KV cache paginado
├── attention.zig         # Kernel CPU reference (online softmax)
├── scheduler.zig         # Batch scheduler con preemption
├── prefix_cache.zig      # Deduplicación por hash de bloques
└── gpu_kernels.zig       # Stubs CUDA

cuda/
└── paged_attention.cu    # Kernels CUDA (decode, reshape, copy)

tests/
└── test_paged_attention.zig
```

## Integración en build.zig

```zig
const paged_attention = b.addModule("paged_attention", .{
    .root_source_file = b.path("src/paged_attention/root.zig"),
});
exe.root_module.addImport("paged_attention", paged_attention);
```

## Uso básico

```zig
const pa = @import("paged_attention");

var config = pa.PagedConfig{
    .block_size = 16,
    .num_blocks = 4096,
    .head_dim = 128,
    .num_kv_heads = 8,
    .num_q_heads = 32,
    .dtype = .f16,
};

var kv = try pa.PagedKVCache.init(allocator, config);
defer kv.deinit();

var sched = pa.Scheduler.init(allocator, config, &kv);
defer sched.deinit();

// Enviar request
const req_id = try sched.submit(.{
    .prompt_tokens = tokens,
    .max_new_tokens = 128,
    .num_samples = 1,
});

// Cada paso de scheduling
const batch = try sched.schedule();
// ... ejecutar forward en batch ...
try sched.appendToken(seq_id, new_token);
```

## Tests

```bash
zig build test
```

## Referencias

- Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
- vLLM: https://github.com/vllm-project/vllm
