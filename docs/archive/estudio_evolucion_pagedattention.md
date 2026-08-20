# Estudio: Evolución y Mejora de PagedAttention

> **Fecha:** 2026-08-04
> **Autor:** Investigación sintetizada a partir de literatura 2024-2026

---

## 1. Resumen Ejecutivo

PagedAttention (SOSP 2023) resolvió el problema de fragmentación de memoria en el KV cache de LLMs, elevando el uso efectivo de memoria del 20-38% al 96.3%. Sin embargo, el ecosistema ha evolucionado en tres años. Este estudio identifica **seis ejes de mejora** concretos, cuantificados y verificables, que transforman PagedAttention de un gestor de memoria paginado en un sistema de gestión de estado de inferencia de proxima generacion.

| Eje | Mejora potencial | Estado técnico |
|-----|-----------------|----------------|
| **Demand paging nativo** (vAttention) | +1.23x throughput, portabilidad de kernels | ASPLOS 2025, produccion en Microsoft |
| **Compresion arquitectonica** (MLA) | -93.3% tamano KV cache | DeepSeek V2/V3, adaptacion a Llama via MHA2MLA |
| **Cuantizacion con rotacion** (SAW-INT4) | -4x bytes/token, <1% perdida de calidad | arXiv 2026, kernels fusionados sin overhead |
| **Separacion prefill/decode** | +variasx throughput bajo latencia fija | DistServe OSDI 2024, Mooncake, NVIDIA Dynamo |
| **Eviccion aprendida** | -50% tokens cacheados con mejora de calidad | ACL 2025, EMNLP 2025 |
| **Co-diseno hardware** (FA3/FA4, FP8) | +1.5-2x (FA3), +4-6x (FP8) | Hopper/Blackwell nativo |

---

## 2. Diagnostico: Limitaciones de PagedAttention Clasico

El paper original de Kwon et al. dejo cinco deudas tecnicas:

1. **Kernels custom no portables.** Cada backend (CUDA, ROCm, TPU) requiere reimplementar el kernel de atencion paginada. El kernel de vLLM v0 ya es hasta **2.8x mas lento** que FlashAttention-2 no paginado.
2. **Redundancia de gestion de memoria.** PagedAttention duplica en user-space lo que el OS y el driver CUDA ya hacen: traduccion virtual a fisica.
3. **Fragmentacion interna residual.** Hasta `block_size - 1` tokens desperdiciados por secuencia en el ultimo bloque.
4. **Overhead de indireccion.** La block table anade latencia de lookup en cada iteracion de decode.
5. **Ausencia de compresion semantica.** PagedAttention gestiona bloques, no contenido. No eviccion, no cuantizacion, no deduplicacion de prefijos entre peticiones distintas mas alla del hash exacto.

---

## 3. Ejes de Evolucion

### 3.1 Alternativa Arquitectonica: vAttention (Demand Paging Nativo)

**Autores:** Prabhu et al., ASPLOS 2025

**Idea central:** En lugar de implementar demand paging en user-space (bloques no contiguos en memoria virtual), usar las **CUDA Virtual Memory Management APIs** para separar asignacion virtual y fisica.

**Mecanismo:**
- Reserva un buffer virtual contiguo enorme al inicio (`2 x N` tensores, donde N = capas).
- La memoria fisica se asigna pagina a pagina via `cuMemMap` + `cuMemSetAccess`.
- El KV cache permanece **contiguo en memoria virtual** -- los kernels de atencion estandar (FlashAttention-2, FlashInfer) funcionan sin modificacion.

**Ventajas cuantificadas:**
- **Portabilidad:** Soporta cualquier kernel de atencion "out-of-the-box".
- **Rendimiento:** Hasta **1.23x mas throughput** que PagedAttention con kernels de FlashAttention-2/FlashInfer.
- **Prefill:** Especialmente superior en cargas prefill-bound porque evita el overhead de indireccion de block tables en secuencias largas.

**Limitaciones:**
- Requiere CUDA 12.1+ y APIs de bajo nivel.
- La latencia de `cuMemMap` (~40us por llamada) requiere ocultarse via background threading, lo que vAttention hace solapando asignacion con compute.
- No soporta nativamente comparticion de bloques (COW) entre secuencias; requiere extension del driver NVIDIA.

**Veredicto:** vAttention es el reemplazo directo mas limpio para PagedAttention en entornos CUDA modernos. Elimina la deuda tecnica de kernels custom.

---

### 3.2 Compresion Arquitectonica: Multi-Head Latent Attention (MLA)

**Autores:** DeepSeek (DeepSeek-V2, 2024); adaptacion MHA2MLA por Ji et al., ACL 2025

**Idea central:** En lugar de reducir el numero de heads (GQA/MQA), **comprimir lo que se almacena** en el KV cache via proyeccion de rango reducido.

**Mecanismo:**
- Se proyecta el estado oculto `h_t` a un vector latente `c_KV` de dimension `d_c << 2 x n_h x d_h`.
- Solo `c_KV` se cachea.
- En atencion, se up-projecta `c_KV` a `k`, `v` via matrices `W_UK`, `W_UV`.
- Con RoPE desacoplado (partial-RoPE), las dimensiones rotadas se manejan separadamente.

**Ahorro cuantificado:**
- DeepSeek-V2: **-93.3%** tamano de KV cache vs MHA denso equivalente.
- DeepSeek-V3/R1: **reduccion 56.9x** vs MHA estandar (de ~4MB/token a ~70KB/token).
- MHA2MLA adapta Llama2-7B: **-92.19%** KV cache con solo 1% drop en LongBench.

**Trade-offs:**
- **Complejidad de serving:** Requiere reconstruccion on-the-fly de K/V. Los kernels de atencion estandar no soportan MLA directamente (aunque FA3 tiene parches).
- **Perdida de calidad:** En modelos no entrenados con MLA, requiere fine-tuning (MHA2MLA necesita solo 0.6-1% de datos de entrenamiento originales).
- **No es ortogonal a PagedAttention:** Se puede paginar el `c_KV` latente en bloques, reduciendo aun mas la memoria.

**Veredicto:** MLA es la mejora arquitectonica mas agresiva. Para nuevos modelos, deberia ser el default. Para modelos existentes, MHA2MLA permite migracion a coste razonable.

---

### 3.3 Compresion Numerica: Cuantizacion con Rotacion (SAW-INT4)

**Autores:** Jia et al., arXiv:2604.19157v1, 2026

**Idea central:** La cuantizacion naive INT4 del KV cache colapsa la calidad del modelo a casi cero en modelos sensibles (Qwen3). La solucion es **rotacion ortogonal (Hadamard) antes de cuantizar**.

**Mecanismo:**
- Token-wise INT4 asimetrico (por token y por head).
- Pre-rotacion con **Block-Diagonal Hadamard Rotation (BDR)** para suavizar outliers channel-wise.
- Kernel fusionado: rotacion + cuantizacion + escritura a KV cache paginado en un solo paso CUDA.
- Dequantizacion fusionada dentro del kernel de atencion (compatible con FlashAttention-style decoding).

**Resultados:**
- **Qwen3-8B:** BDR-128 (K only) recupera 73.78 vs 75.64 BF16 (drop de solo 1.86 puntos), mientras que INT4 naive da **0.00** en todas las metricas.
- **Zero serving overhead:** El kernel fusionado introduce overhead indetectable en end-to-end.
- **4x reduccion** de memoria KV cache.

**Constraints tecnicos clave:**
- Debe ser **token-wise** (no channel-wise) para ser compatible con layouts paginados no contiguos.
- Debe evitar codebook lookups (VQ) o transforms SVD que rompen coalesced access en GPU.
- La rotacion solo en K (no V) es suficiente, simplificando el kernel.

**Veredicto:** SAW-INT4 es el metodo de cuantizacion de KV cache mas practico para produccion en 2026. Es ortogonal a PagedAttention/vAttention y a MLA.

---

### 3.4 Scheduling Avanzado: Separacion Prefill/Decode y Chunked Prefill

**Autores:** DistServe (OSDI 2024); Mooncake (Moonshot AI); NVIDIA Dynamo; vLLM V1

**Idea central:** Prefill y decode tienen perfiles de compute/memory diametralmente opuestos. Compartir GPU los degrada mutuamente.

| Fase | Bottleneck | Latencia sensible |
|------|-----------|-------------------|
| **Prefill** | Compute-bound (matmul grande) | TTFT (Time To First Token) |
| **Decode** | Memory-bound (matvec, KV cache) | ITL (Inter-Token Latency) |

**Mecanismos:**

**a) Disaggregated Serving:**
- Clusters separados de GPU: unos hacen prefill, otros decode.
- Transferencia de KV cache entre clusters via librerias dedicadas (NIXL en NVIDIA Dynamo).
- DeepSeek V3/R1 usa esto en produccion.
- DistServe reporta **variasx mas requests** dentro de los mismos targets de latencia.

**b) Chunked Prefill (vLLM V1):**
- El prefill de una peticion larga se divide en chunks que se intercalan con iteraciones de decode.
- Elimina la "head-of-line blocking" donde un prefill largo bloquea todo un batch.
- En vLLM V1 es **obligatorio y no desactivable**.
- El scheduler V1 representa decisiones como `{request_id: num_tokens}`, unificando prefill y decode.

**Veredicto:** La separacion prefill/decode es inevitable a escala. Chunked prefill es la solucion intermedia para single-node. PagedAttention debe integrarse con scheduling que entienda estas fases, no solo asignar bloques.

---

### 3.5 Eviccion Inteligente del KV Cache

**Autores:** H2O (Zhang et al., 2023); StreamingLLM (Xiao et al., 2024); Attention-Gate (ACL 2025); KVP (arXiv 2026); LookAhead Q-Cache (EMNLP 2025).

**Idea central:** No todos los tokens del KV cache son igualmente importantes. La atencion es inherentemente dispersa.

**Metodos:**

| Metodo | Estrategia | Eviccion | Calidad |
|--------|-----------|----------|---------|
| **StreamingLLM** | Sinks iniciales + ventana reciente | 50% | Buena para streaming, mala para retrieval |
| **H2O** | Heavy hitters (acumulacion de attention scores) + recientes | 50% | Mejor, pero sesgo de atencion |
| **SnapKV** | Ventana de observacion local para estimar importancia | Variable | Fuerte en long context |
| **Attention-Gate** | Modulo aprendido (continual pre-training) por head/layer | >50% | **Supera al baseline sin eviccion** en algunas tareas |
| **KVP** | Policy aprendida query-independent, forward-looking | Variable | Data-driven, sin overhead de atencion en decode |

**Sintesis practica:**
- **Sin entrenamiento:** H2O + StreamingLLM es el estandar. Combinar: sinks (4 tokens) + heavy hitters (64) + recientes (64) = 132 tokens cacheados vs miles.
- **Con fine-tuning:** Attention-Gate o KVP permiten evicciones agresivas (50-70%) **mejorando** la calidad al eliminar ruido.
- **Con PagedAttention:** La eviccion es problematica porque un bloque paginado solo se libera si **todo** el bloque esta evicted. Si evicciones dejan huecos dentro de un bloque, la fragmentacion interna resurge.

**Veredicto:** La eviccion debe operar a **granularidad de bloque** (alineada al block size de PagedAttention) o requerir reordenacion de tokens en bloques. Es el eje menos maduro para integrar con paginacion.

---

### 3.6 Co-diseno Hardware-Kernel: FlashAttention 3/4 y FP8

**Autores:** Dao et al.; FlashAttention-3 (2024), FlashAttention-4 (2025).

**Idea central:** PagedAttention gestiona memoria, pero la atencion en si debe ser lo mas rapida posible. FA3/FA4 son ortogonales pero criticos.

**FlashAttention 3 (Hopper):**
- **Warp specialization:** Productores (fetch async via TMA/cp.async) vs consumidores (GEMM + softmax).
- **Ping-pong scheduling:** Doble buffer en SRAM, sin burbujas.
- **FP8 nativo:** Q@K^T y softmax@V en FP8 con acumulacion BF16.
- **Speedup:** 1.5-1.75x vs FA2 en H100; **4-6x** en FP8 vs FA2 BF16.

**FlashAttention 4 (Blackwell):**
- TMA tile execution, SM100 native.
- Target: B200/B300.

**Implicaciones para PagedAttention:**
- PagedAttention requiere kernels custom que **no** se benefician automaticamente de FA3/FA4.
- vAttention, al mantener contiguidad virtual, puede usar FA3/FA4 directamente.
- La combinacion **vAttention + FA3 FP8** es el stack de maximo rendimiento en Hopper.

---

## 4. Propuesta de Arquitectura Integrada: PagedAttention-NG

Para un motor de inferencia de proxima generacion, se propone la siguiente arquitectura hibrida:

```
+------------------------------------------------------------------+
|                      SCHEDULER UNIFICADO                           |
|  {req_id: num_tokens} -- chunked prefill + decode interleaved    |
|  Fase-aware: prefill workers vs decode workers (disaggregated)   |
+--------------------------+-----------------------------------------+
                           |
          +----------------+----------------+
          v                                 v
+---------------------+          +---------------------+
|   PREFILL CLUSTER   |          |   DECODE CLUSTER    |
|  (compute-bound)    |          |  (memory-bound)     |
|  - FA3/FA4 kernels  |          |  - FA3/FA4 kernels  |
|  - vAttention alloc |<-------->|  - vAttention alloc |
|  - Prefix cache     |   NIXL   |  - Prefix cache     |
|  - MLA compression  |   KV xfer|  - MLA compression  |
+---------------------+          |  - INT4 BDR quant   |
                                 |  - COW block sharing|
                                 +---------------------+
```

### Componentes clave:

1. **Memory backend: vAttention** en lugar de PagedAttention user-space.
   - Usa CUDA VMM APIs para demand paging nativo.
   - Mantiene contiguidad virtual -> kernels estandar sin modificacion.
   - Bloques de tamano configurable (recomendado: 256 para prefill, 16 para decode).

2. **Compresion dual: MLA + INT4-BDR.**
   - Si el modelo soporta MLA: cachear `c_KV` latente (dim ~576) en lugar de K/V completos.
   - Aplicar BDR-128 + INT4 token-wise sobre `c_KV`.
   - Reduccion combinada: **~200x** vs MHA FP16 estandar.

3. **Prefix caching con hash de bloques.**
   - vLLM V1 ya implementa "zero-overhead prefix caching".
   - Extender con **fuzzy matching** (no solo exacto) via embeddings locales de prefijos.

4. **Eviccion por bloques.**
   - En lugar de evictar tokens individuales (que fragmenta bloques), evictar **bloques enteros** basandose en scores de atencion acumulados por bloque.
   - Politica: LRU a nivel de bloque para bloques no compartidos; bloques compartidos (ref_count > 1) nunca se evictan.

5. **Transferencia KV entre clusters.**
   - Para disaggregated serving: serializar bloques MLA-cuantizados y transferir via RDMA/NVLink.
   - El formato de transferencia debe ser el mismo que el de almacenamiento: comprimido y cuantizado.

---

## 5. Hoja de Ruta de Implementacion

### Fase 1: Fundacion (1-2 meses)
- [ ] Implementar **vAttention** como backend de memoria en el runtime.
- [ ] Integrar **FlashAttention-3** (o FA2 si no hay Hopper) sin modificar kernels.
- [ ] Validar con benchmarks: throughput prefill/decode vs vLLM baseline.

### Fase 2: Compresion (2-3 meses)
- [ ] Implementar **token-wise INT4** con BDR-128 en CUDA kernels.
- [ ] Fusionar rotacion + cuantizacion + escritura a KV cache.
- [ ] Medir overhead: debe ser <1% en end-to-end.

### Fase 3: Arquitectura (3-4 meses)
- [ ] Soporte para **MLA** si se usan modelos DeepSeek o se migra Llama via MHA2MLA.
- [ ] Implementar **chunked prefill** en el scheduler.
- [ ] Anadir **prefix caching** con hash exacto.

### Fase 4: Escala (2-3 meses)
- [ ] **Disaggregated serving:** separar prefill/decode workers.
- [ ] Pipeline de transferencia de KV cache comprimido entre workers.
- [ ] **Eviccion por bloques** basada en atencion acumulada.

---

## 6. Conclusiones

PagedAttention fue un avance foundational, pero sus limitaciones (kernels custom, overhead de indireccion, ausencia de compresion) lo hacen suboptimo para 2026. La evolucion pasa por:

1. **Reemplazar el gestor de memoria** por vAttention (demand paging nativo del driver).
2. **Reducir lo que se cachea** via MLA (compresion arquitectonica) e INT4-BDR (compresion numerica).
3. **Separar las fases** de compute y memory bound (disaggregated serving).
4. **Ser inteligente sobre que mantener** (eviccion por bloques, prefix caching).

La combinacion de estos cuatro ejes puede reducir el footprint de memoria por token de **~800 KB (OPT-13B FP16)** a **~200 bytes** (MLA + INT4), un factor de **4000x**, mientras se mantiene o mejora el throughput gracias a kernels nativos FA3/FA4 y scheduling por fases.

Para un stack custom, la recomendacion es: **no reimplementar PagedAttention**. Usar CUDA VMM para gestion de memoria, implementar cuantizacion INT4 con Hadamard, y enfocarse en el scheduler unificado prefill/decode. Los kernels de atencion dejarlos a FlashAttention (via bindings a la libreria C++ de Dao).

---

## Referencias

- Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP '23. arXiv:2309.06180.
- Prabhu, A., et al. (2025). *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention*. ASPLOS 2025.
- DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*.
- Ji, Z., et al. (2025). *Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs*. ACL 2025.
- Jia, et al. (2026). *SAW-INT4: System-AWare 4-Bit KV-Cache Quantization for Real-World LLM Serving*. arXiv:2604.19157v1.
- Zhong, Y., et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*. OSDI 2024.
- Dao, T., et al. (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*.
- Xiao, G., et al. (2024). *Efficient Large Language Models: A Survey*. arXiv:2312.03863.
- Zhang, Z., et al. (2023). *H2O: Heavy-Hitter Oracle for Accurate and Efficient KV Cache Compression*.
- vLLM Documentation: https://docs.vllm.ai/
