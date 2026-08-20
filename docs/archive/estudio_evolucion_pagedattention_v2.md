# Estudio Complementario: Mejoras Adicionales a PagedAttention-NG

> **Fecha:** 2026-08-04
> **Nota:** Este documento complementa el estudio principal "Evolucion y Mejora de PagedAttention" con descubrimientos de 2025-2026 no cubiertos en el reporte anterior.

---

## 1. Resumen Ejecutivo

El estudio anterior identifico seis ejes de mejora (vAttention, MLA, SAW-INT4, disaggregated serving, eviccion inteligente, FA3/FP8). Este documento presenta **nueve mejoras adicionales** descubiertas en la literatura 2025-2026 que elevan aun mas el techo de rendimiento. La combinacion de todas las tecnicas descritas en ambos reportes puede reducir el footprint de memoria por token de ~800 KB a **~50 bytes** y acelerar el prefill de contextos largos hasta **27.78x**.

| Mejora | Tipo | Impacto Cuantificado | Fuente |
|--------|------|---------------------|--------|
| **FlashPrefill** | Prefill sparse dinamico | **27.78x** speedup en 256K tokens, 1.71x en 4K | arXiv 2026 (Tencent) |
| **TailorKV** | Offloading hibrido por capas | Llama-3.1-8B 128K en **RTX 3090**, 82ms/token | ACL 2025 Findings |
| **Harvest** | Peer-to-peer GPU caching | **5.65x** mas rapido que CPU offload, 1.5-2.0x throughput | arXiv 2026 |
| **HSA** | Atencion jerarquica sparse | **16M tokens** con >90% accuracy NIAH | arXiv 2025 (Ant Group) |
| **Smurfs** | Speculative decoding colectivo | **109x** throughput vs Lookahead, 110x vs Ouroboros | SC 2025 |
| **Hybrid Mamba-Transformer** | Arquitectura dual | **2x** throughput, memoria constante en capas SSM | IBM/Tencent/Microsoft 2025 |
| **MIRAGE** | Remapeo de parametros | Pesos a CPU -> espacio extra para KV cache | Jul 2025 |
| **RingX** | Sequence parallelism HPC | **3.4x** vs Ring Attention en 4096 GPUs | SC 2025 |
| **vLLM Offloading Connector** | Layout contiguo de bloques | **10x** throughput en transferencias KV | vLLM 0.12.0 |

---

## 2. FlashPrefill: Prefill Ultra-Rapido via Atencion Sparse Dinamica

**Autores:** Qihang Fan et al., Tencent/UCAS/CASIA, arXiv:2603.06199v1, marzo 2026.

### 2.1 El problema

El prefill es compute-bound pero la atencion densa es O(n^2). Para 256K tokens, el prefill puede tardar decenas de segundos. Los metodos sparse existentes (MInference, FlexPrefill, XAttention) sufren de:
- **Latencia de busqueda:** Descubrir que bloques son relevantes toma tiempo.
- **Sparsidad insuficiente:** No eliminan suficientes bloques irrelevantes.
- **Degradacion en contextos cortos:** Muchos metodos son mas lentos que atencion densa en 4K tokens.

### 2.2 La solucion

FlashPrefill introduce dos mecanismos:

**a) Fast block-searching:** Localiza simultaneamente tres patrones de sparsidad:
- **Vertical:** Tokens que atienden a posiciones especificas del pasado.
- **Slash:** Patrones diagonales (atencion a tokens recientes + algunos lejanos).
- **Block-sparse:** Bloques enteros de tokens irrelevantes.

**b) Dynamic thresholding:** En lugar de ordenar o acumular scores de atencion (costoso), usa un umbral basado en el maximo score por bloque. Esto elimina la cola larga de la distribucion de atencion sin overhead de sorting.

### 2.3 Resultados

| Modelo | Contexto | Speedup vs Dens | Accuracy Retenida |
|--------|----------|-----------------|-------------------|
| Llama-3.1-8B | 256K | **27.78x** | 46.32 vs 48.39 (full) |
| Llama-3.1-8B | 4K | **1.71x** | Sin degradacion |
| Qwen2.5-7B | 256K | Similar | 24.93 vs 23.87 (full) |

**Crucial:** A diferencia de competidores (FlashMoBA colapsa a 17.33 accuracy), FlashPrefill **mantiene calidad** mientras acelera mas de 27x.

### 2.4 Integracion con PagedAttention-NG

FlashPrefill es **complementario** a la arquitectura propuesta:
- En el **Prefill Cluster**, reemplazar la atencion densa por FlashPrefill para secuencias >32K.
- Los bloques identificados como irrelevantes por FlashPrefill **no se almacenan** en el KV cache, reduciendo aun mas la memoria.
- El thresholding dinamico puede alimentar la politica de eviccion por bloques del scheduler.

---

## 3. TailorKV: Framework Hibrido de Offloading por Capas

**Autores:** Dingyu Yao et al., ACL 2025 Findings.

### 3.1 Insight fundamental

No todas las capas de un Transformer son iguales:
- **Capas superficiales (shallow):** Atencion densa, necesitan informacion global. Son **quantization-friendly**.
- **Capas profundas:** Atencion dispersa, pocos tokens dominantes. Son **sparsity-friendly**.

Aplicar la misma estrategia de compresion a todas las capas es suboptimo.

### 3.2 Estrategia hibrida

TailorKV clasifica cada capa con una metrica de identificacion y aplica:

**Capas quantization-friendly:**
- Cuantizacion agresiva a **1-bit** (FP16 x INT1 GEMV).
- Permanece en GPU todo el tiempo.
- Mantiene informacion global sin perdida.

**Capas sparsity-friendly:**
- KV cache completo offloaded a **CPU DRAM** durante prefill.
- Durante decode: **retrieval asincrono** de solo los Top-K tokens dominantes.
- Doble buffering para overlap de PCIe con compute.

### 3.3 Resultados

- **Llama-3.1-8B con 128K contexto** en una sola **RTX 3090** (24GB VRAM).
- **82 ms por token** en decode.
- **53.7% reduccion** de peak GPU memory.
- Cuantiza 1-2 capas a 1-bit y carga solo 1-3% de tokens de las capas offloaded.
- **Casi lossless** en benchmarks de long context.

### 3.4 Integracion con PagedAttention-NG

- Reemplazar la politica de offloading uniforme por **TailorKV-aware layer scheduling**.
- Las capas quantization-friendly usan el bloque vAttention + INT4-BDR del stack base.
- Las capas sparsity-friendly usan CPU offloading + retrieval Top-K con prefetch predictivo.
- El scheduler debe ser **layer-aware**: no todas las capas necesitan bloques GPU simultaneamente.

---

## 4. Harvest: Caching Peer-to-Peer entre GPUs

**Autores:** arXiv:2602.00328v1, enero 2026.

### 4.1 El problema

Cuando el KV cache excede la HBM de una GPU, las opciones son:
1. **CPU offload:** PCIe es lento (16-32 GB/s) y esta en el critical path.
2. **Shardear el modelo:** Mas GPUs, mas costo, mas overhead de sincronizacion.
3. **Reducir batch size:** Menor throughput.

### 4.2 La solucion

**Harvest** trata la memoria HBM de GPUs peer (conectadas via NVLink) como un **cache tier oportunista**:

```
Tier 1: Compute GPU HBM (autoritativo)
Tier 2: Peer GPU HBM via NVLink (cache efimero, ~900 GB/s)
Tier 3: Host DRAM via PCIe (backup)
```

**Caracteristicas clave:**
- **Best-effort:** Si un peer necesita su memoria, los bloques cacheados se revocan sin penalizacion (son reconstructibles).
- **NVLink:** Hasta 5.65x mas rapido que CPU offload para KV cache.
- **Transparente:** Se integra con vLLM extendiendo la KV block table para rastrear residencia (local HBM / peer HBM / host DRAM).

### 4.3 Resultados

| Escenario | Mejora |
|-----------|--------|
| KV cache transfer latency | **5.65x** vs CPU offload |
| MoE expert transfer | **10x** vs CPU offload |
| End-to-end throughput (Qwen, Phi-3.5) | **1.5-2.0x** |

### 4.4 Integracion con PagedAttention-NG

- En un cluster multi-GPU, el **Decode Cluster** puede usar Harvest para:
  - Cachear bloques KV de peticiones inactivas en GPUs peer.
  - Liberar HBM local para nuevas peticiones sin swap a CPU.
- La block table de vAttention se extiende con una columna `peer_gpu_id`.
- Los bloques con `ref_count > 1` (compartidos) son candidatos ideales para peer caching.

---

## 5. HSA: Atencion Jerarquica Sparse para 16M Tokens

**Autores:** Hu et al., Ant Group/Westlake University, arXiv:2511.23319, noviembre 2025.

### 5.1 Tres propiedades requeridas

Para "maquinas que pueden recordar" (ultra-long context):
1. **Sparsidad:** No O(n^2).
2. **Random-access:** Poder recuperar informacion de cualquier parte de la secuencia.
3. **Length generalization:** Entrenado en 32K, funcionar en 16M.

### 5.2 Mecanismo

HSA funciona como un **Mixture of Experts (MoE) de chunks**:

```
1. La secuencia se divide en chunks.
2. Cada chunk produce una "landmark key" (representacion compacta).
3. El token actual computa retrieval scores contra todas las landmark keys.
4. Se seleccionan los top-k chunks mas relevantes.
5. Se ejecuta atencion densa SOLO dentro de los top-k chunks + ventana local.
6. Los resultados se fusionan ponderados por los retrieval scores.
```

### 5.3 Resultados

- **HSA-UltraLong-8B (MoE):** Entrenado en 8T tokens, contexto de entrenamiento 8K-32K.
- **Needle-in-a-Haystack a 16M tokens:** >90% accuracy.
- **In-domain (16K):** Comparable a atencion densa full.
- **Memoria:** Lineal con el numero de chunks seleccionados, no con la longitud total.

### 5.4 Integracion con PagedAttention-NG

- HSA es una **alternativa arquitectonica** a la atencion densa, no solo una optimizacion de memoria.
- Para el **Prefill Cluster**, HSA reduce el compute de prefill de O(n^2) a O(n x k) donde k = chunks seleccionados.
- El KV cache solo almacena los **landmark keys** (muy compactos) y los chunks top-k, no toda la secuencia.
- Combinado con FlashPrefill: FlashPrefill acelera el prefill inicial; HSA reduce el costo de atencion durante toda la generacion.

---

## 6. Smurfs: Speculative Decoding Colectivo con SSMs

**Autores:** SC 2025 (ACM Supercomputing).

### 6.1 El problema del speculative decoding tradicional

Los metodos como Medusa, Lookahead, y EAGLE aceleran el decode generando tokens candidatos y verificandolos en paralelo. Pero:
- **Lookahead:** Solo 9.71 tokens/s en Llama2-70B.
- **Ouroboros:** 15.30 tokens/s.
- **CS Drafting:** 11.96-22.92 tokens/s.
- Todos estan limitados por la generacion secuencial del draft.

### 6.2 La solucion

**Smurfs** usa **State Space Models (SSMs)** como draft models colectivos:
- Los SSMs generan tokens **no autoregresivamente** (en paralelo).
- Varios SSMs colaboran adaptativamente segun la tarea.
- La longitud de especulacion se adapta dinamicamente.
- Pipeline de ejecucion: especulacion y verificacion se solapan.

### 6.3 Resultados

| Metodo | Throughput (tokens/s) | Latencia (s/token) |
|--------|----------------------|-------------------|
| Lookahead | 9.71 | 29.44 |
| Ouroboros | 15.30 | 15.35 |
| CS Drafting | 11.96 | 24.01 |
| **Smurfs** | **109.56** | **2.60** |

**Smurfs es 11x mas rapido que Lookahead y 7x que Ouroboros.**

### 6.4 Integracion con PagedAttention-NG

- En el **Decode Cluster**, anadir un **Speculation Worker** que ejecute SSMs ligeros.
- Los SSMs no necesitan KV cache (estado constante), por lo que no compiten por memoria con el modelo target.
- La verificacion del target usa el KV cache paginado existente.
- La combinacion Smurfs + PagedAttention-NG podria alcanzar **>1000 tokens/s** agregados en decode.

---

## 7. Arquitecturas Hibridas Mamba-Transformer

**Autores:** IBM (Bamba), Tencent (Hunyuan TurboS), Microsoft (Phi-4-mini), 2025.

### 7.1 El concepto

En lugar de puro Transformer o puro SSM, intercalar capas:
```
[Attention] -> [Mamba2] -> [FFN] -> [Attention] -> [Mamba2] -> ...
```

**Ventajas:**
- **Capas Attention:** Manejan retrieval preciso, in-context learning, few-shot.
- **Capas Mamba2:** Complejidad O(n) en prefill, memoria O(1) en decode (estado fijo).

### 7.2 Resultados

| Modelo | Parametros | Contexto | Mejora |
|--------|-----------|----------|--------|
| **Bamba-9B** (IBM) | 9B | 32K | **2x throughput** vs Transformer equivalente |
| **Hunyuan TurboS** (Tencent) | 560B total / 56B active | 256K | MoE + Mamba2 + Attention intercalados |
| **Phi-4-mini-flash** (Microsoft) | 3.8B | 64K | **10x throughput**, 2-3x lower latency |

### 7.3 El desafio: Prefix Caching Dual

En modelos hibridos, el prefix caching debe manejar **dos tipos de estado**:
- **KV cache** (Attention): Solo indices de bloques de memoria.
- **SSM state** (Mamba): Debe copiarse fisicamente (in-place update irreversible).

**SGLang resolvio esto con MambaRadixCache:**
- Arbol Radix dual para KV cache y SSM state.
- **Dual-LRU:** Colas independientes para cada tipo de estado.
- Eviccion de KV cache: leaf-to-root (preserva topologia).
- Eviccion de SSM state: elastic (cualquier nivel).

### 7.4 Integracion con PagedAttention-NG

- Para modelos hibridos, el **Decode Cluster** necesita:
  - **PagedAttention** para capas Attention (KV cache paginado).
  - **State manager** para capas SSM (estado fijo por secuencia, no paginable).
- El scheduler debe ser **layer-type aware**: las capas SSM no necesitan asignacion de bloques.
- **Speculative decoding:** Los SSMs en hibridos ya estan presentes; pueden usarse como draft models nativos (aunque requieren cache isolation para COW).

---

## 8. MIRAGE: Remapeo de Parametros para Liberar HBM

**Autores:** Li et al., julio 2025.

### 8.1 Insight

Los **pesos del modelo** (static, inmutables) y el **KV cache** (dinamico) compiten por HBM. Pero los pesos son mas estables y uniformes en acceso.

### 8.2 La solucion

**MIRAGE** evicta paginas de pesos del modelo a CPU DRAM, **reutilizando** esas paginas de HBM como capacidad adicional de KV cache.

**Caracteristicas:**
- **Unidirectional lending:** Solo pesos -> CPU, nunca bidireccional.
- Los pesos se prefetchan de CPU a GPU justo antes de usarse.
- El compute pipeline es uniforme (capas secuenciales), facilitando prefetch predictivo.

### 8.3 Integracion con PagedAttention-NG

- En escenarios de **memoria extrema** (contextos muy largos, batch grande), MIRAGE complementa Harvest:
  - Primero: Harvest usa peer GPU HBM.
  - Segundo: MIRAGE evicta pesos a CPU para liberar HBM local.
  - Tercero: TailorKV offloads capas sparsity-friendly a CPU.
- La cascada de tiers de memoria se vuelve:
  ```
  1. Local HBM (activo)
  2. Peer HBM via NVLink (Harvest)
  3. HBM liberada por pesos evicted (MIRAGE)
  4. CPU DRAM (TailorKV offloading)
  5. NVMe SSD (vLLM Offloading Connector)
  ```

---

## 9. RingX: Sequence Parallelism Escalable para HPC

**Autores:** SC 2025 (ACM Supercomputing).

### 9.1 El problema de Ring Attention clasico

Ring Attention distribuye la secuencia en un anillo de GPUs via P2P. Pero:
- La comunicacion P2P punto-a-punto no aprovecha las topologias HPC con switches rapidos.
- Load imbalance en secuencias causales.
- Escalabilidad limitada.

### 9.2 RingX

Optimizaciones para plataformas HPC:
- **Workload partitioning optimizado:** Striped layout para modelos causales.
- **Protocolos de comunicacion:** Aprovecha Allreduce cuando es mas rapido que P2P ring.
- **Load balancing dinamico.**

**Resultados:**
- **3.4x speedup** vs Ring Attention base.
- Validado en **4096 GPUs** del supercomputador Frontier.
- Soporta tanto bidirectional (ViT) como causal (GPT).

### 9.3 Integracion con PagedAttention-NG

- Para el **Prefill Cluster** con contextos >1M tokens, usar RingX en lugar de Ring Attention base.
- Context Parallelism (CP) con striped layout es el default en Megatron-Core y SGLang.
- En B200 multi-GPU: CP + FP8 KV + vAttention = 1M tokens viable en 32 GPUs.

---

## 10. vLLM Offloading Connector: Layout Contiguo de Bloques

**Autores:** vLLM team, enero 2026 (vLLM 0.12.0).

### 10.1 El problema

En vLLM tradicional, el layout fisico del KV cache esta **fragmentado**:
- Cada capa tiene su propio bloque.
- K y V estan separados.
- Un bloque logico de 16 tokens se fragmenta en `2 x num_layers` sub-bloques fisicos.

Esto es invisible para compute, pero **devastador** para transferencias (offloading, peer-to-peer, RDMA).

### 10.2 La solucion

vLLM 0.12.0 introdujo un layout contiguo:
- **Un solo bloque fisico** contiene K+V de **todas las capas**.
- El block size efectivo se multiplica por `2 x num_layers`.
- **10x mejora** en throughput del offloading connector.

**Ejemplo (Llama-3.1-8B, 32 capas, bloque 16 tokens):**
- Antes: 64 fragmentos por bloque logico.
- Ahora: 1 fragmento contiguo de ~2MB.

### 10.3 Integracion con PagedAttention-NG

- **Obligatorio** para cualquier sistema que haga offloading, Harvest, o transferencia RDMA.
- El layout contiguo es **independiente** del backend de atencion (FlashAttention, FlashInfer, etc.).
- vAttention ya mantiene contiguidad virtual; este cambio garantiza contiguidad fisica para transferencias.

---

## 11. Arquitectura Integrada Final: PagedAttention-NG v2.0

Combinando ambos reportes, la arquitectura evolucionada es:

```
+--------------------------------------------------------------------------+
|                         SCHEDULER UNIFICADO V2                             |
|  Fase-aware + Layer-aware + Chunk-aware + Speculation-aware              |
|  {req_id: {phase, layer_type, num_tokens, speculate_len}}                |
+----------------------------+---------------------------------------------+
                             |
          +------------------+------------------+
          v                  v                  v
+------------------+  +------------------+  +------------------+
| PREFILL CLUSTER  |  | DECODE CLUSTER   |  | SPECULATION      |
| (compute-bound)  |  | (memory-bound)   |  | WORKERS          |
+------------------+  +------------------+  +------------------+
| - FlashPrefill   |  | - FA3/FA4 kernels|  | - SSM drafts     |
|   (sparse >32K)  |  | - vAttention     |  |   (Smurfs)       |
| - HSA            |  | - MLA + INT4-BDR |  | - Cache isolation|
|   (chunks top-k) |  | - Prefix cache   |  |   for COW        |
| - RingX          |  | - Harvest (peer) |  +------------------+
|   (seq parallel) |  | - TailorKV       |
| - Chunked prefill|  |   (layer offloading|
+------------------+  | - Eviccion por   |
          |           |   bloques        |
          |           +------------------+
          |                      |
          |         +------------+------------+
          |         v                         v
          |  +------------------+      +------------------+
          |  | Tier 1: Local HBM|      | Tier 2: Peer HBM |
          |  | (MIRAGE: pesos   |      | (Harvest: NVLink)|
          |  |  evicted a CPU)  |      +------------------+
          |  +------------------+               |
          |         |                    +------+------+
          |         v                    v             v
          |  +------------------+  +------------------+  +------------------+
          |  | Tier 3: CPU DRAM |  | Tier 4: NVMe SSD |  | Tier 5: Remote   |
          |  | (TailorKV offloading| (vLLM Connector) |  | (NIXL/RDMA)      |
          |  +------------------+  +------------------+  +------------------+
          |
          v
+--------------------------------------------------------------------------+
|                    TRANSFERENCIA KV (NIXL / RDMA)                        |
|  - Layout contiguo (vLLM 0.12.0)                                       |
|  - Comprimido (MLA + INT4-BDR)                                         |
|  - Entre Prefill y Decode clusters                                     |
+--------------------------------------------------------------------------+
```

### Jerarquia de memoria de 5 tiers:

| Tier | Tecnologia | Latencia | Uso |
|------|-----------|----------|-----|
| 1 | Local HBM | ~1 TB/s | Activos, pesos esenciales |
| 2 | Peer HBM (NVLink) | ~900 GB/s | Cache oportunista (Harvest) |
| 3 | CPU DRAM | ~50 GB/s (PCIe) | Offloading selectivo (TailorKV) |
| 4 | NVMe SSD | ~7 GB/s | Cache persistente (vLLM Connector) |
| 5 | Remote (RDMA) | ~25-100 GB/s | Comparticion cross-node (NIXL) |

---

## 12. Hoja de Ruta Ampliada

### Fase 1: Fundacion (1-2 meses)
- [x] Implementar vAttention como backend de memoria.
- [x] Integrar FlashAttention-3.
- [ ] **NUEVO:** Adoptar layout contiguo de bloques (vLLM 0.12.0 style).
- [ ] **NUEVO:** Integrar Harvest para peer-to-peer GPU caching.

### Fase 2: Compresion y Offloading (2-3 meses)
- [x] Implementar token-wise INT4 con BDR-128.
- [ ] **NUEVO:** Implementar clasificacion de capas (TailorKV-style).
- [ ] **NUEVO:** Anadir CPU offloading asincrono con double buffering.
- [ ] **NUEVO:** Implementar MIRAGE para remapeo de pesos a CPU.

### Fase 3: Prefill Avanzado (3-4 meses)
- [x] Implementar chunked prefill.
- [ ] **NUEVO:** Integrar FlashPrefill para secuencias >32K.
- [ ] **NUEVO:** Implementar HSA para modelos que lo soporten.
- [ ] **NUEVO:** Anadir RingX para sequence parallelism en HPC.

### Fase 4: Decode y Speculation (2-3 meses)
- [x] Prefix caching con hash exacto.
- [ ] **NUEVO:** Implementar Smurfs-style speculative decoding con SSMs.
- [ ] **NUEVO:** Cache isolation para COW en SSM states.
- [ ] **NUEVO:** Eviccion por bloques con scores de atencion acumulados.

### Fase 5: Escala y Hibridos (2-3 meses)
- [x] Disaggregated serving.
- [ ] **NUEVO:** Soporte para modelos hibridos Mamba-Transformer.
- [ ] **NUEVO:** Prefix caching dual (KV + SSM state).
- [ ] **NUEVO:** Pipeline de transferencia comprimida entre clusters.

---

## 13. Conclusiones

El estudio anterior establecio una base solida con seis ejes de mejora. Este complemento anade **nueve tecnicas adicionales** que elevan el techo de rendimiento:

1. **FlashPrefill** elimina el cuello de botella del prefill en contextos largos (27.78x).
2. **TailorKV** permite servir modelos 128K en GPUs de gama media (RTX 3090).
3. **Harvest** aprovecha NVLink para caching entre GPUs (5.65x vs CPU).
4. **HSA** habilita contextos de 16M tokens con calidad preservada.
5. **Smurfs** acelera el decode 11x via speculative decoding con SSMs.
6. **Hibridos Mamba-Transformer** reducen memoria a O(1) en capas SSM.
7. **MIRAGE** convierte pesos estaticos en capacidad dinamica de KV cache.
8. **RingX** escala sequence parallelism a miles de GPUs.
9. **Layout contiguo** hace viable el offloading masivo (10x throughput).

**La vision final:** Un sistema donde el prefill de 1M tokens tarda segundos (FlashPrefill + HSA + RingX), el decode corre a >100 tokens/s por stream (Smurfs + FA3), y la memoria se gestiona automaticamente a traves de 5 tiers (Harvest + MIRAGE + TailorKV + vAttention + vLLM Connector), todo orquestado por un scheduler unificado que entiende fases, capas, chunks y especulacion.

---

## Referencias Adicionales

- Fan, Q., et al. (2026). *FlashPrefill: Instantaneous Pattern Discovery and Thresholding for Ultra-Fast Long-Context Prefilling*. arXiv:2603.06199v1.
- Yao, D., et al. (2025). *TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization*. ACL 2025 Findings.
- Harvest Team. (2026). *Opportunistic Peer-to-Peer GPU Caching for LLM Inference*. arXiv:2602.00328v1.
- Hu, X., et al. (2025). *Every Token Counts: Generalizing 16M Ultra-Long Context in Large Language Models*. arXiv:2511.23319.
- Smurfs Team. (2025). *Towards Efficient LLM Inference via Collective and Adaptive Speculative Decoding*. SC 2025.
- IBM Research. (2025). *Bamba-9B: Hybrid Mamba-Transformer*.
- Tencent Hunyuan Team. (2025). *Hunyuan-TurboS: Hybrid Transformer-Mamba2-MoE*.
- Microsoft. (2025). *Phi-4-mini-flash-reasoning: SambaY Architecture*.
- Li, et al. (2025). *MIRAGE: Parameter Remapping for KV Cache Expansion*.
- RingX Team. (2025). *RingX: Scalable Parallel Attention for Long-Context Learning on HPC*. SC 2025.
- vLLM Team. (2026). *Inside vLLM's New KV Offloading Connector*. vLLM Blog, enero 2026.
- SGLang Team. (2026). *Hybrid Model Support for Mamba-Transformer*. Alibaba Cloud Blog, febrero 2026.
