#!/bin/bash
# 1.4a: bench pp512 — matriz {fp16,q4_0} × {LMSPLIT=0,1}
# Uso: bash benchmarks/bench_pp512.sh   (requiere GPU LIBRE)
MODEL=/ai/models/Qwen3.5-0.8B-Q4_0.gguf
P=$(cat /tmp/prompt512.txt)

run() {
    local ctk=$1 ctv=$2 lmsplit=$3 tag=$4
    out=$(LMSPLIT=$lmsplit timeout 300 ./zig-out/bin/zig-ai-engine \
        -m "$MODEL" -p "$P" -n 1 --temperature 0 --seed 42 \
        -ctk "$ctk" -ctv "$ctv" 2>&1 | grep -a "Métricas" | head -1)
    pre_ms=$(echo "$out" | grep -aoE "prefill [0-9.]+" | grep -aoE "[0-9.]+")
    ntok=$(echo "$out" | grep -aoE "\(([0-9]+) tok\)" | grep -aoE "[0-9]+")
    if [ -n "$pre_ms" ] && [ -n "$ntok" ]; then
        tps=$(python3 -c "print(f'{$ntok/($pre_ms/1000):.1f}')")
        echo "$tag: prefill=${pre_ms}ms ntok=$ntok => $tps tok/s  [$out]"
    else
        echo "$tag: FAIL — [$out]"
    fi
}

echo "== GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
run fp16 fp16 0 "fp16 LMSPLIT=0"
run fp16 fp16 1 "fp16 LMSPLIT=1"
run q4_0 q4_0 0 "q4_0 LMSPLIT=0"
run q4_0 q4_0 1 "q4_0 LMSPLIT=1"
