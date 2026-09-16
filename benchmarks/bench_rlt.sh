#!/usr/bin/env bash
# bench_rlt.sh — A/B benchmark: RLT feedback ON vs OFF
#
# Mide tok/s, VRAM pico, y calidad (texto generado) con un modelo Qwen3.5.
# Compara: --no-rlt-feedback (baseline) vs --rlt-feedback (RLT ON).
#
# Uso:
#   benchmarks/bench_rlt.sh -m <modelo.gguf> [opciones]
#
# Opciones:
#   -n <tok>        tokens a generar (def 64)
#   -r <reps>       repeticiones (def 3)
#   -p <prompt>     prompt (def: "The capital of France is")
#   --engine <p>    ruta al binario (def: zig-out/bin/zig-ai-engine)
#   --tag <s>       sufijo de fichero (def: "")
#   --rlt-alpha <f> forzar alpha (def: auto desde GGUF)
#   --swa <n>       SWA window size (def: 0 = off)
set -euo pipefail

MODEL="" N_TOKENS=64 REPS=3 PROMPT="The capital of France is" ENGINE="zig-out/bin/zig-ai-engine"
TAG="" RLT_ALPHA="" SWA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) MODEL="$2"; shift 2;;
    -n) N_TOKENS="$2"; shift 2;;
    -r) REPS="$2"; shift 2;;
    -p) PROMPT="$2"; shift 2;;
    --engine) ENGINE="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --rlt-alpha) RLT_ALPHA="$2"; shift 2;;
    --swa) SWA="$2"; shift 2;;
    *) echo "opción desconocida: $1" >&2; exit 1;;
  esac
done
[[ -z "$MODEL" ]] && { echo "-m <modelo.gguf> requerido" >&2; exit 1; }
[[ -x "$ENGINE" ]] || { echo "engine no encontrado: $ENGINE (¿build ReleaseFast?)" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/results"; mkdir -p "$OUT_DIR"
OUT_TSV="$OUT_DIR/rlt_ab${TAG}.tsv"
[[ -f "$OUT_TSV" ]] || printf 'label\tmodel\trep\tttft_ms\ttok_s\tvram_peak_mb\tgen_tokens\toutput_text\n' > "$OUT_TSV"

# Acquire GPU lock
exec 9>"$ROOT/.bench.lock"
if ! flock -n 9; then
  echo "[bench] esperando .bench.lock (otro bench GPU en curso)..." >&2
  flock 9
fi

# VRAM poll
start_vram_poll() {
  local pid=$1 peak_file=$2
  (
    local mx=0 cur
    while kill -0 "$pid" 2>/dev/null; do
      cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [[ -n "$cur" && "$cur" -gt "$mx" ]] && mx=$cur
      sleep 0.05
    done
    echo "$mx" > "$peak_file"
  ) &
  POLL_PID=$!
}

run_once() {
  local label=$1 rep=$2
  shift 2
  local extra_flags=("$@")
  local log_vram; log_vram=$(mktemp /tmp/opencode/vram.XXXX)
  local log_run;  log_run=$(mktemp /tmp/opencode/run.XXXX)

  "$ENGINE" -m "$MODEL" \
    --prompt "$PROMPT" \
    -n "$N_TOKENS" \
    --temperature 0 --seed 42 \
    "${extra_flags[@]}" >"$log_run" 2>&1 &
  local eng_pid=$!
  start_vram_poll "$eng_pid" "$log_vram"
  wait "$eng_pid"; local rc=$?
  wait "$POLL_PID" 2>/dev/null || true

  local vram_peak; vram_peak=$(<"$log_vram" 2>/dev/null || echo 0)
  local ttft_ms tok_s gen_tokens output_text
  ttft_ms=$(grep -oP 'TTFT[: ]+\K[0-9.]+' "$log_run" 2>/dev/null | head -1 || echo "0")
  tok_s=$(grep -oP '[0-9.]+(?= *tok/s)' "$log_run" 2>/dev/null | head -1 || echo "0")
  gen_tokens=$(grep -oP 'generated[: ]+\K[0-9]+' "$log_run" 2>/dev/null | head -1 || echo "$N_TOKENS")
  output_text=$(tail -1 "$log_run" 2>/dev/null | tr '\t' ' ' | head -c 80)

  printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$(basename "$MODEL")" "$rep" "$ttft_ms" "$tok_s" "$vram_peak" "$gen_tokens" "$output_text" \
    >> "$OUT_TSV"

  rm -f "$log_vram" "$log_run"
}

echo "═══════════════════════════════════════════════════════════════"
echo "RLT A/B Benchmark"
echo "  Modelo:    $(basename "$MODEL")"
echo "  Tokens:    $N_TOKENS"
echo "  Réplicas:  $REPS"
echo "  Prompt:    $PROMPT"
echo "  SWA:       $SWA"
echo "  Resultado: $OUT_TSV"
echo "═══════════════════════════════════════════════════════════════"

# Build extra flags for SWA
SWA_FLAGS=()
if [[ "$SWA" -gt 0 ]]; then
  SWA_FLAGS=()
fi

echo ""
echo "── Config A: baseline (RLT OFF) ──"
for rep in $(seq 1 "$REPS"); do
  echo "  réplica $rep/$REPS..."
  run_once "baseline" "$rep" --no-rlt-feedback "${SWA_FLAGS[@]}"
done

echo ""
echo "── Config B: RLT feedback ON ──"
for rep in $(seq 1 "$REPS"); do
  echo "  réplica $rep/$REPS..."
  run_once "rlt_on" "$rep" --rlt-feedback "${SWA_FLAGS[@]}"
done

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Resultados (tok/s promedio):"
echo ""
awk -F'\t' 'NR>1 {
  sum[$1] += $5; n[$1]++
}
END {
  for (label in sum) {
    printf "  %-12s %.1f tok/s  (n=%d)\n", label, sum[label]/n[label], n[label]
  }
}' "$OUT_TSV"
echo ""
echo "VRAM pico (MB):"
awk -F'\t' 'NR>1 {
  if ($6+0 > max[$1]+0) max[$1] = $6
}
END {
  for (label in max) {
    printf "  %-12s %s MB\n", label, max[label]
  }
}' "$OUT_TSV"
echo ""
echo "Salida generada (última réplica):"
awk -F'\t' 'NR>1 { last[$1] = $8 }
END {
  for (label in last) {
    printf "  %-12s \"%s\"\n", label, last[label]
  }
}' "$OUT_TSV"
echo "═══════════════════════════════════════════════════════════════"
