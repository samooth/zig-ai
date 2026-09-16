#!/usr/bin/env bash
# bench_baseline.sh — Baseline reproducible del motor (lane-c C1).
#
# Métricas por corrida: TTFT (ms prefill), tok/s decode, VRAM pico (MB).
# Determinista: temperature 0 (greedy) + seed fijo + prompt fijo.
#
# Uso:
#   benchmarks/bench_baseline.sh -m <modelo.gguf> [opciones]
#
# Opciones:
#   -n <tok>        tokens a generar (def 128)
#   -r <reps>       repeticiones (def 3)
#   -l              añade --layer-stream a esta config
#   -ctk/-ctv <f>   formato KV (se pasan tal cual al engine)
#   --label <s>     etiqueta de la config en la salida (def: default)
#   --engine <p>    ruta al binario (def: zig-out/bin/zig-ai-engine)
#   --lock          toma .bench.lock (OBLIGATORIO para el 27B)
#   --tag <s>       sufijo de fichero de resultados (def: "")
#
# Salida: results/baseline<TAG>.tsv con una fila por réplica:
#   label  model  rep  ttft_ms  tok_s  vram_peak_mb  gen_tokens
set -euo pipefail

MODEL="" N_TOKENS=128 REPS=3 EXTRA=() LABEL="default" ENGINE="zig-out/bin/zig-ai-engine"
LOCK=0 TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) MODEL="$2"; shift 2;;
    -n) N_TOKENS="$2"; shift 2;;
    -r) REPS="$2"; shift 2;;
    -l) EXTRA+=(--layer-stream); shift;;
    -ctk|-ctv) EXTRA+=("$1" "$2"); shift 2;;
    --label) LABEL="$2"; shift 2;;
    --engine) ENGINE="$2"; shift 2;;
    --lock) LOCK=1; shift;;
    --tag) TAG="$2"; shift 2;;
    *) echo "opción desconocida: $1" >&2; exit 1;;
  esac
done
[[ -z "$MODEL" ]] && { echo "-m <modelo.gguf> requerido" >&2; exit 1; }
[[ -x "$ENGINE" ]] || { echo "engine no encontrado: $ENGINE (¿build ReleaseFast?)" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/results"; mkdir -p "$OUT_DIR"
OUT_TSV="$OUT_DIR/baseline${TAG}.tsv"
[[ -f "$OUT_TSV" ]] || printf 'label\tmodel\trep\tttft_ms\ttok_s\tvram_peak_mb\tgen_tokens\n' > "$OUT_TSV"

PROMPT="The capital of France is Paris. The largest planet in the solar system is"

acquire_lock() {
  if [[ "$LOCK" == "1" ]]; then
    exec 9>"$ROOT/.bench.lock"
    if ! flock -n 9; then
      echo "[bench] esperando .bench.lock (otro bench GPU en curso)..." >&2
      flock 9
    fi
  fi
}

# VRAM pico: sondeo del proceso mientras corre el engine (MB usados por la GPU).
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
  local rep=$1
  local log_vram; log_vram=$(mktemp /tmp/opencode/vram.XXXX)
  local log_run;  log_run=$(mktemp /tmp/opencode/run.XXXX)

  # shellcheck disable=SC2086
  "$ENGINE" -m "$MODEL" \
    --prompt "$PROMPT" \
    -n "$N_TOKENS" \
    --temperature 0 --seed 42 \
    "${EXTRA[@]}" >"$log_run" 2>&1 &
  local eng_pid=$!
  start_vram_poll "$eng_pid" "$log_vram"
  wait "$eng_pid"; local rc=$?
  wait "$POLL_PID" 2>/dev/null || true

  if [[ $rc -ne 0 ]]; then
    echo "[bench] FALLÓ rep=$rep (rc=$rc):" >&2
    tail -5 "$log_run" >&2
    rm -f "$log_vram" "$log_run"
    return 1
  fi

  local ttft tok_s gen
  ttft=$(grep -oP 'prefill \K[0-9]+\.[0-9]+' "$log_run" | tail -1)
  tok_s=$(grep -oP 'generación [0-9]+\.[0-9]+ ms \(\K[0-9]+\.[0-9]+' "$log_run" | tail -1 || true)
  gen=$(grep -oP 'Generados \K[0-9]+' "$log_run" | tail -1)
  local vram; vram=$(cat "$log_vram")

  if [[ -z "$ttft" || -z "$tok_s" ]]; then
    echo "[bench] no pude parsear métricas (rep=$rep):" >&2; tail -3 "$log_run" >&2
    rm -f "$log_vram" "$log_run"; return 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$(basename "$MODEL")" "$rep" "$ttft" "$tok_s" "${vram:-NA}" "${gen:-NA}" >> "$OUT_TSV"
  echo "[bench] ${LABEL} rep=${rep}: ttft=${ttft}ms tok/s=${tok_s} vram_peak=${vram}MB"
  rm -f "$log_vram" "$log_run"
}

acquire_lock
for ((i=1;i<=REPS;i++)); do
  run_once "$i" || exit 1
done
echo "[bench] resultados en $OUT_TSV"
