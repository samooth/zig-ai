#!/usr/bin/env bash
# bench_sweep.sh — Barrido prefill×decode zig-ai vs tabla llama-bench (comparativa).
#
# Genera prompts de tamaño aproximado (repitiendo una frase semilla) y corre
# el engine ReleaseFast, parseando: prefill ms (N tok) + generación tok/s.
# Determinista: temperature 0 + seed 42.
#
# Uso:
#   benchmarks/bench_sweep.sh -m <modelo.gguf> [-r 3] [-n 128] [--tag s]
#
# Salida: results/sweep<TAG>.tsv  (label prompt_tok ttft_ms ttft_tok tok_s vram_peak_mb)
set -euo pipefail

MODEL="" REPS=3 N_TOKENS=128 TAG="" ENGINE="zig-out/bin/zig-ai-engine"
SEED_PHRASE="The capital of France is Paris. The largest planet in the solar system is Jupiter. Water boils at 100 degrees Celsius."

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) MODEL="$2"; shift 2;;
    -r) REPS="$2"; shift 2;;
    -n) N_TOKENS="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --engine) ENGINE="$2"; shift 2;;
    *) echo "opción desconocida: $1" >&2; exit 1;;
  esac
done
[[ -z "$MODEL" ]] && { echo "-m <modelo.gguf> requerido" >&2; exit 1; }
[[ -x "$ENGINE" ]] || { echo "engine no encontrado: $ENGINE" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/results"; mkdir -p "$OUT_DIR"
OUT_TSV="$OUT_DIR/sweep${TAG}.tsv"
[[ -f "$OUT_TSV" ]] || printf 'label\tprompt_tok\tttft_ms\tttft_tok\ttok_s\tvram_peak_mb\n' > "$OUT_TSV"

# frase semilla ≈ 24 tokens; repetir para alcanzar el objetivo
make_prompt() {
  local target_reps=$1
  local p=""
  for ((i=0;i<target_reps;i++)); do p+="$SEED_PHRASE "; done
  printf '%s' "$p"
}

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
  local label=$1 reps_tok=$2
  local log_vram log_run
  log_vram=$(mktemp /tmp/opencode/vram.XXXX)
  log_run=$(mktemp /tmp/opencode/run.XXXX)
  local PROMPT; PROMPT=$(make_prompt "$reps_tok")

  "$ENGINE" -m "$MODEL" --prompt "$PROMPT" -n "$N_TOKENS" \
    --temperature 0 --seed 42 >"$log_run" 2>&1 &
  local eng_pid=$!
  start_vram_poll "$eng_pid" "$log_vram"
  wait "$eng_pid"; local rc=$?
  wait "$POLL_PID" 2>/dev/null || true
  if [[ $rc -ne 0 ]]; then
    echo "[sweep] FALLÓ $label (rc=$rc):" >&2; tail -5 "$log_run" >&2
    rm -f "$log_vram" "$log_run"; return 1
  fi

  local ttft tok_s ptok
  ttft=$(grep -oP 'prefill \K[0-9]+\.[0-9]+' "$log_run" | tail -1)
  ptok=$(grep -oP 'prefill [0-9]+\.[0-9]+ ms \(\K[0-9]+' "$log_run" | tail -1)
  tok_s=$(grep -oP 'generación [0-9]+\.[0-9]+ ms \(\K[0-9]+\.[0-9]+' "$log_run" | tail -1 || true)
  local vram; vram=$(cat "$log_vram")
  if [[ -z "$ttft" || -z "$tok_s" ]]; then
    echo "[sweep] no pude parsear métricas ($label):" >&2; tail -3 "$log_run" >&2
    rm -f "$log_vram" "$log_run"; return 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "${ptok:-NA}" "$ttft" "${ptok:-NA}" "$tok_s" "${vram:-NA}" >> "$OUT_TSV"
  echo "[sweep] ${label}: prompt=${ptok}tok ttft=${ttft}ms tok/s=${tok_s} vram=${vram}MB"
  rm -f "$log_vram" "$log_run"
}

# ~24 tok/frase: 1→~24tok(no exacto), 11→~264, 43→~1032
for rep in 1 11 43; do
  for ((i=1;i<=REPS;i++)); do
    run_once "pp${rep}" "$rep" || exit 1
  done
done
echo "[sweep] resultados en $OUT_TSV"
