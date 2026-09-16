#!/usr/bin/env bash
# 5.5 (lane-f): matriz de aceptación spec-v2 — rejection sampling (temp>0).
# Categorías de prompts naturales × 3 seeds × 4 modos sobre Qwen3.8-4B-Q6_K
# (cabeza MTP blk.32):
#   A base   : greedy sin spec (baseline coherencia/velocidad)
#   B sg     : spec greedy (paridad A==B; drift f16 esperado en cola)
#   C rej    : spec rejection temp 0.8 (LA métrica de la matriz)
#   D nospec : sampling sin spec temp 0.8 (control distribución de C)
# Salida: acceptance rate, tok/round, t/s + paridad greedy con punto de
# divergencia (byte/palabras) para separar drift f16 de corrupción de estado.
#
# Uso:
#   flock .bench.lock ./benchmarks/matrix_acceptance_v2.sh              # 4 modos
#   flock .bench.lock ./benchmarks/matrix_acceptance_v2.sh --only-rej   # re-cosecha C
#   flock .bench.lock ./benchmarks/matrix_acceptance_v2.sh --cats "repetitive narrative code"
#   ./benchmarks/matrix_acceptance_v2.sh --parse-only                    # solo tablas
#
# Protocolo GPU: OBLIGATORIO flock .bench.lock para los modos con engine.
# NOTA post-fix (2026-09-09): modo C requiere el fix spec_lmlogits_host del
# camino CPU-GEMV del verify (src/inference/cli.zig, landed) — cosechas de C
# anteriores a esa fecha son INVÁLIDAS (p(x)=0 ⇒ rechazo artificial).
set -u
ENGINE=./zig-out/bin/zig-ai-engine
MODEL=/ai/models/Qwen3.8-4B-Q6_K.gguf
CTX=512
N=48
TEMP=0.8
OUT=/tmp/opencode/matrix55
mkdir -p "$OUT"

PARSE_ONLY=0
ONLY_REJ=0
CAT_FILTER=""
CATS_ORDER=(factual reasoning repetitive narrative code)
for arg in "$@"; do
  case "$arg" in
    --parse-only) PARSE_ONLY=1 ;;
    --only-rej) ONLY_REJ=1 ;;
    --cats) CAT_FILTER="PENDING" ;;
    --cats=*) CAT_FILTER="${arg#--cats=}" ;;
    *)
      if [ "$CAT_FILTER" = "PENDING" ]; then CAT_FILTER="$arg"
      else echo "uso: $0 [--parse-only|--only-rej|--cats \"a b c\"]" >&2; exit 2; fi ;;
  esac
done

declare -A CATS=(
  [factual]="The capital of France is"
  [reasoning]="If all roses are flowers and some flowers fade quickly, then"
  [code]="def fibonacci(n):\n    if n <= 1:\n        return n\n    return"
  [repetitive]="Monday Tuesday Wednesday Thursday Friday Saturday Sunday Monday Tuesday"
  [narrative]="Once upon a time in a small village, there lived"
)
SEEDS=(42 1337 2026)

txt_of() { # $1=log → solo el texto generado (entre cabecera y Métricas)
  sed -n "/Generación (/,/Métricas/p" "$1" 2>/dev/null | sed '1d;$d'
}

run_case() { # $1=cat $2=seed $3=modo(base|sg|rej|nospec) $4=args-extra
  local cat="$1" seed="$2" mode="$3"; shift 3
  timeout 900 "$ENGINE" --model "$MODEL" --prompt "${CATS[$cat]}" \
    -n "$N" --ctx-size "$CTX" --seed "$seed" "$@" \
    > "$OUT/${cat}_s${seed}_${mode}.log" 2>&1
}

if [ "$PARSE_ONLY" -eq 0 ]; then
  for cat in "${CATS_ORDER[@]}"; do
    if [ -n "$CAT_FILTER" ]; then
      case " $CAT_FILTER " in *" $cat "*) ;; *) continue;; esac
    fi
    for seed in "${SEEDS[@]}"; do
      if [ "$ONLY_REJ" -eq 0 ]; then
        run_case "$cat" "$seed" base   --temperature 0
        run_case "$cat" "$seed" sg     --temperature 0 --spec-type draft-mtp
        run_case "$cat" "$seed" nospec --temperature "$TEMP"
      fi
      run_case "$cat" "$seed" rej --temperature "$TEMP" --spec-type draft-mtp
    done
  done
fi

echo "=== MATRIZ 5.5 — Qwen3.8-4B-Q6_K, temp=$TEMP, n=$N, ctx=$CTX ==="
printf "%-12s %6s | %10s %7s %7s | %s\n" "cat" "seed" "accept%" "tok/rd" "t/s" "salida(rej, primeros 30 chars)"
for f in "$OUT"/{factual,reasoning,repetitive,narrative,code}_s*.log; do
  base=$(basename "$f" .log)
  cat=${base%%_s*}; seed=${base#*_s}; seed=${seed%%_*}; mode=${base##*_}
  [ "$mode" = "rej" ] || continue
  acc=$(grep -aoE "aceptados=[0-9]+ \([0-9.]+%\)" "$f" | tail -1 | grep -oE "[0-9.]+%" | tr -d '%')
  trd=$(grep -aoE "tok/ronda=[0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+$")
  tps=$(grep -aoE "\([0-9.]+ tok/s\)" "$f" | tail -1 | grep -oE "[0-9.]+")
  txt=$(txt_of "$f" | tr '\n' ' ' | head -c 30)
  printf "%-12s %6s | %10s %7s %7s | %s\n" "$cat" "$seed" "${acc:-NA}" "${trd:-NA}" "${tps:-NA}" "$txt"
done

echo
echo "=== Coherencia C (spec-rej) vs D (no-spec) — primeros 60 chars ==="
for cat in "${CATS_ORDER[@]}"; do
  for seed in "${SEEDS[@]}"; do
    c=$(txt_of "$OUT/${cat}_s${seed}_rej.log" | tr '\n' ' ' | head -c 60)
    d=$(txt_of "$OUT/${cat}_s${seed}_nospec.log" | tr '\n' ' ' | head -c 60)
    printf "%-12s s%s\n  C: %s\n  D: %s\n" "$cat" "$seed" "$c" "$d"
  done
done

echo
echo "=== Paridad greedy (A vs B): prefijo común antes de diverger ==="
for cat in "${CATS_ORDER[@]}"; do
  for seed in "${SEEDS[@]}"; do
    a=$(txt_of "$OUT/${cat}_s${seed}_base.log" | tr '\n' ' ')
    b=$(txt_of "$OUT/${cat}_s${seed}_sg.log" | tr '\n' ' ')
    if [ -z "$a" ] || [ -z "$b" ]; then echo "$cat s$seed: SIN DATOS"; continue; fi
    if [ "$a" = "$b" ]; then
      echo "$cat s$seed: IDENTICO byte a byte"
    else
      d=$(cmp <(printf '%s' "$a") <(printf '%s' "$b") 2>/dev/null | head -1 | grep -oE 'byte [0-9]+' | grep -oE '[0-9]+')
      if [ -z "$d" ]; then
        la=$(printf '%s' "$a" | wc -c); lb=$(printf '%s' "$b" | wc -c)
        d=$(( la < lb ? la : lb ))
      fi
      w=$(printf '%s' "${a:0:$d}" | wc -w)
      echo "$cat s$seed: diverge @byte $d (~$w palabras coincidentes)"
    fi
  done
done
