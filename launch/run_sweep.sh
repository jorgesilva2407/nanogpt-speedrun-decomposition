#!/usr/bin/env bash
# Run every job of the given arm(s) on THIS host, sequentially, atomically.
#   launch/run_sweep.sh --host lab-a100 --nproc 1 --arm A3 --arm A7 [--dry-run]
# The arm -> host binding is recorded in experiments/hosts.json; the planner refuses to put an arm's comparison pair
# or any of its seeds on a second host, and adds the fixed points this host needs (host-qualified replicas when another
# host already owns them). Completed jobs (DONE) are skipped; incomplete ones are re-run from scratch.
set -euo pipefail
HOST=""; NPROC=1; DRY=""; ARMS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --arm) ARMS+=("--arm" "$2"); shift 2 ;;
    --dry-run) DRY="--dry-run"; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$HOST" && ${#ARMS[@]} -gt 0 ]] || { echo "usage: $0 --host LABEL --arm ID [--arm ID ...] [--nproc N] [--dry-run]" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLAIM="--claim"; [[ -n "$DRY" ]] && CLAIM=""
JOBS="$(python -I "$ROOT/launch/plan.py" --host "$HOST" "${ARMS[@]}" $CLAIM)"   # exits non-zero (refusal) on a host-rule violation
while read -r EXP NAME SEED; do
  [[ -z "$EXP" ]] && continue
  python -I "$ROOT/launch/run_one.py" --exp "$EXP" --name "$NAME" --seed "$SEED" --host "$HOST" --nproc "$NPROC" $DRY
done <<< "$JOBS"
