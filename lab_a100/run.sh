#!/usr/bin/env bash
# Runs the lab-A100 learning runs of lab_a100/PROTOCOL.md from ONE invocation (default phase "all": anchor, gaps, rq2, seeds2, rq3b,
# in that order) inside the study image, and uploads every finished run to RESULTS_REPO (the A100-only HF dataset, never the H100 one).
# Nothing here is shared with the H100 tooling except launch/run_one.py and the trainer, which do the training itself; this script
# replaces launch/cloud_worker.sh and launch/make_jobs.py for the lab.
# Stopping the container loses only the run in progress (the contract of launch/cloud_worker.sh):
#   * a run already in RESULTS_REPO is skipped;
#   * a run with DONE on the local disk is skipped by run_one.py and only uploaded;
#   * a run without DONE is moved aside to seed<k>.incomplete-<timestamp> and restarts from step 0 (there is no resume);
#   * MAX_FAILS consecutive failed runs halt the invocation (HALTED-<phase>; delete it to try again).
# Restart with the same command (docker start -a <name>): finished runs are skipped and the compile cache in $WORKDIR/cache survives.
#
#   bash run.sh [all|anchor|gaps|rq2|seeds2|rq3b]      DRY_RUN=1 bash run.sh    (clone, dry-run every job, test upload; no training)
#
# Environment: GH_TOKEN (read-only clone), HF_TOKEN (write to RESULTS_REPO only), RESULTS_REPO, STUDY_TAG (the a100-* tag: it holds the
# RQ3b configurations that study-freeze lacks), DOCKER_TAG, HOST_LABEL (lab-a100); optional WORKDIR (default /workspace; mount a host
# directory there), MAX_FAILS (default 2).
set -euo pipefail
: "${GH_TOKEN:?}" "${HF_TOKEN:?}" "${RESULTS_REPO:?}" "${STUDY_TAG:?}" "${DOCKER_TAG:?}" "${HOST_LABEL:?}"
PHASE="${1:-all}"
WORKDIR="${WORKDIR:-/workspace}"; ROOT="$WORKDIR/study"; MAX_FAILS="${MAX_FAILS:-2}"; fails=0
REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/jorgesilva2407/nanogpt-speedrun-decomposition.git"
exec > >(tee -a "$WORKDIR/worker-$PHASE.log") 2>&1
echo "== $(date -u +%FT%TZ) lab_a100 phase $PHASE on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU')${DRY_RUN:+ (DRY RUN)}"

# compile caches live in the mounted work directory, so they survive removing the container (aoi_A5 compiles for about 2 h)
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKDIR/cache/inductor}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKDIR/cache/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

HALTED="$WORKDIR/HALTED-$PHASE"
[ -e "$HALTED" ] && { echo "== HALTED earlier: $(cat "$HALTED"); not starting. Delete $HALTED after fixing the cause."; exit 3; }

jobs() {  # one "<configuration>:<seeds>" per line; the order is the run order
  case "$1" in
    anchor) echo "fp_r11:0 1 2"; echo "aoi_A6:0 1"; echo "aoi_A3:0" ;;
    gaps)   for c in aoi_A1b aoi_A4 aoi_A5 aoi_A5b; do echo "$c:0"; done ;;
    rq2)    echo "fp_r18:0 1"; echo "loo_A5b:0" ;;
    seeds2) for c in aoi_A1b aoi_A4 aoi_A5 aoi_A5b loo_A5b; do echo "$c:1"; done ;;
    rq3b)   for s in 0.5 0.75; do for c in fp_r11 aoi_A6 aoi_A3; do echo "rq3b_s${s}_${c}:0"; done; done ;;
    all)    jobs anchor; jobs gaps; jobs rq2; jobs seeds2; jobs rq3b ;;
    *) echo "unknown phase $1 (all|anchor|gaps|rq2|seeds2|rq3b)" >&2; exit 2 ;;
  esac
}

[ -d "$ROOT/.git" ] || git clone -q --depth 1 --branch "$STUDY_TAG" "$REPO_URL" "$ROOT"
cd "$ROOT"
[ -n "$(git tag --points-at HEAD | grep -x "$STUDY_TAG")" ] || { echo "checkout is not at $STUDY_TAG"; exit 1; }
export DOCKER_TAG HF_TOKEN
PY=$(command -v python || command -v python3)

upload() {  # upload results/<rel> to <rel> in RESULTS_REPO (one commit, so a stopped upload leaves nothing half there)
  $PY -I - "$RESULTS_REPO" "$1" <<'PY'
import sys
from huggingface_hub import HfApi
repo, rel = sys.argv[1], sys.argv[2]
api = HfApi()
if not api.repo_exists(repo, repo_type='dataset'):
    api.create_repo(repo, repo_type='dataset', private=True)
api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=f'results/{rel}', path_in_repo=rel, commit_message=f'{rel}')
PY
}
upload_retry() {  # a network blip must not cost a finished run: 4 tries, a minute apart
  for i in 1 2 3 4; do upload "$1" && return 0; echo "   upload of $1 failed (try $i/4)"; sleep 60; done; return 1
}
in_repo() {
  $PY -I -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().file_exists('$RESULTS_REPO', '$1', repo_type='dataset') else 1)" 2>/dev/null
}

JOBFILE="$WORKDIR/jobs-$PHASE.txt"
while IFS=: read -r cfg seeds; do for s in $seeds; do echo "$cfg $s"; done; done < <(jobs "$PHASE") > "$JOBFILE"
echo "-- phase $PHASE: $(wc -l < "$JOBFILE") runs"; cat "$JOBFILE"

if [ "${DRY_RUN:-0}" = 1 ]; then
  while read -r cfg seed; do
    $PY -I launch/run_one.py --exp "experiments/runs/$cfg" --name "$cfg" --seed "$seed" --host "$HOST_LABEL" --nproc 1 --dry-run >/dev/null \
      && echo "   ok  $cfg seed$seed" || { echo "   FAILED dry run: $cfg seed$seed"; exit 1; }
  done < "$JOBFILE"
  mkdir -p results/_selftest && echo "selftest $(date -u +%FT%TZ) phase=$PHASE tag=$STUDY_TAG image=$DOCKER_TAG host=$HOST_LABEL" > results/_selftest/ok.txt
  upload _selftest || { echo "   FAILED: cannot upload to $RESULTS_REPO"; exit 1; }
  echo "== DRY RUN passed for phase $PHASE"; exit 0
fi

[ -e data/fineweb10B/fineweb_train_000018.bin ] || $PY data/cached_fineweb10B.py 18   # val + 18 x 100M tokens

while read -r cfg seed <&3; do
  if in_repo "$cfg/seed$seed/DONE"; then echo "skip $cfg seed$seed (already in $RESULTS_REPO)"; continue; fi
  t0=$(date +%s)
  if $PY -I launch/run_one.py --exp "experiments/runs/$cfg" --name "$cfg" --seed "$seed" --host "$HOST_LABEL" --nproc 1; then
    fails=0
    [ -e "results/$cfg/seed$seed/wall_seconds" ] || echo $(( $(date +%s) - t0 )) > "results/$cfg/seed$seed/wall_seconds"
    if upload_retry "$cfg/seed$seed"; then echo "uploaded $cfg seed$seed ($(cat results/$cfg/seed$seed/wall_seconds) s)"
    else echo "UPLOAD FAILED $cfg seed$seed: the run is kept on disk and is uploaded at the end or at the next start"; fi
  else
    echo "FAILED $cfg seed$seed after $(( $(date +%s) - t0 )) s (see output above)"
    fails=$((fails + 1))
    if [ "$fails" -ge "$MAX_FAILS" ]; then
      echo "$(date -u +%FT%TZ) phase $PHASE: $fails consecutive failures, last $cfg seed$seed" > "$HALTED"
      echo "== HALTED: $fails consecutive failures"; exit 3
    fi
  fi
done 3< "$JOBFILE"

# last pass: upload any finished run whose upload failed above
while read -r cfg seed; do
  [ -e "results/$cfg/seed$seed/DONE" ] && ! in_repo "$cfg/seed$seed/DONE" && { upload_retry "$cfg/seed$seed" && echo "uploaded $cfg seed$seed (late)"; } || true
done < "$JOBFILE"
echo "== $(date -u +%FT%TZ) phase $PHASE finished"
