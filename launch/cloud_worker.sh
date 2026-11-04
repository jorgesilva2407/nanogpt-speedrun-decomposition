#!/usr/bin/env bash
# Runs one worker's share of a tier on ONE cloud GPU instance, inside the study image, and uploads every finished run
# to a private Hugging Face dataset repo (the single place results are collected). Safe to restart: runs already in the
# HF repo are skipped, a run with DONE on local disk is only uploaded, an incomplete one is restarted from step 0.
#
# Set these as environment variables on the instance (Vast.ai: template "Environment variables"):
#   GH_TOKEN      fine-grained GitHub token, read-only "Contents" on this repo only (to clone it; revoke after Nov 13)
#   HF_TOKEN      Hugging Face token with write access (results upload)
#   RESULTS_REPO  e.g. <hf-user>/speedrun-results (created private on first use)
#   STUDY_TAG     the frozen git tag, e.g. study-freeze
#   DOCKER_TAG    the image you launched, e.g. ghcr.io/jorgesilva2407/modded-nanogpt-flagged:<sha> (recorded in meta.json)
#   HOST_LABEL    one label for the whole study, e.g. vast-h100 (must equal study.yaml throughput.host)
#   TIER          fixed_points | p1 | p2 | p3 | throughput
#   WORKERS, WORKER   this instance's share, e.g. WORKERS=8 WORKER=3 (0-based); default 1 / 0
#   DRY_RUN=1     check everything except training, on any machine (no GPU, no data download): clone at the tag,
#                 build the job list, dry-run each job, upload a test file to RESULTS_REPO. Run this on a laptop
#                 before renting anything, e.g. WORKDIR=/tmp/wtest DRY_RUN=1 ... bash launch/cloud_worker.sh
# Then:  bash -c "$(curl -fsSL -H "Authorization: token $GH_TOKEN" \
#          https://raw.githubusercontent.com/jorgesilva2407/nanogpt-speedrun-decomposition/$STUDY_TAG/launch/cloud_worker.sh)"
set -euo pipefail
: "${GH_TOKEN:?}" "${HF_TOKEN:?}" "${RESULTS_REPO:?}" "${STUDY_TAG:?}" "${DOCKER_TAG:?}" "${HOST_LABEL:?}" "${TIER:?}"
WORKERS="${WORKERS:-1}"; WORKER="${WORKER:-0}"
REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/jorgesilva2407/nanogpt-speedrun-decomposition.git"
ROOT="${WORKDIR:-/workspace}/study"; LOG="${WORKDIR:-/workspace}/worker.log"
exec > >(tee -a "$LOG") 2>&1
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU")
echo "== $(date -u +%FT%TZ) worker $WORKER/$WORKERS tier $TIER on $GPU${DRY_RUN:+ (DRY RUN)}"

[ -d "$ROOT/.git" ] || git clone -q --depth 1 --branch "$STUDY_TAG" "$REPO_URL" "$ROOT"
cd "$ROOT"
[ -n "$(git tag --points-at HEAD | grep -x "$STUDY_TAG")" ] || { echo "checkout is not at $STUDY_TAG"; exit 1; }
export DOCKER_TAG HF_TOKEN
PY=$(command -v python || command -v python3)


upload() {  # upload results/<rel> to <rel> in the HF dataset repo
  $PY -I - "$RESULTS_REPO" "$1" <<'PY'
import sys
from huggingface_hub import HfApi
repo, rel = sys.argv[1], sys.argv[2]
api = HfApi(); api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=f'results/{rel}', path_in_repo=rel,
                  commit_message=f'{rel}')
PY
}
in_repo() {
  $PY -I -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().file_exists('$RESULTS_REPO', '$1', repo_type='dataset') else 1)" 2>/dev/null
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "-- jobs for tier $TIER, worker $WORKER/$WORKERS:"
  if [ "$TIER" = throughput ]; then
    $PY -I -c "import json;print('\n'.join(j['exp_id'] for j in json.load(open('experiments/matrix.json'))['throughput_jobs']))"
  else
    $PY -I launch/make_jobs.py --tier "$TIER" --workers "$WORKERS" --worker "$WORKER" | tee jobs.txt
    while read -r exp name seed; do
      [ -z "$exp" ] && continue
      $PY -I launch/run_one.py --exp "$exp" --name "$name" --seed "$seed" --host "$HOST_LABEL" --nproc 1 --dry-run >/dev/null \
        && echo "   ok  $name seed$seed" || { echo "   FAILED dry run: $name seed$seed"; exit 1; }
    done < jobs.txt
  fi
  echo "-- test upload to $RESULTS_REPO:"
  mkdir -p results/_selftest && echo "selftest $(date -u +%FT%TZ) tag=$STUDY_TAG image=$DOCKER_TAG host=$HOST_LABEL" > results/_selftest/ok.txt
  upload _selftest || { echo "   FAILED: cannot upload to $RESULTS_REPO (check HF_TOKEN write access, RESULTS_REPO name, huggingface_hub installed)"; exit 1; }
  echo "   upload ok: $RESULTS_REPO/_selftest/ok.txt"
  echo "== DRY RUN passed: clone, job list, configs and upload all work"; exit 0
fi

[ -e data/fineweb10B/fineweb_train_000018.bin ] || $PY data/cached_fineweb10B.py 18   # val + 18 x 100M tokens

if [ "$TIER" = throughput ]; then
  # All throughput jobs on this one instance, back to back (the whole GPU is rented, so 'idle' is answered yes).
  for exp in $($PY -I -c "import json;print(' '.join(j['exp_id'] for j in json.load(open('experiments/matrix.json'))['throughput_jobs']))"); do
    if [ -n "$(ls -d results/throughput/$exp/$HOST_LABEL/*/run/seed0/DONE 2>/dev/null)" ]; then echo "skip throughput $exp"; continue; fi
    echo yes | $PY -I launch/throughput.py --exp "experiments/runs/$exp" --host "$HOST_LABEL" --nproc 1 \
      || { echo "FAILED throughput $exp"; continue; }
    upload "throughput/$exp"
  done
  echo "== throughput finished $(date -u +%FT%TZ)"; exit 0
fi

$PY -I launch/make_jobs.py --tier "$TIER" --workers "$WORKERS" --worker "$WORKER" > jobs.txt
while read -r exp name seed <&3; do
  [ -z "$exp" ] && continue
  if in_repo "$name/seed$seed/DONE"; then echo "skip $name seed$seed (already in $RESULTS_REPO)"; continue; fi
  t0=$(date +%s)
  if $PY -I launch/run_one.py --exp "$exp" --name "$name" --seed "$seed" --host "$HOST_LABEL" --nproc 1; then
    [ -e "results/$name/seed$seed/wall_seconds" ] || echo $(( $(date +%s) - t0 )) > "results/$name/seed$seed/wall_seconds"
    upload "$name/seed$seed" && echo "uploaded $name seed$seed ($(cat results/$name/seed$seed/wall_seconds) s)"
  else
    echo "FAILED $name seed$seed after $(( $(date +%s) - t0 )) s (see output above)"
  fi
done 3< jobs.txt
echo "== worker $WORKER/$WORKERS tier $TIER finished $(date -u +%FT%TZ)"
