#!/bin/bash
# Runs each tier in $TIERS, and pushes /workspace/worker.log to the HF results repo every 15 min and at the end.
cd /workspace
PY=$(command -v python || command -v python3)
LOGNAME_HF="_logs/${HOST_LABEL}-${CONTAINER_ID:-$(hostname)}-w${WORKER:-0}of${WORKERS:-1}.log"
pushlog() { $PY -I -c "import os; from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/workspace/worker.log', path_in_repo='$LOGNAME_HF', repo_id=os.environ['RESULTS_REPO'], repo_type='dataset', commit_message='worker log')" >/dev/null 2>&1 || true; }
touch /workspace/worker.log
( while sleep 900; do pushlog; done ) & PUSHER=$!
for T in ${TIERS//,/ }; do
  TIER=$T bash study/launch/cloud_worker.sh; rc=$?
  echo "== $(date -u +%FT%TZ) tier $T exited rc=$rc" >> /workspace/worker.log
done
echo "== $(date -u +%FT%TZ) all tiers finished: ${TIERS}" >> /workspace/worker.log
kill $PUSHER 2>/dev/null; pushlog
