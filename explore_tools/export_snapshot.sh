#!/bin/sh
# export_snapshot.sh <NN> <tree_sha> <log_path_at_5deb47d> <script_name> <code_sha_or_none>
# Writes records/rNN/{tree/,log_source/<script>,ANCHOR.txt}. Read-only on upstream.
set -eu
NN=$1; TREE=$2; LOG=$3; SCRIPT=$4; CODE=$5
W=$(cd "$(dirname "$0")/.." && pwd)
D="$W/records/r$NN"
rm -rf "$D"; mkdir -p "$D/tree" "$D/log_source"
git -C "$W/upstream" archive "$TREE" | tar -x -C "$D/tree"
"$W/.venv/bin/python" -I - "$W/upstream" "$LOG" "$D/log_source/$SCRIPT" <<'PY'
import subprocess, sys
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0] if '/' in sys.argv[0] else '.')
repo, log, out = sys.argv[1:4]
SEP = '=' * 100
t = subprocess.run(['git', '-C', repo, 'show', f'5deb47d:{log}'], capture_output=True, text=True, check=True).stdout
lines = t.split('\n')
if lines[0] == SEP:
    src = '\n'.join(lines[1:lines.index(SEP, 1)]) + '\n'
else:
    src = '\n'.join(lines[:lines.index(SEP)])
open(out, 'w').write(src)
PY
{
  echo "record: $NN"
  echo "code_sha: $CODE"
  echo "tree_sha: $(git -C "$W/upstream" rev-parse "$TREE")"
  echo "log: $LOG (read at 5deb47d)"
  echo "log_source_sha1: $(shasum "$D/log_source/$SCRIPT" | cut -d' ' -f1)"
  echo "log_source_lines: $(wc -l < "$D/log_source/$SCRIPT" | tr -d ' ')"
  if [ "$CODE" != none ]; then
    if git -C "$W/upstream" show "$CODE:$SCRIPT" > /tmp/.snap_$$ 2>/dev/null; then
      if cmp -s /tmp/.snap_$$ "$D/log_source/$SCRIPT"; then echo "code_vs_log: identical"; else echo "code_vs_log: differs ($(diff /tmp/.snap_$$ "$D/log_source/$SCRIPT" | grep -c '^[<>]') changed lines; see MANIFEST relation)"; fi
      rm -f /tmp/.snap_$$
    fi
  fi
} > "$D/ANCHOR.txt"
cat "$D/ANCHOR.txt"
