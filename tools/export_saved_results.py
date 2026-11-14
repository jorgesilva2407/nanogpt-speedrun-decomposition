"""Writes the compact copy of the study's results that is committed to the repository (saved_results/).

The full result datasets (Hugging Face: speedrun-decomp/speedrun-results for the H100, speedrun-decomp/speedrun-results-a100 for the lab
A100) hold every training-loss row and per-step bookkeeping, ~35 MB of text. analysis/tables.py needs only: meta.json, DONE, the
full-validation rows of metrics.jsonl and, of timing.jsonl, the step, GPU time and warm-up flag. This keeps exactly those (values are copied
unchanged, nothing is rounded or recomputed) so that the tables of reports/table.md can be rebuilt from a plain clone, offline.
Kept besides: wall_seconds, resolved.json, and the H100 job lists of the sweep (_logs). Left out: training-loss rows, A100 throughput
(a compile-and-behaviour check, not a result; DEVIATIONS.md 7), self-tests and raw logs.

  python -I tools/export_saved_results.py <downloaded h100 dir> <downloaded a100 dir> saved_results
"""
import json, os, shutil, sys

def trim_run(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in ('DONE', 'meta.json', 'wall_seconds', 'resolved.json'):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), dst)
    with open(os.path.join(src, 'metrics.jsonl')) as fi, open(os.path.join(dst, 'metrics.jsonl'), 'w') as fo:
        for line in fi:
            if json.loads(line)['kind'] == 'val':
                fo.write(line)
    tp = os.path.join(src, 'timing.jsonl')
    if os.path.exists(tp):
        with open(tp) as fi, open(os.path.join(dst, 'timing.jsonl'), 'w') as fo:
            for line in fi:
                r = json.loads(line)
                fo.write(json.dumps({k: r[k] for k in ('step', 'gpu_ms', 'compile_warmup')}) + '\n')

def runs_in(root, skip=('_logs', '_selftest', 'throughput')):
    for cfg in sorted(os.listdir(root)):
        if cfg in skip or not os.path.isdir(os.path.join(root, cfg)):
            continue
        for seed in sorted(os.listdir(os.path.join(root, cfg))):
            d = os.path.join(root, cfg, seed)
            if seed.startswith('seed') and os.path.exists(os.path.join(d, 'DONE')):
                yield cfg, seed, d

def main(h100, a100, out):
    for name, root in (('h100', h100), ('a100', a100)):
        for cfg, seed, d in runs_in(root):
            trim_run(d, os.path.join(out, name, cfg, seed))
    tp = os.path.join(h100, 'throughput')
    for dirpath, dirs, files in os.walk(tp):
        if 'DONE' in files and os.path.basename(dirpath).startswith('seed'):
            trim_run(dirpath, os.path.join(out, 'h100', os.path.relpath(dirpath, h100)))
    for log in sorted(os.listdir(os.path.join(h100, '_logs'))):
        d = os.path.join(h100, '_logs', log)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if os.path.isfile(os.path.join(d, f)):
                os.makedirs(os.path.join(out, 'h100', '_logs', log), exist_ok=True)
                shutil.copy(os.path.join(d, f), os.path.join(out, 'h100', '_logs', log, f))

if __name__ == '__main__':
    main(*sys.argv[1:4])
