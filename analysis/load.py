"""Load run outputs (results/<name>/seed<k>/) into arrays. CPU only (numpy).

Only completed runs (DONE present) are loaded. Every run carries its host label; `group` refuses to pool runs from
different hosts (proposal 3.1: measurements are never pooled across hosts; the cross-hardware check is analysed
separately).
"""
import glob, json, os
import numpy as np

class HostPoolingError(ValueError):
    pass

class Run:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, 'meta.json')) as f:
            self.meta = json.load(f)
        self.host = self.meta.get('host_label')
        self.seed = self.meta.get('seed')
        self.exp_id = self.meta.get('exp_id')
        self.name = os.path.basename(os.path.dirname(path))
        train, val = [], []
        with open(os.path.join(path, 'metrics.jsonl')) as f:
            for line in f:
                r = json.loads(line)
                (train if r['kind'] == 'train' else val).append(r)
        self.train_step = np.array([r['step'] for r in train], dtype=float)
        self.train_loss = np.array([r['train_loss'] for r in train], dtype=float)
        self.val_step = np.array([r['step'] for r in val], dtype=float)
        self.val_tokens = np.array([r['tokens'] for r in val], dtype=float)
        self.val_loss = np.array([r['val_loss'] for r in val], dtype=float)
        self.val_fraction = np.array([r['val_fraction'] for r in val], dtype=float)
        timing = []
        tpath = os.path.join(path, 'timing.jsonl')
        if os.path.exists(tpath):
            with open(tpath) as f:
                timing = [json.loads(l) for l in f]
        self.time_step = np.array([r['step'] for r in timing], dtype=float)
        self.time_ms = np.array([r['gpu_ms'] for r in timing], dtype=float)
        self.time_warmup = np.array([bool(r['compile_warmup']) for r in timing], dtype=bool)

    def val_curve(self, full_only=True):
        """(steps, losses) of validation; full_only keeps evaluations on the whole validation set."""
        m = self.val_fraction >= 1.0 - 1e-12 if full_only else np.ones_like(self.val_step, dtype=bool)
        return self.val_step[m], self.val_loss[m]

def load_runs(results_root, pattern='*'):
    runs = []
    for d in sorted(glob.glob(os.path.join(results_root, pattern, 'seed*'))):
        if os.path.exists(os.path.join(d, 'DONE')) and os.path.exists(os.path.join(d, 'meta.json')):
            runs.append(Run(d))
    return runs

def group(runs, key=lambda r: r.name):
    """Group runs by result name; raises HostPoolingError if a group mixes hosts."""
    out = {}
    for r in runs:
        out.setdefault(key(r), []).append(r)
    for k, rs in out.items():
        hosts = {r.host for r in rs}
        if len(hosts) > 1:
            raise HostPoolingError(f'{k}: runs from several hosts {sorted(hosts)} must not be pooled')
    return out

def same_host(*groups):
    """Assert that the groups being compared come from one host (an arm's comparison pair)."""
    hosts = {r.host for g in groups for r in g}
    if len(hosts) > 1:
        raise HostPoolingError(f'comparison across hosts {sorted(hosts)} refused')
    return hosts.pop() if hosts else None
