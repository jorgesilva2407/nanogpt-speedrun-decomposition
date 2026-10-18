"""Loader equivalence harness (pure Python; torch is NOT imported).

Extracts the `DistributedDataLoader` class (and nothing else) from a record snapshot and from
train_flagged.py with `ast`, and executes only that class against list-based stand-ins for torch/numpy and
synthetic shards whose token values encode (shard, position). It then compares, step by step, the token
offsets each virtual rank trains on:

    record snapshot run as an 8-GPU job (8 loader instances, the record's own training-loop protocol)
    vs train_flagged.py at world_size in {1, 2, 4, 8} (unified batch convention).

What this verifies: index arithmetic, shard-advance rules and first-batch replay of the loader code as
written. What it does not verify: anything involving real tensors, devices, or the model.

Usage: python -I tools/loader_equivalence.py <snapshot.py> <train_flagged.py> <preset.yaml> [--steps N]
"""
import argparse, ast, json, os, sys, types

# ---------------------------------------------------------------------------------------------- stand-ins
class FakeArr:
    """1-D token array stand-in: supports slicing, astype, view, cuda, to, len, numpy (identity-ish)."""
    def __init__(self, data):
        self.data = data
    def __getitem__(self, k):
        return FakeArr(self.data[k])
    def __len__(self):
        return len(self.data)
    def astype(self, *a, **k):
        return self
    def view(self, *shape):
        return self
    def reshape(self, *shape):
        return self
    def cuda(self, *a, **k):
        return self
    def to(self, *a, **k):
        return self
    def numpy(self):
        return self
    def first(self):
        return self.data[0]

def make_torch_np(shards):
    torch = types.SimpleNamespace(tensor=lambda buf, dtype=None: buf, long='long', int32='int32',
                                  int64='int64', uint16='uint16')
    np = types.SimpleNamespace(int32='int32', uint16='uint16')
    return torch, np

def synthetic_shards(lengths):
    # token value = shard * 10**9 + position  -> identifies exactly where a micro-batch came from
    return {f'shard_{i:03d}.bin': FakeArr(range(i * 10**9, i * 10**9 + n)) for i, n in enumerate(lengths)}

def load_loader_class(path, shards):
    src = open(path).read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DistributedDataLoader')
    mod = ast.Module(body=[cls], type_ignores=[])
    torch, np = make_torch_np(shards)
    files = sorted(shards)
    ns = dict(torch=torch, np=np,
              glob=types.SimpleNamespace(glob=lambda pattern: list(files)),
              Path=types.SimpleNamespace(cwd=lambda: types.SimpleNamespace(glob=lambda pattern: list(files))),
              _peek_data_shard=lambda f, *a: len(shards[os.path.basename(str(f))]),
              _load_data_shard=lambda f, *a: shards[os.path.basename(str(f))],
              _load_data_shard_torch=lambda f, *a: shards[os.path.basename(str(f))])
    exec(compile(mod, path, 'exec'), ns)
    return ns['DistributedDataLoader']

# ---------------------------------------------------------------------------------------------- protocols
def record_stream(snapshot_cls, protocol, B, T, W, steps):
    """Token offsets per step for the record run as a W-GPU job. protocol: dict describing the record loop."""
    loaders = [snapshot_cls('pattern', *protocol['ctor'](B, T, r, W)) for r in range(W)]
    cur = [ld.next_batch() for ld in loaders]          # prefetch before the loop
    if protocol.get('reset_before_loop'):
        for ld in loaders:
            ld.reset()
    out = []
    for _ in range(steps):
        out.append([c[0].first() for c in cur])
        cur = [ld.next_batch() for ld in loaders]       # accumulation steps == 1 at W=8
    return out

def flagged_stream(flagged_cls, hp, flags, B, T, W, steps):
    vr = hp['global_batch_tokens'] // hp['micro_batch_tokens']
    A = vr // W
    kw = {}
    if flags.get('long_context'):
        B, T, kw = 1, B * T, dict(rows_2d=False)
    if flags.get('code_opts_15'):
        kw['torch_io'] = True
    loaders = [flagged_cls('pattern', B, T, vr, range(r * A, (r + 1) * A), hp['loader_shard_rule'],
                           hp['loader_replay_first_round'], **kw) for r in range(W)]
    cur = [ld.next_batch() for ld in loaders]
    out = []
    for _ in range(steps):
        step = [None] * vr
        for a in range(A):
            for r in range(W):
                step[r * A + a] = cur[r][0].first()
                cur[r] = loaders[r].next_batch()
        out.append(step)
    return out

PROTOCOLS = {
    # record #11: DistributedDataLoader(pattern, B, T, rank, W); prefetch; train_loader.reset() before loop
    'r11': dict(ctor=lambda B, T, r, W: (B, T, r, W), reset_before_loop=True),
    # records #12-#13: DistributedDataLoader(pattern, B=1, T=64K, rank, W); no reset before the loop
    'r12': dict(ctor=lambda B, T, r, W: (1, B * T, r, W)),
    'r13': dict(ctor=lambda B, T, r, W: (1, B * T, r, W)),
    # records #14-#17: DistributedDataLoader(pattern, T=64K, rank, W)
    'r14': dict(ctor=lambda B, T, r, W: (B * T, r, W)),
    'r15': dict(ctor=lambda B, T, r, W: (B * T, r, W)),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('snapshot'); ap.add_argument('flagged'); ap.add_argument('preset')
    ap.add_argument('--protocol', required=True, choices=sorted(PROTOCOLS))
    ap.add_argument('--steps', type=int, default=400)
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(a.preset))
    hp, flags = cfg['hparams'], cfg.get('flags') or {}
    # scaled-down geometry with the record's ratios: 8 micro-batches per step, micro = B*T
    B, T = 4, 8                                  # micro = 32 tokens (record: 64 x 1024)
    micro = B * T
    hp = dict(hp, global_batch_tokens=8 * micro, micro_batch_tokens=micro)
    # shard lengths chosen so boundaries fall at different offsets (record shards: 100M tokens = 190.7 steps)
    lengths = [6103, 6100, 6119, 6096, 6145, 6087, 6111, 6130]
    shards = synthetic_shards(lengths)
    rec = record_stream(load_loader_class(a.snapshot, shards), PROTOCOLS[a.protocol], B, T, 8, a.steps)
    res = {}
    for W in (1, 2, 4, 8):
        fl = flagged_stream(load_loader_class(a.flagged, shards), hp, flags, B, T, W, a.steps)
        first_diff = next((s for s in range(a.steps) if fl[s] != rec[s]), None)
        res[W] = first_diff
        print(f'world_size={W}: first step whose micro-batches differ from the 8-GPU record stream: {first_diff}')
    boundary_steps = [s for s in range(1, a.steps) if any(x // 10**9 != y // 10**9 for x, y in zip(rec[s], rec[s - 1]))]
    print(f'(shard boundaries crossed in the record stream at steps {boundary_steps[:6]}...; '
          f'first two steps: {rec[0][:2]}, {rec[1][:2]})')
    sys.exit(0 if all(v is None for v in res.values()) else 1)

if __name__ == '__main__':
    main()
