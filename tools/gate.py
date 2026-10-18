"""Run every static gate for one record and write gates/rNN.txt. Exit 1 if any gate fails.

Gates (nothing touches torch or a GPU):
  1. syntax      train_flagged.py parses with no SyntaxWarning.
  2. constants   configs/rNN.yaml hparams == constants extracted from the snapshot (tools/extract_constants.py).
  3. ledger      every hunk of diffs/rNN-vs-r(N-1).diff classified (tools/check_ledger.py)   [N >= 12]
  4. params      parameter names/shapes of train_flagged.py under preset rNN == those of the snapshot after the
                 documented name map; totals printed; #11 and #18 totals checked against the brief.
  5. loader      micro-batch stream of train_flagged.py at world_size 1/2/4/8 == the snapshot's 8-GPU stream
                 (tools/loader_equivalence.py; loader classes only, torch stubbed).
Usage: python -I tools/gate.py NN [--records ../records]
"""
import argparse, ast, io, math, os, re, subprocess, sys, warnings, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import param_inventory as pinv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# expected totals from the brief, re-derived analytically in CHANGES/r11.md and CHANGES/r18.md
EXPECTED = {11: dict(total=162_201_642, embed=38_633_472, lm_head=38_633_472, blocks=84_934_692),
            18: dict(total=275_742_772, embed=38_633_472, lm_head=38_633_472, value_embeddings=115_900_416,
                     blocks=82_575_406)}

# snapshot parameter name -> flagged parameter name (regex rewrites applied in order); filled per record
_R16 = [(r'^embed\.', 'transformer.wte.'), (r'^blocks\.', 'transformer.h.')]
NAME_MAP = {16: _R16, 17: _R16, 18: _R16}

def snapshot_path(records, n):
    d = os.path.join(records, f'r{n}', 'log_source')
    return os.path.join(d, os.listdir(d)[0])

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    return r.returncode, r.stdout + r.stderr

def gate_syntax(out):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            with open(os.path.join(HERE, 'train_flagged.py')) as f:
                ast.parse(f.read())
            out.append('syntax: PASS')
            return True
        except (SyntaxError, SyntaxWarning) as e:
            out.append(f'syntax: FAIL {e}')
            return False

def map_name(n, rec):
    for pat, rep in NAME_MAP.get(rec, []):
        n = re.sub(pat, rep, n)
    return n

def gate_params(records, n, out):
    snap = snapshot_path(records, n)
    sp, sb = pinv.inventory(snap)
    fp, fb = pinv.inventory(os.path.join(HERE, 'train_flagged.py'), os.path.join(HERE, 'configs', f'r{n}.yaml'))
    smap = {map_name(k, n): s for k, s in sp}
    fmap = dict(fp)
    ok = True
    only_s = sorted(set(smap) - set(fmap)); only_f = sorted(set(fmap) - set(smap))
    shape_diff = sorted(k for k in set(smap) & set(fmap) if tuple(smap[k]) != tuple(fmap[k]))
    if only_s or only_f or shape_diff:
        ok = False
        out.append(f'params: FAIL only_in_snapshot={only_s[:8]} only_in_flagged={only_f[:8]} shape_diff={shape_diff[:8]}')
    st, sg = pinv.summarize(sp); ft, fg = pinv.summarize(fp)
    out.append(f'params: snapshot total {st:,} ({st/1e6:.1f}M) {dict((k, v) for k, v in sg.items())}')
    out.append(f'params: flagged  total {ft:,} ({ft/1e6:.1f}M) {dict((k, v) for k, v in fg.items())}')
    out.append(f'params: {len(sp)} tensors in snapshot, {len(fp)} in flagged; buffers snapshot={len(sb)} flagged={len(fb)}')
    bs = sorted((map_name(k, n), tuple(s)) for k, s in sb); bf = sorted((k, tuple(s)) for k, s in fb)
    if bs != bf:
        ok = False
        out.append(f'params: FAIL buffers differ: snapshot-only {sorted(set(bs)-set(bf))[:4]} flagged-only {sorted(set(bf)-set(bs))[:4]}')
    if n in EXPECTED:
        for k, v in EXPECTED[n].items():
            got = ft if k == 'total' else fg.get(k)
            good = got == v
            ok &= good
            out.append(f'params: expected {k} {v:,} -> flagged {got:,} {"PASS" if good else "FAIL"}')
    if ok:
        out.append('params: PASS (names and shapes identical after name map)')
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('n', type=int); ap.add_argument('--records', default=os.path.join(HERE, '..', 'records'))
    ap.add_argument('--protocol', default=None)
    a = ap.parse_args()
    n = a.n
    out = [f'# static gates for r{n}']
    results = {}
    results['syntax'] = gate_syntax(out)
    snap = snapshot_path(a.records, n)
    rc, txt = run([PY, '-I', '-W', 'ignore', 'tools/extract_constants.py', snap, f'configs/r{n}.yaml'])
    out.append('constants:\n' + txt.rstrip()); results['constants'] = rc == 0
    if n >= 12:
        rc, txt = run([PY, '-I', 'tools/check_ledger.py', f'ledger/r{n}.yaml'])
        out.append('ledger:\n' + txt.rstrip()); results['ledger'] = rc == 0
    try:
        results['params'] = gate_params(a.records, n, out)
    except Exception as e:
        out.append(f'params: FAIL interpreter error {e!r}'); results['params'] = False
    proto = a.protocol or f'r{n}'
    rc, txt = run([PY, '-I', '-W', 'ignore', 'tools/loader_equivalence.py', snap, 'train_flagged.py',
                   f'configs/r{n}.yaml', '--protocol', proto, '--steps', '600'])
    out.append('loader:\n' + txt.rstrip()); results['loader'] = rc == 0
    out.append('SUMMARY: ' + ' '.join(f'{k}={"PASS" if v else "FAIL"}' for k, v in results.items()))
    os.makedirs(os.path.join(HERE, 'gates'), exist_ok=True)
    open(os.path.join(HERE, 'gates', f'r{n}.txt'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))
    sys.exit(0 if all(results.values()) else 1)

if __name__ == '__main__':
    main()
