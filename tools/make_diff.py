"""Write diffs/rNN-vs-rMM.diff: every difference between two pristine snapshots.

Part 1 — training script: the LOG-EMBEDDED sources (records/rNN/log_source/*, the snapshots proper),
         unified diff with 1 line of context (fine-grained hunks for the ledger).
Part 2 — every other file of the two pristine trees (records/rNN/tree), unified diff with 3 lines of context;
         binary files get one 'Binary files differ' entry. Excluded, and listed in the diff header:
         records/** (committed logs and record write-ups: evidence, catalogued in records/MANIFEST.md) and the
         trees' own training scripts (superseded by Part 1; tree-vs-log relation is in MANIFEST.md).
Usage: python -I tools/make_diff.py <records> <old NN> <new NN> <out.diff>
"""
import difflib, os, sys

def read(p):
    with open(p, 'rb') as f:
        b = f.read()
    try:
        return b.decode('utf-8'), False
    except UnicodeDecodeError:
        return b, True

def walk(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            full = os.path.join(d, f)
            out[os.path.relpath(full, root)] = full
    return out

SCRIPTS = ('train_gpt2.py', 'train_gpt.py')

def main(records, a, b, out):
    A, B = os.path.join(records, f'r{a}'), os.path.join(records, f'r{b}')
    la, = os.listdir(os.path.join(A, 'log_source'))
    lb, = os.listdir(os.path.join(B, 'log_source'))
    lines = [f'# diff r{b} vs r{a}\n',
             f'# part 1: log-embedded training script (records/r{a}/log_source/{la} -> records/r{b}/log_source/{lb}), -U1\n',
             f'# part 2: all other tree files (records/r{a}/tree -> records/r{b}/tree), -U3;'
             f' excluded: records/**, tree copies of {", ".join(SCRIPTS)}\n']
    sa, _ = read(os.path.join(A, 'log_source', la)); sb, _ = read(os.path.join(B, 'log_source', lb))
    lines += difflib.unified_diff(sa.splitlines(True), sb.splitlines(True), f'a/{la}', f'b/{lb}', n=1)
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n\\ No newline at end of file\n'
    ta, tb = walk(os.path.join(A, 'tree')), walk(os.path.join(B, 'tree'))
    keep = lambda p: not p.startswith('records' + os.sep) and p not in SCRIPTS
    for p in sorted(set(ta) | set(tb)):
        if not keep(p):
            continue
        xa, bina = read(ta[p]) if p in ta else ('', False)
        xb, binb = read(tb[p]) if p in tb else ('', False)
        if bina or binb:
            if xa != xb:
                lines.append(f'Binary files a/{p} and b/{p} differ\n')
            continue
        if xa == xb:
            continue
        d = list(difflib.unified_diff(xa.splitlines(True), xb.splitlines(True),
                                      f'a/{p}' if p in ta else '/dev/null', f'b/{p}' if p in tb else '/dev/null', n=3))
        for i, l in enumerate(d):
            if not l.endswith('\n'):
                d[i] = l + '\n\\ No newline at end of file\n'
        lines += d
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        f.writelines(lines)

if __name__ == '__main__':
    main(*sys.argv[1:5])
