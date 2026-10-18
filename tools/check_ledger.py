"""Static gate: every hunk of diffs/rNN-vs-rMM.diff is classified exactly once in ledger/rNN.yaml.

Checks: 1:1 correspondence (file + exact hunk header); required fields; allowed vocabularies; 'flag: none'
carries a reason; every referenced flag exists in train_flagged.py's FLAGS registry and every 'config:<key>'
or 'coupled' key is a Hyperparameters field (both read with ast, nothing imported); no TODO left.
Exit status 1 on any failure. Usage: python -I tools/check_ledger.py ledger/rNN.yaml [--trainer train_flagged.py]
"""
import argparse, ast, os, sys
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffparse import hunks

CATEGORIES = {'model-mechanism', 'throughput-only', 'hyperparameter/schedule', 'infra/logging', 'capacity-change', 'cosmetic'}
PROVENANCE = {'direct', 'inferred'}
NUMERICS = {'identical', 'ulp-level', 'changed', 'n/a'}
REQUIRED = ('id', 'file', 'header', 'description', 'category', 'flag', 'provenance', 'bundled', 'numerics')

def trainer_registry(path):
    tree = ast.parse(open(path).read())
    flags, hparams = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FLAGS' for t in node.targets):
            flags |= {k.value for k in node.value.keys}
        if isinstance(node, ast.ClassDef) and node.name == 'Hyperparameters':
            hparams |= {n.target.id for n in node.body if isinstance(n, ast.AnnAssign)}
    return flags, hparams

def as_list(x):
    return x if isinstance(x, list) else [x]

def check(ledger_path, trainer):
    led = yaml.safe_load(open(ledger_path))
    base = os.path.dirname(os.path.dirname(os.path.abspath(ledger_path)))
    diff_hunks = hunks(open(os.path.join(base, led['diff'])).read())
    flags, hparams = trainer_registry(trainer)
    errs = []
    entries = led.get('hunks') or []
    seen_ids = set()
    index = {}
    for e in entries:
        miss = [k for k in REQUIRED if k not in e]
        if miss:
            errs.append(f"{e.get('id', '?')}: missing fields {miss}")
            continue
        if e['id'] in seen_ids:
            errs.append(f"{e['id']}: duplicate id")
        seen_ids.add(e['id'])
        if 'TODO' in yaml.safe_dump(e):
            errs.append(f"{e['id']}: TODO left")
        cats = as_list(e['category'])
        if not set(cats) <= CATEGORIES:
            errs.append(f"{e['id']}: bad category {cats}")
        if e['provenance'] not in PROVENANCE:
            errs.append(f"{e['id']}: bad provenance {e['provenance']}")
        if e['numerics'] not in NUMERICS:
            errs.append(f"{e['id']}: bad numerics {e['numerics']}")
        if not isinstance(e['bundled'], bool):
            errs.append(f"{e['id']}: bundled must be true/false")
        for f in as_list(e['flag']):
            if f == 'none':
                if not e.get('flag_reason'):
                    errs.append(f"{e['id']}: flag none without flag_reason")
            elif str(f).startswith('config:'):
                if f[7:] not in hparams:
                    errs.append(f"{e['id']}: config key {f[7:]} is not a Hyperparameters field")
            elif f not in flags:
                errs.append(f"{e['id']}: flag {f} not in train_flagged.py FLAGS")
        for k in as_list(e.get('coupled', [])):
            key = str(k).split(':')[0]
            if key and key not in hparams:
                errs.append(f"{e['id']}: coupled key {key} is not a Hyperparameters field")
        key = (e['file'], e['header'])
        if key in index:
            errs.append(f"{e['id']}: same hunk as {index[key]}")
        index[key] = e['id']
    diff_keys = [(h['file'], h['header']) for h in diff_hunks]
    dupes = {k for k in diff_keys if diff_keys.count(k) > 1}
    if dupes:
        errs.append(f'diff has repeated (file, header) pairs, cannot key uniquely: {sorted(dupes)}')
    missing = [k for k in diff_keys if k not in index]
    extra = [index[k] for k in index if k not in set(diff_keys)]
    for k in missing:
        errs.append(f'unclassified hunk: {k[0]} {k[1]}')
    for i in extra:
        errs.append(f'ledger entry {i} matches no hunk in the diff')
    return errs, len(diff_hunks), len(entries)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger'); ap.add_argument('--trainer', default='train_flagged.py')
    a = ap.parse_args()
    errs, nh, ne = check(a.ledger, a.trainer)
    for e in errs:
        print('FAIL', e)
    print(f'{a.ledger}: {nh} hunks in diff, {ne} ledger entries, {len(errs)} problems')
    sys.exit(1 if errs else 0)

if __name__ == '__main__':
    main()
