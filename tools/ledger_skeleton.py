"""Write a ledger skeleton (one TODO entry per hunk) for a diff; check_ledger.py fails until filled in.
Usage: python -I tools/ledger_skeleton.py diffs/rNN-vs-rMM.diff NN MM > ledger/rNN.yaml"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffparse import hunks

diff, nn, mm = sys.argv[1:4]
hs = hunks(open(diff).read())
print(f'record: {nn}\nsnapshot_from: r{mm}\nsnapshot_to: r{nn}\ndiff: {diff}\nhunks:')
for i, h in enumerate(hs, 1):
    changed = [l for l in h['body'] if l[:1] in '+-'][:6]
    for l in changed:
        print(f'  # {l[:150]}')
    print(f'''  - id: H{i:03d}
    file: {h['file']}
    header: "{h['header']}"
    description: TODO
    category: TODO
    flag: TODO
    provenance: TODO
    bundled: TODO
    numerics: TODO''')
