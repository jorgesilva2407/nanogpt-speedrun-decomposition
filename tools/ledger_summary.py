"""Print a markdown summary of a ledger: hunks grouped by category, flags touched, coupled hparams.
Usage: python -I tools/ledger_summary.py ledger/rNN.yaml"""
import sys, yaml
from collections import defaultdict
led = yaml.safe_load(open(sys.argv[1]))
by_cat = defaultdict(list); flags = defaultdict(list); coupled = []
for h in led['hunks']:
    cats = h['category'] if isinstance(h['category'], list) else [h['category']]
    for c in cats:
        by_cat[c].append(h['id'])
    for f in (h['flag'] if isinstance(h['flag'], list) else [h['flag']]):
        flags[f].append(h['id'])
    coupled += [f"{h['id']}: {c}" for c in h.get('coupled', [])]
print(f"Hunks: {len(led['hunks'])} ({', '.join(f'{c} {len(v)}' for c, v in sorted(by_cat.items()))}; a bundled hunk counts in each of its categories)\n")
print('| category | hunks |\n|---|---|')
for c, v in sorted(by_cat.items()):
    print(f"| {c} | {', '.join(v)} |")
print('\n| flag / config key | hunks |\n|---|---|')
for f, v in sorted(flags.items(), key=lambda kv: (kv[0] == 'none', kv[0])):
    print(f"| `{f}` | {', '.join(v)} |")
if coupled:
    print('\nCoupled / co-tuned hyperparameters:\n')
    for c in coupled:
        print(f'* {c}')
