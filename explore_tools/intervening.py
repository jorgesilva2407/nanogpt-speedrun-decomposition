"""For every master commit (full ancestry, topo order) touching the training script between 2024-11-09 and
2025-01-05, compute which record snapshot (embedded log sources r11..r18) its ADDED lines first survive into.
Heuristic evidence only: 'first snapshot containing >=60% of the commit's non-trivial added lines'.
Usage: intervening.py <repo> <srcdir with rNN.log.py> <out.json>"""
import json, re, subprocess, sys
repo, srcdir, out = sys.argv[1:4]
g = lambda *a: subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True, errors='replace').stdout
snap = {n: set(l.strip() for l in open(f'{srcdir}/r{n}.log.py').read().split('\n')) for n in range(11, 19)}
fp = set(g('rev-list', '--first-parent', 'master').split())
rows = []
for line in g('log', '--reverse', '--topo-order', '--no-merges', '--format=%H\t%h\t%aI\t%an\t%s',
              '--since=2024-11-09', '--until=2025-01-06', 'master', '--', 'train_gpt2.py', 'train_gpt.py').strip().split('\n'):
    H, h, date, author, subj = line.split('\t')
    diff = g('show', '--format=', '--unified=0', H, '--', 'train_gpt2.py', 'train_gpt.py')
    added = [l[1:].strip() for l in diff.split('\n') if l.startswith('+') and not l.startswith('+++')]
    removed = [l[1:].strip() for l in diff.split('\n') if l.startswith('-') and not l.startswith('---')]
    sig = [l for l in added if len(l) > 3 and not l.startswith('#')]
    frac = {n: (sum(l in snap[n] for l in sig) / len(sig)) if sig else None for n in snap}
    first = next((n for n in range(11, 19) if sig and frac[n] >= 0.6), None)
    only_comments = bool(added or removed) and all((not l) or l.startswith('#') for l in added + removed)
    rows.append(dict(sha=h, date=date, author=author, subject=subj, first_parent=H in fp, plus=len(added), minus=len(removed),
                     sig_added=len(sig), only_comments_or_blank=only_comments,
                     frac={k: (round(v, 2) if v is not None else None) for k, v in frac.items()}, first_snapshot=first))
json.dump(rows, open(out, 'w'), indent=1)
for r in rows:
    fr = ' '.join(f"{k}:{'-' if v is None else int(100*v)}" for k, v in r['frac'].items())
    print(f"{r['sha']} {r['date'][:16]} {'FP' if r['first_parent'] else 'sd'} {r['author'][:14]:14} +{r['plus']}/-{r['minus']} -> r{r['first_snapshot']} | {fr} | {r['subject'][:38]}")
