"""Markdown table: every master-ancestry commit touching the training script (2024-11-09..2025-01-05) and the
snapshot it belongs to.
RULE (deterministic): commit c belongs to the first record N (11..18) whose code BASE has c as ancestor-or-self.
  Bases: master-lineage snapshots use their code commit; snapshots whose code is not in git (r11, r15, r17) use
  the master commit their embedded source was built on (established by diffing, see EXPLORATION.md section 5):
    r11 <- 7c1a1af (+ external U-net hunk from @brendanh0gan's fork)   r12 <- ebce774   r13 <- 759b8fc
    r14 <- 49c3afa   r15 <- 155afd5 + 8e5bbd60 (PR #43) (+ external)  r16 <- 0690f2c   r17 <- 0690f2c (+ external)
    r18 <- 45544f5
VISIBILITY (heuristic, reported, not used for assignment): share of the commit's non-trivial added lines present
in snapshot N and of its removed lines absent from N. Low visibility = superseded/reverted before N."""
import subprocess, sys
repo, srcdir = sys.argv[1:3]
g = lambda *a: subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True, errors='replace')
BASES = {11: ['7c1a1af'], 12: ['ebce774'], 13: ['759b8fc'], 14: ['49c3afa'], 15: ['155afd5', '8e5bbd60'],
         16: ['0690f2c'], 17: ['0690f2c'], 18: ['45544f5']}
snap = {n: set(l.strip() for l in open(f'{srcdir}/r{n}.log.py').read().split('\n')) for n in range(11, 19)}
fp = set(g('rev-list', '--first-parent', 'master').stdout.split())
anc = lambda c, b: g('merge-base', '--is-ancestor', c, b).returncode == 0
print('| commit | date (author) | path | author | +/- | belongs to | visibility in that snapshot | subject |')
print('|---|---|---|---|---|---|---|---|')
counts = {}
log = g('log', '--reverse', '--topo-order', '--no-merges', '--full-history', '--format=%H\t%h\t%aI\t%an\t%s', '--since=2024-11-09',
        '--until=2025-01-06', 'master', '--', 'train_gpt2.py', 'train_gpt.py').stdout.strip().split('\n')
for line in log:
    H, h, date, author, subj = line.split('\t')
    N = next((n for n in range(11, 19) if any(anc(H, b) for b in BASES[n])), None)
    diff = g('show', '--format=', '--unified=0', '-M', H, '--', 'train_gpt2.py', 'train_gpt.py').stdout
    add = [l[1:].strip() for l in diff.split('\n') if l.startswith('+') and not l.startswith('+++')]
    rem = [l[1:].strip() for l in diff.split('\n') if l.startswith('-') and not l.startswith('---')]
    sa = [l for l in add if len(l) > 3 and not l.startswith('#')]
    sr = [l for l in rem if len(l) > 3 and not l.startswith('#')]
    if N is None:
        vis = 'n/a'
    elif not sa and not sr:
        vis = 'comment/blank-only'
    else:
        pa = f'{int(100*sum(l in snap[N] for l in sa)/len(sa))}% added kept' if sa else 'no adds'
        pr = f'{int(100*sum(l not in snap[N] for l in sr)/len(sr))}% removed gone' if sr else 'no removals'
        vis = f'{pa}; {pr}'
    key = f'r{N}' if N else 'none (after r18 base)'
    counts[key] = counts.get(key, 0) + 1
    print(f'| `{h}` | {date[:16]} | {"first-parent" if H in fp else "side (merged)"} | {author} | +{len(add)}/-{len(rem)} | **{key}** | {vis} | {subj[:48].replace("|", "/")} |')
print(f'\nCounts: {dict(sorted(counts.items()))}')
