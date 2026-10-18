"""Walk README.md history (all of master's ancestry, oldest first) and track rows for records #11-#18,
keyed by record time. Prints first appearance + every textual change of each row until 2025-02-01.
Usage: readme_rows.py <repo> <out.json>"""
import json, re, subprocess, sys
repo, out = sys.argv[1:3]
g = lambda *a: subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True, errors='replace').stdout
KEYS = {11: r'7\.2 minutes|7\.23 minutes', 12: r'5\.0\d? minutes', 13: r'4\.66 minutes', 14: r'4\.41 minutes',
        15: r'3\.95 minutes', 16: r'3\.80? minutes', 17: r'3\.57 minutes', 18: r'3\.4 minutes'}
commits = [l.split('\t') for l in g('log', '--reverse', '--topo-order', '--format=%h\t%aI\t%an\t%s', '--until=2025-02-01', 'master', '--', 'README.md').strip().split('\n')]
hist = {k: [] for k in KEYS}
for sha, date, author, subj in commits:
    txt = g('show', f'{sha}:README.md')
    for k, pat in KEYS.items():
        rows = [l.strip() for l in txt.split('\n') if re.search(pat, l) and ('|' in l)]
        row = rows[0] if rows else None
        if not hist[k] or hist[k][-1]['row'] != row:
            hist[k].append(dict(sha=sha, date=date, author=author, subject=subj, row=row))
json.dump(hist, open(out, 'w'), indent=1)
for k, h in hist.items():
    print(f'#### record {k}')
    for e in h:
        print(f"  {e['sha']} {e['date']} {e['author']} | {e['subject'][:40]}\n      {e['row']}")
