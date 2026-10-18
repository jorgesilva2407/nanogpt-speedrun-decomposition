"""For every log in records #11-#18, find which first-parent master commit's training script
matches the embedded source (exact, whitespace-only, or nearest by diff size).
Usage: python -I match_logs.py <upstream_repo> <out.json>
"""
import difflib, hashlib, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logsrc import extract, norm_ws

REPO, OUT = sys.argv[1], sys.argv[2]
REC_REF = '5deb47d'  # README + records layout before the 2025-07-27 folder rename
DIRS = ['111024_UNetDoubleLr', '111924_FlexAttention', '112424_WindowWarmup', '120424_ValueEmbed',
        '120824_UNetValueEmbedsTweaks', '121024_MFUTweaks', '121724_SparsifyEmbeds', '010425_SoftCap']


def git(*a):
    return subprocess.run(['git', '-C', REPO, *a], capture_output=True, text=True, check=True).stdout


def git_ok(*a):
    r = subprocess.run(['git', '-C', REPO, *a], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


# candidate script versions: every first-parent commit 2024-11-01..2025-01-31, both filenames
SCOPE = sys.argv[3] if len(sys.argv) > 3 else 'first-parent'
scope_args = ['--first-parent', 'master'] if SCOPE == 'first-parent' else ['--all']
commits = git('log', *scope_args, '--reverse', '--format=%h %ad', '--date=short',
              '--since=2024-11-01', '--until=2025-02-01').split('\n')
commits = [c.split() for c in commits if c.strip()]
versions = {}  # blob sha -> dict(first commit, filename, text)
order = []
for sha, date in commits:
    for fn in ('train_gpt2.py', 'train_gpt.py'):
        b = git_ok('rev-parse', f'{sha}:{fn}')
        if b is None:
            continue
        b = b.strip()
        if b not in versions:
            versions[b] = dict(first=sha, date=date, file=fn, text=git('show', f'{sha}:{fn}'), commits=[sha])
            order.append(b)
        else:
            versions[b]['commits'].append(sha)

norm_index = {}
for b in order:
    norm_index.setdefault(norm_ws(versions[b]['text']), []).append(b)

res = {}
for d in DIRS:
    files = [f for f in git('ls-tree', '-r', '--name-only', REC_REF, f'records/{d}').split('\n') if f.endswith('.txt')]
    groups = {}
    for f in files:
        txt = git('show', f'{REC_REF}:{f}')
        try:
            src = extract(txt)
        except ValueError:
            groups.setdefault('NO_SOURCE', []).append(f)
            continue
        h = hashlib.sha1(src.encode()).hexdigest()[:12]
        g = groups.setdefault(h, dict(files=[], src=src))
        g['files'].append(os.path.basename(f))
    out = {}
    for h, g in groups.items():
        if h == 'NO_SOURCE':
            out[h] = g
            continue
        src = g['src']
        exact = [b for b in order if versions[b]['text'] == src]
        ws = norm_index.get(norm_ws(src), [])
        # nearest by changed-line count
        best = None
        for b in order:
            sm = difflib.SequenceMatcher(None, versions[b]['text'].split('\n'), src.split('\n'), autojunk=False)
            changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal')
            if best is None or changed < best[0]:
                best = (changed, b)
        out[h] = dict(n_logs=len(g['files']), files=sorted(g['files'])[:5], lines=src.count('\n'),
                      exact=[(versions[b]['first'], versions[b]['date'], versions[b]['file']) for b in exact],
                      ws_only=[(versions[b]['first'], versions[b]['date'], versions[b]['file']) for b in ws],
                      nearest=dict(changed_lines=best[0], first=versions[best[1]]['first'],
                                   date=versions[best[1]]['date'], file=versions[best[1]]['file'],
                                   commits_with_blob=versions[best[1]]['commits']))
    res[d] = out
json.dump(dict(versions=[(versions[b]['first'], versions[b]['date'], versions[b]['file'], b[:10],
                          len(versions[b]['commits'])) for b in order], records=res),
          open(OUT, 'w'), indent=1)
print('script versions:', len(order))
for d, out in res.items():
    print('##', d)
    for h, g in out.items():
        if h == 'NO_SOURCE':
            print('   NO_SOURCE', len(g)); continue
        print(f"   src {h}: logs={g['n_logs']} lines={g['lines']} exact={g['exact'][:3]} ws={g['ws_only'][:3]} "
              f"nearest={g['nearest']['first']}/{g['nearest']['date']}/{g['nearest']['file']} Δ={g['nearest']['changed_lines']}")
