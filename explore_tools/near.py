"""changed-line count between a log's embedded source and the script at each listed commit"""
import difflib, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logsrc import extract
repo, logspec = sys.argv[1:3]; commits = sys.argv[3:]
g = lambda *a: subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True)
src = extract(g('show', logspec).stdout).split('\n')
for c in commits:
    for fn in ('train_gpt2.py', 'train_gpt.py'):
        r = g('show', f'{c}:{fn}')
        if r.returncode == 0: break
    else:
        print(c, 'no script'); continue
    sm = difflib.SequenceMatcher(None, r.stdout.split('\n'), src, autojunk=False)
    ch = sum(max(i2-i1, j2-j1) for op,i1,i2,j1,j2 in sm.get_opcodes() if op!='equal')
    print(c, fn, g('log','-1','--format=%ad %s','--date=iso',c).stdout.strip(), 'changed=', ch)
