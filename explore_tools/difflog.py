"""diff embedded log source vs a commit's script. usage: difflog.py repo log_ref:path commit[:file]"""
import difflib, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logsrc import extract
repo, logspec, cspec = sys.argv[1:4]
g = lambda *a: subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True, check=True).stdout
src = extract(g('show', logspec))
if ':' not in cspec:
    for fn in ('train_gpt2.py', 'train_gpt.py'):
        r = subprocess.run(['git', '-C', repo, 'show', f'{cspec}:{fn}'], capture_output=True, text=True)
        if r.returncode == 0:
            cspec = f'{cspec}:{fn}'; break
script = g('show', cspec)
sys.stdout.writelines(difflib.unified_diff(script.splitlines(True), src.splitlines(True), cspec, 'LOG:' + logspec.split('/')[-1], n=int(os.environ.get('CTX', 1))))
