"""Static gate over the generated matrix: build every experiments/runs/*/config.json with the AST parameter
inventory (exercises model construction for every flag combination, including non-historical ones).
Usage: python -I tools/matrix_inventory.py > gates/matrix_inventory.txt"""
import glob, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from param_inventory import inventory, summarize
warnings.simplefilter('ignore', SyntaxWarning)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = 0
for p in sorted(glob.glob(os.path.join(HERE, 'experiments', 'runs', '*', 'config.json'))):
    name = p.split(os.sep)[-2]
    try:
        params, bufs = inventory(os.path.join(HERE, 'train_flagged.py'), p)
        t, g = summarize(params)
        print(f"{name:24s} {t:>13,d}  tensors={len(params):3d} buffers={len(bufs):2d}  "
              f"VE={g.get('value_embeddings', 0)/1e6:6.1f}M blocks={g.get('blocks', 0)/1e6:5.1f}M")
    except Exception as e:
        bad += 1
        print(f'{name:24s} FAIL {e!r}')
print(f'{bad} failures')
sys.exit(1 if bad else 0)
