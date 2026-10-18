"""Per-log runtime metadata for records #11-#18 (torch version, nvidia-smi date, #GPUs, final val loss/time)."""
import json, re, subprocess, sys, collections
repo, out = sys.argv[1:3]
g = lambda *a: subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True, errors='replace').stdout
DIRS = ['111024_UNetDoubleLr', '111924_FlexAttention', '112424_WindowWarmup', '120424_ValueEmbed',
        '120824_UNetValueEmbedsTweaks', '121024_MFUTweaks', '121724_SparsifyEmbeds', '010425_SoftCap']
res = {}
for d in DIRS:
    rows = []
    for f in [x for x in g('ls-tree', '-r', '--name-only', '5deb47d', f'records/{d}').split('\n') if x.endswith('.txt')]:
        t = g('show', f'5deb47d:{f}')
        tv = next((m for m in re.finditer(r'Running [pP]y[tT]orch (\S+)', t) if '{' not in m.group(1)), None)
        dt = re.search(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\w+ +\d+ [\d:]+ \d{4})', t, re.M)
        ngpu = len(re.findall(r'NVIDIA H100', t)); gpu = re.search(r'\|\s+\d\s+(NVIDIA [^|]+?)\s{2,}', t)
        vals = re.findall(r'step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms', t)
        last = vals[-1] if vals else None
        rows.append(dict(file=f.split('/')[-1], torch=tv.group(1) if tv else None, date=dt.group(2) if dt else None,
                         ngpu_lines=ngpu, gpu=gpu.group(1).strip() if gpu else None,
                         final_step=int(last[0]) if last else None, num_iter=int(last[1]) if last else None,
                         final_val=float(last[2]) if last else None, train_ms=int(last[3]) if last else None,
                         val_every=(int(vals[1][0]) - int(vals[0][0])) if len(vals) > 1 else None))
    res[d] = rows
json.dump(res, open(out, 'w'), indent=1)
for d, rows in res.items():
    tv = collections.Counter(r['torch'] for r in rows); dates = sorted(r['date'] for r in rows if r['date'])
    vals = [r['final_val'] for r in rows if r['final_val']]
    print(f"{d}: n={len(rows)} torch={dict(tv)} gpus={collections.Counter(r['ngpu_lines'] for r in rows)} gpu={rows[0]['gpu']}")
    print(f"    dates {dates[0] if dates else None} .. {dates[-1] if dates else None}; num_iter={collections.Counter(r['num_iter'] for r in rows)} val_every={collections.Counter(r['val_every'] for r in rows)}")
    print(f"    final val mean={sum(vals)/len(vals):.4f} min={min(vals)} max={max(vals)}; train_ms first={rows[0]['train_ms']}")
