"""Run one (configuration, seed) atomically on this host.

  python launch/run_one.py --exp experiments/runs/aoi_A3 --seed 0 --host lab-a100 --nproc 1 \
         [--name aoi_A3] [--docker-tag modded-nanogpt-flagged:<sha>] [--dry-run]

* results/<name>/seed<k>/ receives metrics.jsonl, timing.jsonl, meta.json and DONE (written by the trainer).
* If DONE exists the job is skipped. If the directory exists without DONE it is moved aside to
  seed<k>.incomplete-<timestamp> and the run starts from step 0: there is no resume.
* Refuses a dirty git tree for real runs (the trainer refuses too, via the run section); --dry-run only prints.
* The resolved config (preset values + run section: exp_id, config_hash, git SHA/dirty, docker tag, host label) is
  written to results/<name>/seed<k>/resolved.json and passed to the trainer as JSON.
"""
import argparse, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'experiments'))
from matrix import config_hash

def git(*a):
    return subprocess.run(['git', '-C', HERE, *a], capture_output=True, text=True).stdout.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='experiments/runs/<exp_id>')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--host', required=True, help='host label, e.g. lab-a100 or rented-h100')
    ap.add_argument('--nproc', type=int, default=1, help='GPUs on this host (must divide 8)')
    ap.add_argument('--name', default=None, help='result name (defaults to the exp_id; host-qualified for baseline replicas)')
    ap.add_argument('--docker-tag', default=os.environ.get('DOCKER_TAG'))
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='extra trainer args (e.g. --throughput-steps 300)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    with open(os.path.join(HERE, a.exp, 'config.json')) as f:
        cfg = json.load(f)
    assert a.seed in cfg['seeds'] or a.dry_run, f"seed {a.seed} not planned for {cfg['exp_id']} ({cfg['seeds']})"
    assert 8 % a.nproc == 0, 'nproc must divide 8 (8 micro-batches of 65,536 tokens per step)'
    name = a.name or cfg['exp_id']
    out = os.path.join(a.results, name, f'seed{a.seed}')
    if os.path.exists(os.path.join(out, 'DONE')):
        print(f'skip: {out} is complete'); return
    dirty = bool(git('status', '--porcelain', '--untracked-files=no'))
    if dirty and not a.dry_run:
        raise SystemExit('refusing a real run from a dirty tree: commit first (results must map to a commit)')
    if not a.docker_tag and not a.dry_run:
        raise SystemExit('set --docker-tag or DOCKER_TAG (recorded in meta.json)')
    resolved = {k: cfg[k] for k in ('preset', 'flags', 'hparams', 'measurement')}
    resolved['run'] = dict(exp_id=cfg['exp_id'], result_name=name, config_hash=config_hash(cfg), git_sha=git('rev-parse', 'HEAD'),
                           git_dirty=dirty, allow_dirty=a.dry_run, docker_tag=a.docker_tag, host_label=a.host,
                           provenance=cfg['provenance'])
    assert resolved['run']['config_hash'] == cfg['config_hash'], 'config changed since matrix.py generated it'
    cmd = ['torchrun', '--standalone', f'--nproc_per_node={a.nproc}', os.path.join(HERE, 'train_flagged.py'),
           '--config', os.path.join(out, 'resolved.json'), '--seed', str(a.seed), '--out-dir', out, *a.extra]
    if a.dry_run:
        print('DRY RUN:', ' '.join(cmd)); print(json.dumps(resolved['run'], indent=1)); return
    if os.path.isdir(out):
        aside = f"{out}.incomplete-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(out, aside)
        print(f'moved incomplete run aside: {aside}')
    os.makedirs(out)
    with open(os.path.join(out, 'resolved.json'), 'w') as f:
        json.dump(resolved, f, indent=1, sort_keys=True)
    rc = subprocess.call(cmd, cwd=HERE)
    if rc != 0 or not os.path.exists(os.path.join(out, 'DONE')):
        raise SystemExit(f'run failed (rc={rc}); no DONE in {out}: re-run it, do not resume')

if __name__ == '__main__':
    main()
