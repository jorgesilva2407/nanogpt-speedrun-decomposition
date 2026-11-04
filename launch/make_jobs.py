"""Job lists for the all-cloud sweep: which (config, seed) runs belong to a tier, split across N parallel workers.

  python -I launch/make_jobs.py --tier p1                          # all jobs of tier p1, one "exp_dir name seed" per line
  python -I launch/make_jobs.py --tier p1 --workers 8 --worker 3   # the share of worker 3 of 8
  python -I launch/make_jobs.py --tier p1 --status --repo USER/speedrun-results   # which jobs are DONE in the HF repo

Why this exists: launch/run_sweep.sh adds the fixed points an arm needs on every host it runs on, and its host
bindings live in a per-checkout experiments/hosts.json. With many cloud instances of ONE GPU model and ONE image
(the study's single host class, label e.g. vast-h100) that would re-run fp_r11/fp_r18 on every instance. Here the
fixed points are their own tier and every job runs exactly once.

Assignment: jobs are sorted seed-major (all seed-0 jobs first, so a partial sweep is still a balanced design), longest
first within a seed, and given greedily to the least-loaded worker. Each worker runs its jobs in that order.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIERS = {   # arm ids per tier (decided 2026-11-04, OPEN_DECISIONS.md D13; fixed points are a tier of their own)
    'fixed_points': None,
    # both-direction arms + the arms with a primary-source prediction (A1a kernel, A3 value embeddings: H1 needs them)
    'p1': ['A1a', 'A2', 'A3', 'A4b', 'A6', 'A6b', 'A7', 'A8'],
    'p2': ['A1b', 'A4', 'A5'],          # add-one-in-only VE ladder rungs and the 64K layout
    'p3': ['A5b'],                      # #16 MFU tweaks: throughput is measured regardless; steps only if money remains
}
FAMILIES = ('aoi', 'loo', 'ctrl')          # the AdamW LR sweep is not run (study.yaml adamw_lr_sweep.bases: [])

def load():
    with open(os.path.join(HERE, 'experiments', 'matrix.json')) as f:
        m = json.load(f)
    out = []
    for c in m['configurations']:
        with open(os.path.join(HERE, 'experiments', 'runs', c['exp_id'], 'config.json')) as f:
            steps = json.load(f)['hparams']['num_iterations']
        out.append(dict(c, steps=steps))
    return out

def jobs_for(tier):
    cfgs = load()
    if tier == 'fixed_points':
        sel = [c for c in cfgs if c['family'] == 'fp']
    else:
        arms = set(TIERS[tier])
        sel = [c for c in cfgs if c['family'] in FAMILIES and c['arm'] in arms]
    jobs = [(c['steps'], os.path.join('experiments', 'runs', c['exp_id']), c['exp_id'], s) for c in sel for s in c['seeds']]
    return sorted(jobs, key=lambda j: (j[3], -j[0], j[2]))

def split(jobs, n):
    load_, share = [0] * n, [[] for _ in range(n)]
    for j in jobs:
        w = min(range(n), key=lambda k: (load_[k], k))
        share[w].append(j); load_[w] += j[0]
    return share, load_

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True, choices=sorted(TIERS))
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--status', action='store_true'); ap.add_argument('--repo', default=os.environ.get('RESULTS_REPO'))
    a = ap.parse_args()
    jobs = jobs_for(a.tier)
    if a.status:
        from huggingface_hub import HfApi
        files = set(HfApi().list_repo_files(a.repo, repo_type='dataset'))
        done = [j for j in jobs if f'{j[2]}/seed{j[3]}/DONE' in files]
        for j in jobs:
            print('DONE ' if j in done else 'todo ', j[2], j[3])
        print(f'{len(done)}/{len(jobs)} done', file=sys.stderr); return
    share, load_ = split(jobs, a.workers)
    assert 0 <= a.worker < a.workers
    for steps, exp_dir, name, seed in share[a.worker]:
        print(exp_dir, name, seed)
    print(f'# worker {a.worker}/{a.workers}: {len(share[a.worker])} jobs, {load_[a.worker]} steps '
          f'(max over workers {max(load_)} steps)', file=sys.stderr)

if __name__ == '__main__':
    main()
