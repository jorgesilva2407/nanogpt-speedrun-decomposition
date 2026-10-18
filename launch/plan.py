"""Host rules and job planning for sweeps (pure Python).

Rules enforced (proposal 3.1):
  * a run is atomic: (exp_id, seed) either has DONE or is re-run from scratch;
  * an arm's comparison pair (its LOO with fp_r18, its AOI with fp_r11, A8's control/sweep with both fixed points) and
    all its seeds run on ONE host. Arms are bound to hosts in experiments/hosts.json; a sweep refuses to run an arm on
    a second host. Fixed points are needed on every host that runs an arm: the first host to claim them uses the plain
    exp_id (fp_r11, fp_r18); other hosts get host-qualified replicas (fp_r11__on_<host>) that are analysed only
    against arms of that host.
Usage (called by launch/run_sweep.sh):
  python -I launch/plan.py --host lab-a100 --arm A3 --arm A7 [--claim]   -> prints "exp_dir seed" lines, one per job
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSTS = os.path.join(HERE, 'experiments', 'hosts.json')
RESULTS = os.path.join(HERE, 'results')

def load_hosts():
    if os.path.exists(HOSTS):
        with open(HOSTS) as f:
            return json.load(f)
    return dict(arms={}, fixed_points={})

def save_hosts(h):
    with open(HOSTS, 'w') as f:
        json.dump(h, f, indent=1, sort_keys=True)

def matrix():
    with open(os.path.join(HERE, 'experiments', 'matrix.json')) as f:
        return json.load(f)

def result_host(exp_id, seed):
    meta = os.path.join(RESULTS, exp_id, f'seed{seed}', 'meta.json')
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f).get('host_label')
    return None

def plan(host, arm_ids, claim=False):
    m = matrix()
    h = load_hosts()
    errors, jobs = [], []
    cfgs = {c['exp_id']: c for c in m['configurations']}
    for a in arm_ids:
        owner = h['arms'].get(a)
        if owner is not None and owner != host:
            errors.append(f'arm {a} is bound to host {owner}; refusing to split its comparisons/seeds onto {host}')
    arm_exps = [c for c in m['configurations'] if c['arm'] in arm_ids and c['family'] in ('aoi', 'loo', 'ctrl', 'sweep', 'rq3b')]
    for c in arm_exps:
        for s in c['seeds']:
            rh = result_host(c['exp_id'], s)
            if rh not in (None, host):
                errors.append(f"{c['exp_id']} seed {s} already ran on {rh}; refusing to add seeds on {host}")
    needed_fp = set()
    for c in arm_exps:
        fam = c['family']
        if fam == 'loo':
            needed_fp.add('fp_r18')
        elif fam == 'aoi':
            needed_fp.add('fp_r11')
        elif fam in ('ctrl', 'sweep'):
            needed_fp |= {'fp_r11', 'fp_r18'}
        elif fam == 'rq3b':
            base = 'fp_r18' if c['hparams_from'] == 'r18' else 'fp_r11'
            needed_fp.add(f"rq3b_s{c['exp_id'].split('_')[1][1:]}_{base}")
    if errors:
        return errors, []
    fp_jobs = []
    for fp in sorted(needed_fp):
        fp_owner = h['fixed_points'].get(fp)
        if fp_owner in (None, host):
            name = fp
            if claim:
                h['fixed_points'][fp] = host
        else:
            name = f'{fp}__on_{host}'           # host-qualified replica of the baseline
        fp_jobs.append((fp, name))
    for fp, name in fp_jobs:
        src = cfgs.get(fp)
        if src is None:
            errors.append(f'baseline {fp} is not in the matrix (generate RQ3b first)')
            continue
        for s in src['seeds']:
            jobs.append((os.path.join('experiments', 'runs', fp), name, s))
    for c in arm_exps:
        for s in c['seeds']:
            jobs.append((os.path.join('experiments', 'runs', c['exp_id']), c['exp_id'], s))
    if claim and not errors:
        for a in arm_ids:
            h['arms'][a] = host
        save_hosts(h)
    return errors, jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True); ap.add_argument('--arm', action='append', required=True)
    ap.add_argument('--claim', action='store_true', help='record the arm -> host binding in experiments/hosts.json')
    a = ap.parse_args()
    errors, jobs = plan(a.host, a.arm, a.claim)
    if errors:
        for e in errors:
            print('REFUSED:', e, file=sys.stderr)
        sys.exit(2)
    for exp_dir, name, s in jobs:
        print(exp_dir, name, s)

if __name__ == '__main__':
    main()
