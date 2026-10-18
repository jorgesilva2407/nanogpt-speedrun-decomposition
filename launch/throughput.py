"""Throughput measurement: a fixed number of steady-state steps with a per-step trace, on an otherwise-idle node.

  python launch/throughput.py --exp experiments/runs/fp_r18 --host lab-a100 --nproc 1 [--steps 300]

* Asks (interactively) whether the node is otherwise idle and records the answer; refuses to run unless the answer is
  'yes' (proposal 3.1 rule 3). nvidia-smi is captured before and after.
* Output: results/throughput/<exp_id>/<host>/<timestamp>/ with timing.jsonl (every step, compile warm-up marked),
  metrics.jsonl, meta.json, idle.json, nvidia-smi-{before,after}.txt, DONE. Host-tagged: never pooled across hosts
  (analysis/load.py refuses).
* Seed 0; the training path is the configuration's own (no validation).
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def smi():
    r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    return r.stdout + r.stderr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True); ap.add_argument('--host', required=True)
    ap.add_argument('--nproc', type=int, default=1); ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--docker-tag', default=os.environ.get('DOCKER_TAG'))
    a = ap.parse_args()
    with open(os.path.join(HERE, 'experiments', 'study.yaml')) as f:
        import yaml
        study = yaml.safe_load(f)
    steps = a.steps or study['throughput']['steps']
    with open(os.path.join(HERE, a.exp, 'config.json')) as f:
        exp_id = json.load(f)['exp_id']
    answer = input('Is this node otherwise idle (no other jobs on ANY of its GPUs, no heavy CPU/IO jobs)? [yes/no] ').strip().lower()
    out_root = os.path.join(HERE, 'results', 'throughput', exp_id, a.host, time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, 'idle.json'), 'w') as f:
        json.dump(dict(answer=answer, asked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), host=a.host), f)
    if answer != 'yes':
        raise SystemExit('refusing: throughput is only measured on an otherwise-idle node (answer recorded in idle.json)')
    open(os.path.join(out_root, 'nvidia-smi-before.txt'), 'w').write(smi())
    run_dir = os.path.join(out_root, 'run')
    cmd = [sys.executable, os.path.join(HERE, 'launch', 'run_one.py'), '--exp', a.exp, '--seed', '0', '--host', a.host,
           '--nproc', str(a.nproc), '--name', os.path.relpath(run_dir, os.path.join(HERE, 'results')),
           '--extra', '--throughput-steps', str(steps)]
    env = dict(os.environ, DOCKER_TAG=a.docker_tag or '')
    rc = subprocess.call(cmd, env=env)
    open(os.path.join(out_root, 'nvidia-smi-after.txt'), 'w').write(smi())
    sys.exit(rc)

if __name__ == '__main__':
    main()
