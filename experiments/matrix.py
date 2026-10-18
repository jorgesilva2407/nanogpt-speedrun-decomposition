"""Derive the experiment matrix from experiments/arms.yaml, experiments/study.yaml and the two fixed-point presets.

Nothing is hand-listed. Families of configurations:
  fp      the two fixed points (r11 = floor, r18 = ceiling)
  loo     leave-one-out: r18 with one arm switched back to its r11 values; keeps r18's hyperparameters
  aoi     add-one-in: r11 with one arm switched to its r18 values; keeps r11's hyperparameters. Prerequisite switches
          (trainer REQUIRES) are added and recorded, which makes the VE ladder (and every arm that needs FlexAttention
          or the 64K layout) cumulative.
  ctrl    arm A8 (Muon -> AdamW) applied to both fixed points
  sweep   AdamW learning-rate sweep for A8 (values and bases from study.yaml)
  rq3b    budget-scaled copies (num_iterations x scale, absolute-step hparams rescaled) of fp/loo/aoi for resolved arms
A leave-one-out configuration that would switch off a prerequisite of a switch that stays on is UNDEFINED: no
configuration is emitted; the open decision and its options are reported (OPEN_DECISIONS.md). Nothing is chosen.

Usage:
  python -I experiments/matrix.py                 # summary + write experiments/runs/<exp_id>/config.json, matrix.json
  python -I experiments/matrix.py --dry-run       # summary only
  python -I experiments/matrix.py --assume-resolved all   # preview RQ3b counts as if every arm were resolved
"""
import argparse, ast, copy, hashlib, json, math, os, shutil, sys
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_yaml(p):
    with open(os.path.join(HERE, p)) as f:
        return yaml.safe_load(f)

def trainer_registries():
    tree = ast.parse(open(os.path.join(HERE, 'train_flagged.py')).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('FLAGS', 'REQUIRES', 'FLAG_CHOICES'):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out['FLAGS'], out['REQUIRES'], out['FLAG_CHOICES']

def config_hash(cfg):
    # identical to train_flagged.config_hash (checked by experiments/tests)
    blob = json.dumps({k: cfg.get(k) for k in ('flags', 'hparams', 'measurement')}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()

def violations(flags, requires):
    return [(f, r) for f, reqs in requires.items() if flags.get(f) for r in reqs if not flags.get(r)]

def transitive_dependants(switches, flags, requires):
    """Flags that are on and (transitively) require any of `switches`."""
    out, changed = set(), True
    while changed:
        changed = False
        for f, reqs in requires.items():
            if flags.get(f) and f not in out and f not in switches and any(r in switches or r in out for r in reqs):
                out.add(f); changed = True
    return sorted(out)

class Matrix:
    def __init__(self, study, arms, floor, ceiling, registries):
        self.study, self.arms, self.floor, self.ceiling = study, arms, floor, ceiling
        self.FLAGS, self.REQUIRES, self.CHOICES = registries
        self.configs, self.open_items = [], []
        self.owner = {f: a['id'] for a in arms for f in (a.get('flags') or {})}

    # ------------------------------------------------------------------ helpers
    def arm_units(self):
        """Arms, or separable sub-arms when split_subarms is on."""
        units = []
        for a in self.arms:
            subs = a.get('subarms') or []
            if self.study.get('split_subarms') and subs:
                for s in subs:
                    units.append(dict(a, id=s['id'], name=f"{a['name']} [{s['id']}]",
                                      flags={f: a['flags'][f] for f in s['flags']}, parent=a['id']))
            else:
                units.append(a)
        return units

    def set_unit(self, cfg, unit, side):
        idx = 0 if side == 'from' else 1
        for f, vals in (unit.get('flags') or {}).items():
            cfg['flags'][f] = vals[idx]
        for k, vals in (unit.get('config') or {}).items():
            cfg['hparams'][k] = vals[idx]

    def base(self, which):
        src = self.floor if which == 'r11' else self.ceiling
        cfg = dict(preset=src['preset'], flags=copy.deepcopy(src['flags']), hparams=copy.deepcopy(src['hparams']),
                   measurement=copy.deepcopy(src['measurement']))
        mode = self.study['validation']['mode']
        if mode == 'subset20':
            cfg['measurement'].update(val_every_final_third=5, val_final_third_fraction=0.2)
        elif mode != 'v8':
            raise SystemExit(f'unknown validation mode {mode}')
        return cfg

    def seeds_for(self, arm_id, family):
        s = self.study['seeds']
        rq2 = set(self.study.get('rq2_arms') or [])
        if family in ('fp',):
            n = s['rq2'] if rq2 else s['default']
        elif arm_id is not None and (arm_id in rq2 or (arm_id.split('.')[0] in rq2)):
            n = s['rq2']
        else:
            n = s['default']
        return list(range(s['first_seed'], s['first_seed'] + n))

    def add(self, exp_id, cfg, provenance, family, arm_id=None):
        v = violations(cfg['flags'], self.REQUIRES)
        assert not v, f'{exp_id}: prerequisite violation {v}'
        assert set(cfg['flags']) == set(self.FLAGS), f'{exp_id}: flags incomplete'
        cfg = dict(cfg, exp_id=exp_id, family=family, arm=arm_id, provenance=provenance,
                   seeds=self.seeds_for(arm_id, family))
        cfg['config_hash'] = config_hash(cfg)
        self.configs.append(cfg)

    # ------------------------------------------------------------------ families
    def build(self, assume_resolved=None):
        self.add('fp_r11', self.base('r11'), dict(base_preset='r11', hparams_from='r11'), 'fp')
        self.add('fp_r18', self.base('r18'), dict(base_preset='r18', hparams_from='r18'), 'fp')
        for u in self.arm_units():
            if u.get('direction') == 'removal_from_both':
                continue
            self.build_aoi(u)
            self.build_loo(u)
        self.build_control()
        self.build_rq3b(assume_resolved)

    def build_aoi(self, u):
        cfg = self.base('r11')
        self.set_unit(cfg, u, 'to')
        added = []
        changed = True
        while changed:                      # close over prerequisites, recording which were added
            changed = False
            for f, r in violations(cfg['flags'], self.REQUIRES):
                cfg['flags'][r] = True
                added.append(dict(flag=r, arm=self.owner.get(r), required_by=f))
                changed = True
        self.add(f"aoi_{u['id']}", cfg, dict(base_preset='r11', hparams_from='r11', arm=u['id'], switched_to='r18 values',
                                             prerequisites_added=added, cumulative=bool(added)), 'aoi', u['id'])

    def build_loo(self, u):
        cfg = self.base('r18')
        switches = list(u.get('flags') or {})
        self.set_unit(cfg, u, 'from')
        v = violations(cfg['flags'], self.REQUIRES)
        if v:
            dependants = transitive_dependants(switches, self.ceiling['flags'], self.REQUIRES)
            self.open_items.append(dict(
                id=f"LOO-{u['id']}", arm=u['id'], name=u['name'],
                meaning=(f"r18 minus {u['id']} is undefined: switches that stay on in r18 require it "
                         f"({', '.join(f'{a} requires {b}' for a, b in v)})."),
                dependant_switches=dependants,
                dependant_arms=sorted({self.owner[f] for f in dependants if f in self.owner} - {u['id']}),
                status='open decision: no configuration emitted'))
            return
        superseded = []
        if u['id'].startswith('A4b') and cfg['flags'].get('block_window') and 'blockmask_manual' in switches:
            superseded.append('blockmask_manual has no effect while block_window is on (block_window builds its own mask)')
        self.add(f"loo_{u['id']}", cfg, dict(base_preset='r18', hparams_from='r18', arm=u['id'], switched_to='r11 values',
                                             superseded_notes=superseded), 'loo', u['id'])

    def build_control(self):
        ctrl = next(a for a in self.arms if a.get('direction') == 'removal_from_both')
        for b in ('r18', 'r11'):
            cfg = self.base(b)
            self.set_unit(cfg, ctrl, 'to')
            self.add(f"ctrl_{ctrl['id']}_{b}", cfg, dict(base_preset=b, hparams_from=b, arm=ctrl['id'],
                                                         note='Muon is in both fixed points; both configurations remove it'),
                     'ctrl', ctrl['id'])
        sw = self.study['adamw_lr_sweep']
        default = self.ceiling['hparams']['adamw_lr']
        for b in sw['bases']:
            for lr in sw['values']:
                if lr == default:
                    continue                # already the ctrl configuration
                cfg = self.base(b)
                self.set_unit(cfg, ctrl, 'to')
                cfg['hparams']['adamw_lr'] = lr
                self.add(f"sweep_{ctrl['id']}_lr{lr}_{b}", cfg, dict(base_preset=b, hparams_from=b, arm=ctrl['id'],
                                                                     adamw_lr=lr), 'sweep', ctrl['id'])

    def build_rq3b(self, assume_resolved):
        r = self.study['rq3b']
        resolved = list(r.get('resolved_arms') or [])
        if assume_resolved == 'all':
            resolved = [c['arm'] for c in self.configs if c['family'] in ('aoi', 'loo')]
        resolved = sorted(set(resolved))
        if not resolved:
            return
        base_cfgs = [c for c in self.configs if c['family'] == 'fp' or (c['family'] in ('aoi', 'loo') and c['arm'] in resolved)]
        for scale in r['budget_scales']:
            for c in base_cfgs:
                cfg = dict(preset=c['preset'], flags=copy.deepcopy(c['flags']), hparams=copy.deepcopy(c['hparams']),
                           measurement=copy.deepcopy(c['measurement']))
                n0 = cfg['hparams']['num_iterations']
                cfg['hparams']['num_iterations'] = n1 = round(n0 * scale)
                scaled = {'num_iterations': [n0, n1]}
                for k in r['absolute_step_hparams']:
                    v0 = cfg['hparams'][k]
                    if k == 'warmdown_iters' and cfg['hparams']['lr_schedule_form'] == 'frac':
                        continue            # unused under the fractional schedule
                    v1 = round(v0 * scale)
                    if v1 != v0:
                        cfg['hparams'][k] = v1
                        scaled[k] = [v0, v1]
                prov = dict(c['provenance'], rq3b_scale=scale, scaled_hparams=scaled, from_exp=c['exp_id'])
                self.add(f"rq3b_s{scale}_{c['exp_id']}", cfg, prov, 'rq3b', c['arm'])

    # ------------------------------------------------------------------ reporting
    def summary(self):
        fam = {}
        for c in self.configs:
            fam.setdefault(c['family'], []).append(c)
        runs = sum(len(c['seeds']) for c in self.configs)
        lines = ['configurations by family: ' + ', '.join(f'{k} {len(v)}' for k, v in fam.items()),
                 f'total configurations: {len(self.configs)}; total training runs (configs x seeds): {runs}']
        n_arms = len([u for u in self.arm_units() if u.get('direction') != 'removal_from_both'])
        lines.append(f'non-control arm units: {n_arms}; leave-one-out undefined (open decision): {len(self.open_items)} '
                     f"({', '.join(o['arm'] for o in self.open_items)})")
        return '\n'.join(lines)

RECONCILIATION = """\
Reconciliation with the proposal's counts (v8 section 3, step 3, and section 6.3):
  * "six flags ... 14 configurations including the two fixed points" = 6 arms x 2 directions + 2. This predates the
    arm table, which lists seven non-control arms (1-7) plus the control (8): 7 x 2 + 2 = 16. The 14 is stale; 16 is
    the count implied by the arm table (the budget's "14 configs x 3 seeds" inherits the stale 14).
  * The code generates a different number for two reasons, both reported rather than hidden:
    - more arms: the proposal's arms 1, 4, 5, 6 bundle sub-mechanisms that the replay could separate; arms.yaml lists
      them as A1a/A1b, A4/A4b, A5/A5b, A6/A6b (11 non-control units), plus the A8 control on both fixed points and
      the AdamW LR sweep;
    - fewer leave-one-outs: for arms that other switches in r18 require (prerequisites), leave-one-out is undefined
      and nothing is emitted until the group decides (OPEN_DECISIONS.md)."""

OPTIONS = """\
Options for an undefined leave-one-out (listed, not chosen; study.yaml loo_prerequisite_policy stays 'undecided'):
  O1 cumulative removal: r18 minus the arm AND every switch that (transitively) requires it. Always defined; it
     measures the joint contribution of the arm and its dependants. A per-arm figure would then be a difference of
     two cumulative LOOs (twice the variance), e.g. rung 3 = LOO(rungs 3-4) - LOO(rung 4).
  O2 separable parts only: remove only the arm's sub-switches that no remaining switch requires (e.g. A4's
     ve_lambdas2, A1b's doc_masking, A5's blockmask_separate/block_window); the rest of the arm has no LOO.
  O3 re-implement the dependants without the prerequisite (e.g. sparsified VE on a single chunked table, or the
     attention masks without FlexAttention). Not any record's code; changes what the dependent arm means; not built.
  O4 no leave-one-out for prerequisite arms: report add-one-in only and mark RQ2 as not estimable for them."""

def write(m, out_root):
    runs_dir = os.path.join(out_root, 'runs')
    if os.path.isdir(runs_dir):
        shutil.rmtree(runs_dir)
    os.makedirs(runs_dir)
    for c in m.configs:
        d = os.path.join(runs_dir, c['exp_id'])
        os.makedirs(d)
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump(c, f, indent=1, sort_keys=True)
    manifest = dict(
        generated_by='experiments/matrix.py', study=m.study,
        configurations=[dict(exp_id=c['exp_id'], family=c['family'], arm=c['arm'], seeds=c['seeds'], config_hash=c['config_hash'],
                             prerequisites_added=c['provenance'].get('prerequisites_added', []),
                             hparams_from=c['provenance'].get('hparams_from')) for c in m.configs],
        open_decisions=m.open_items,
        throughput_jobs=[dict(exp_id=c['exp_id'], steps=m.study['throughput']['steps'], host=m.study['throughput']['host'])
                         for c in m.configs if c['family'] in ('fp', 'aoi', 'loo', 'ctrl')],
        cross_hardware=m.study['cross_hardware'])
    with open(os.path.join(out_root, 'matrix.json'), 'w') as f:
        json.dump(manifest, f, indent=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--assume-resolved', choices=['all'], default=None, help='preview RQ3b as if every arm were resolved')
    ap.add_argument('--out', default=os.path.join(HERE, 'experiments'))
    a = ap.parse_args()
    study = load_yaml('experiments/study.yaml')
    arms = load_yaml('experiments/arms.yaml')['arms']
    m = Matrix(study, arms, load_yaml(study['presets']['floor']), load_yaml(study['presets']['ceiling']), trainer_registries())
    m.build(a.assume_resolved)
    print(m.summary())
    for o in m.open_items:
        print(f"  OPEN {o['id']}: {o['meaning']} Dependent arms: {', '.join(o['dependant_arms']) or '-'}")
    print(OPTIONS)
    print(RECONCILIATION)
    if not a.dry_run and not a.assume_resolved:
        write(m, a.out)
        print(f"wrote {len(m.configs)} configs to {os.path.join(a.out, 'runs')} and {os.path.join(a.out, 'matrix.json')}")

if __name__ == '__main__':
    main()
