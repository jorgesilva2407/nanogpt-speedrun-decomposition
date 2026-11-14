"""Transfer check of the lab A100 against the H100 (lab_a100/PROTOCOL.md section 3), evaluated on the `anchor` phase.

The decomposition of an arm that only the A100 measured combines its Δsteps from the A100 with the throughput of the H100, so
Δsteps has to carry over between the two GPUs. The pass rule is the one written before any A100 learning run:
  1 offset   Welch 95% interval of (A100 mean − H100 mean) of the fp_r11 final validation loss lies inside ±0.010
  2 curve    the A100 mean fp_r11 curve is within 0.010 of the H100 one at every 125-step checkpoint of the final third
  3 seeding  the A100 fp_r11 seed SD of the final loss is in [0.0002, 0.0057]
  4 bridge   aoi_A6: Δsteps(A100) has the sign of Δsteps(H100) and the Welch 95% interval of Δ(A100) − Δ(H100) contains 0;
             aoi_A3 (one seed per GPU, no interval): same sign
Secondary, reported and not a gate: the offset paired by seed index, and the fp_r18 offset.
Inputs are groups of loaded runs per host and `per_seed_steps` (tables.py), so the rule is applied to the same step counts as
the table (full-validation points, never-crossed fallback).
"""
import numpy as np
from mde import _crit

OFFSET_LIMIT = 0.010
CURVE_LIMIT = 0.010
SD_RANGE = (0.0002, 0.0057)
CHECKPOINT_EVERY = 125

def combo(groups, coefs):
    """Linear combination Σ c_i·mean(g_i) with a Welch–Satterthwaite 95% t half-width.
    Returns (estimate, halfwidth); the half-width is None when a group has fewer than 2 values."""
    gs = [np.asarray([x for x in g if x is not None], float) for g in groups]
    est = float(sum(c * g.mean() for c, g in zip(coefs, gs)))
    if any(len(g) < 2 for g in gs):
        return est, None
    terms = [c * c * g.var(ddof=1) / len(g) for c, g in zip(coefs, gs)]
    var = sum(terms)
    if var <= 0:
        return est, 0.0
    df = var ** 2 / sum(t ** 2 / (len(g) - 1) for t, g in zip(terms, gs))
    return est, float(_crit(int(round(df)), True) * np.sqrt(var))

def final_loss(run):
    return float(run.val_curve(full_only=True)[1][-1])

def final_losses(runs):
    return [final_loss(r) for r in runs]

def checkpoint_curve(runs):
    """Mean full-validation loss over seeds at the steps that are multiples of 125 in the final third of the budget."""
    curves = [dict(zip(*map(list, r.val_curve(full_only=True)))) for r in runs]
    budget = max(max(c) for c in curves)
    steps = sorted(s for s in set.intersection(*(set(c) for c in curves))
                   if s >= budget * 2 / 3 - 1e-9 and s % CHECKPOINT_EVERY == 0)
    return steps, np.array([np.mean([c[s] for c in curves]) for s in steps])

def paired_by_seed(h_runs, a_runs):
    """A100 − H100 final-loss differences for the seeds both hosts ran (seed k fixes initialisation and data order)."""
    h = {r.seed: final_loss(r) for r in h_runs}
    return [final_loss(r) - h[r.seed] for r in a_runs if r.seed in h]

def sign(x):
    return (x > 0) - (x < 0)

def transfer_check(h, a, per_seed_steps):
    """h, a: {configuration: [Run]} of the H100 and the A100. Returns the numbers of the pass rule and `passed`."""
    out = {}
    h11, a11 = h['fp_r11'], a['fp_r11']
    fh, fa = final_losses(h11), final_losses(a11)
    est, hw = combo([fa, fh], [1, -1])
    out['offset'] = dict(estimate=est, halfwidth=hw, n_a100=len(fa), n_h100=len(fh), limit=OFFSET_LIMIT,
                         passed=hw is not None and abs(est) + hw <= OFFSET_LIMIT)
    steps, ca = checkpoint_curve(a11)
    _, ch = checkpoint_curve(h11)
    gap = np.abs(ca - ch)
    out['curve'] = dict(steps=steps, max_gap=float(gap.max()), at_step=steps[int(gap.argmax())], limit=CURVE_LIMIT,
                        passed=bool(gap.max() <= CURVE_LIMIT))
    sd = float(np.std(fa, ddof=1))
    out['seeding'] = dict(sd_a100=sd, sd_h100=float(np.std(fh, ddof=1)), range=SD_RANGE,
                          passed=SD_RANGE[0] <= sd <= SD_RANGE[1])
    sh0, sa0 = per_seed_steps(h11), per_seed_steps(a11)
    bridge = {}
    for arm in ('aoi_A6', 'aoi_A3'):
        sh1, sa1 = per_seed_steps(h[arm]), per_seed_steps(a[arm])
        d_h = float(np.mean([x for x in sh1 if x is not None]) - np.mean([x for x in sh0 if x is not None]))
        d_a = float(np.mean([x for x in sa1 if x is not None]) - np.mean([x for x in sa0 if x is not None]))
        dd, dhw = combo([sa1, sa0, sh1, sh0], [1, -1, -1, 1])
        same_sign = sign(d_h) == sign(d_a)
        bridge[arm] = dict(delta_h100=d_h, delta_a100=d_a, n_h100=len(sh1), n_a100=len(sa1), same_sign=same_sign,
                           difference=dd, halfwidth=dhw,
                           passed=same_sign and (dhw is None or abs(dd) <= dhw))
    out['bridge'] = bridge
    out['paired'] = dict(diffs=paired_by_seed(h11, a11))
    if h.get('fp_r18') and a.get('fp_r18'):
        e18, hw18 = combo([final_losses(a['fp_r18']), final_losses(h['fp_r18'])], [1, -1])
        out['fp_r18_offset'] = dict(estimate=e18, halfwidth=hw18, n_a100=len(a['fp_r18']), n_h100=len(h['fp_r18']))
    out['passed'] = all([out['offset']['passed'], out['curve']['passed'], out['seeding']['passed'],
                         all(b['passed'] for b in bridge.values())])
    return out

def format_transfer(t):
    ok = lambda b: 'pass' if b else 'FAIL'
    o, c, s = t['offset'], t['curve'], t['seeding']
    ci = lambda e, hw: f'{e:+.4f}' + (f' ± {hw:.4f}' if hw is not None else '')
    L = ['| check (PROTOCOL.md section 3) | measured | rule | |', '|---|---|---|---|',
         f"| 1 baseline offset, fp_r11 final loss, A100 − H100 ({o['n_a100']} vs {o['n_h100']} seeds) | {ci(o['estimate'], o['halfwidth'])} "
         f"| interval inside ±{o['limit']:.3f} | {ok(o['passed'])} |",
         f"| 2 curve, largest gap of the mean fp_r11 curve over the final-third 125-step checkpoints | {c['max_gap']:.4f} (step {c['at_step']:.0f}) "
         f"| ≤ {c['limit']:.2f} | {ok(c['passed'])} |",
         f"| 3 seeding, A100 fp_r11 SD of the final loss (H100: {s['sd_h100']:.4f}) | {s['sd_a100']:.4f} | in [{s['range'][0]}, {s['range'][1]}] | {ok(s['passed'])} |"]
    for arm, b in t['bridge'].items():
        what = (f"Δ(A100) − Δ(H100) = {b['difference']:+.0f} ± {b['halfwidth']:.0f} steps" if b['halfwidth'] is not None
                else 'one seed per GPU, no interval')
        L.append(f"| 4 bridge {arm}: Δsteps A100 {b['delta_a100']:+.0f} (n={b['n_a100']}) vs H100 {b['delta_h100']:+.0f} (n={b['n_h100']}) "
                 f"| {what} | same sign" + (', interval contains 0' if b['halfwidth'] is not None else '') + f" | {ok(b['passed'])} |")
    L.append(f"| **result** | | all of the above | **{ok(t['passed'])}** |")
    p = t['paired']['diffs']
    text = '\n'.join(L) + '\n\n'
    text += ('Not gates: the offset paired by seed index (A100 − H100 final loss, seeds 0, 1, 2) is '
             + ', '.join(f'{d:+.4f}' for d in p) + f' (mean {np.mean(p):+.4f}).')
    if 'fp_r18_offset' in t:
        f = t['fp_r18_offset']
        text += (f" The fp_r18 offset ({f['n_a100']} A100 vs {f['n_h100']} H100 seeds, the FlexAttention path) is "
                 f"{ci(f['estimate'], f['halfwidth'])}.")
    return text + '\n'
