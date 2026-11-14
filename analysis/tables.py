"""Primary deliverable table: one row per arm (proposal 5, 'primary deliverable').

Columns: registered prediction + provenance (prereg/predictions.yaml), Δ steps to 3.28 for LOO (r18 → r18−arm) and
AOI (r11 → r11+arm) with 95% intervals, Δ tokens/s (throughput runs, same host only), additive wall-clock split as
fractions (Shapley), and the LOO-vs-AOI gap reported as an upper bound on interaction (co-tuning bias cannot be
separated from interaction). RQ3a columns (other thresholds) only for arms listed as resolved (study.yaml
rq3b.resolved_arms). Arms without a defined LOO (OPEN_DECISIONS.md D1) show 'undefined (D1)'.
Optional --results-a100 (the separate A100 dataset, lab_a100/PROTOCOL.md): the arms the H100 sweep never ran (A1b, A4, A5, A5b
add-one-in, A5b leave-one-out) are read from it with the A100's own baselines (fp_r11, fp_r18) and marked [A100]; their
throughput always comes from the H100, and their wall-clock split is shown only if transfer.py's pass rule passes. The same
option adds the transfer check, the RQ3b table, the A8 row and the whole-run timing table below the primary table.
Usage: python -I analysis/tables.py [--results results] [--results-a100 results-a100] [--out reports/table.md]
"""
import argparse, json, os, sys
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_runs, group, same_host
from persistence import steps_to_threshold, delta_ci, RQ3A_THRESHOLDS
from decompose import decompose, mean_step_time
from transfer import transfer_check, format_transfer, final_loss, combo

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = 3.28

def per_seed_steps(runs, threshold=TARGET, which='interpolated'):
    """Steps to threshold per seed. A run that never crosses gets persistence.py's fallback, budget step +
    step-equivalent shortfall (an extrapolation); None only if the local slope is not positive."""
    out = []
    for r in runs:
        s, l = r.val_curve(full_only=True)
        res = steps_to_threshold(s, l, threshold)
        if res['crossed']:
            out.append(res[which])
        else:
            short = res['step_equivalent_shortfall']
            out.append(res['budget_step'] + short if np.isfinite(short) else None)
    return out

def n_extrapolated(runs, threshold=TARGET):
    """How many runs never cross threshold (their steps come from the fallback)."""
    return sum(not steps_to_threshold(*r.val_curve(full_only=True), threshold)['crossed'] for r in runs)

def extrap_note(base, arm, threshold=TARGET):
    kb, ka = n_extrapolated(base, threshold), n_extrapolated(arm, threshold)
    return f' [extrap. {kb}/{len(base)} base, {ka}/{len(arm)} arm]' if kb or ka else ''

def throughput_tokens_per_s(results_root, exp_id, host, tokens_per_step):
    """Mean tokens/s from throughput runs of exp_id on host (results/throughput/<exp_id>/<host>/*/run/seed0)."""
    runs = load_runs(os.path.join(results_root, 'throughput', exp_id, host), pattern='*/run')
    if not runs:
        return None, None
    t = [mean_step_time(r.time_step, r.time_ms, r.time_warmup) for r in runs]
    return tokens_per_step / (np.mean(t) / 1000.0), float(np.mean(t))

def fmt(d):
    if d is None or d.get('delta') is None:
        return 'n/a'
    return f"{d['delta']:+.0f}" + (f" ± {d['halfwidth']:.0f}" if d.get('halfwidth') is not None else '')

def split_text(S0, T0, S1, T1):
    dec = decompose(S0, T0, S1, T1)
    return f"{dec['sample_efficiency_fraction']:.0%} / {dec['throughput_fraction']:.0%} / {dec['cross_term_fraction']:.1%}"

def mean_steps(runs):
    return float(np.mean([x for x in per_seed_steps(runs) if x is not None]))

def build(results_root, out_path, a100_root=None):
    study = yaml.safe_load(open(os.path.join(HERE, 'experiments', 'study.yaml')))
    arms = yaml.safe_load(open(os.path.join(HERE, 'experiments', 'arms.yaml')))['arms']
    prereg = yaml.safe_load(open(os.path.join(HERE, 'prereg', 'predictions.yaml')))
    preds = {p['ref']: p for p in prereg['predictions']}
    matrix = json.load(open(os.path.join(HERE, 'experiments', 'matrix.json')))
    undefined = {o['arm'] for o in matrix['open_decisions']}
    resolved = set(study['rq3b'].get('resolved_arms') or [])
    configs = {c['exp_id'] for c in matrix['configurations']}
    g = group(load_runs(results_root))
    ga = group(load_runs(a100_root)) if a100_root else {}      # never merged with g: both hosts have fp_r11 and fp_r18
    tr = None
    if ga.get('fp_r11') and g.get('fp_r11') and all(ga.get(k) and g.get(k) for k in ('aoi_A6', 'aoi_A3')):
        tr = transfer_check(g, ga, per_seed_steps)
    tok = 524288
    tp_host = study['throughput']['host']       # throughput is read from the H100 only, never from the A100 dataset
    rows = ['| arm | registered prediction (provenance) | Δ steps LOO (r18 → −arm) | Δ steps AOI (r11 → +arm) | '
            'Δ tokens/s LOO | Δ tokens/s AOI | wall-clock split LOO (sample-eff / throughput / cross) | '
            'wall-clock split AOI (sample-eff / throughput / cross) | '
            'LOO−AOI gap (upper bound on interaction) |'
            + (' RQ3a Δ steps @3.40/3.35/3.30 |' if resolved else ''),
            '|---|---|---|---|---|---|---|---|---|' + ('---|' if resolved else '')]
    for a in arms:
        p = preds.get(a.get('prediction_ref'), {})
        pred = f"{p.get('registered_prediction', '?')} ({p.get('basis', '?')})"
        fp11, fp18 = g.get('fp_r11', []), g.get('fp_r18', [])
        aoi_id, loo_id = f"aoi_{a['id']}", f"loo_{a['id']}"
        aoi, src_aoi = g.get(aoi_id, []), 'H100'
        if not aoi and ga.get(aoi_id) and ga.get('fp_r11'):       # the A100 fills the arms the H100 never ran
            aoi, fp11, src_aoi = ga[aoi_id], ga['fp_r11'], 'A100'
        loo = g.get(loo_id, []) if a['id'] not in undefined else None
        src_loo = 'H100'
        if loo is not None and not loo and ga.get(loo_id) and ga.get('fp_r18'):
            loo, fp18, src_loo = ga[loo_id], ga['fp_r18'], 'A100'
        tag = lambda src: ' [A100]' if src == 'A100' else ''
        trusted = lambda src: src == 'H100' or (tr is not None and tr['passed'])
        no_split = 'unsupported (A100 transfer check not passed)'
        d_aoi = delta_ci(per_seed_steps(fp11), per_seed_steps(aoi)) if fp11 and aoi else None
        if aoi and fp11:
            same_host(fp11, aoi)
        # an arm whose configuration is in the matrix but has no completed run is 'not run' (throughput-only arms)
        t0, t1 = (throughput_tokens_per_s(results_root, e, tp_host, tok) for e in ('fp_r11', aoi_id))
        tps_aoi_cell = f'{t1[0] - t0[0]:+.0f}' if t0[0] and t1[0] else 'n/a'
        split_aoi_cell = 'n/a'
        if t0[0] and t1[0] and d_aoi and d_aoi['delta'] is not None:
            S0 = mean_steps(fp11)
            split_aoi_cell = split_text(S0, t0[1], S0 + d_aoi['delta'], t1[1]) if trusted(src_aoi) else no_split
        if loo is None:
            loo_cell, split_cell, tps_cell, gap_cell = 'undefined (D1)', 'n/a', 'n/a', 'n/a'
        else:
            d_loo = delta_ci(per_seed_steps(fp18), per_seed_steps(loo)) if fp18 and loo else None
            if fp18 and loo:
                host = same_host(fp18, loo)
            loo_cell = fmt(d_loo) + (extrap_note(fp18, loo) if d_loo else '') + (tag(src_loo) if loo else '')
            if not loo and f"loo_{a['id']}" in configs:
                loo_cell = 'not run'
            split_cell = tps_cell = 'n/a'
            t0 = throughput_tokens_per_s(results_root, 'fp_r18', tp_host, tok)
            t1 = throughput_tokens_per_s(results_root, loo_id, tp_host, tok)
            if t0[0] and t1[0]:
                tps_cell = f'{t1[0] - t0[0]:+.0f}'          # throughput needs no learning run (throughput-only arms)
                if fp18 and loo and d_loo and d_loo['delta'] is not None:
                    split_cell = (split_text(mean_steps(fp18), t0[1], mean_steps(fp18) + d_loo['delta'], t1[1])
                                  if trusted(src_loo) else no_split)
            gap_cell = 'n/a'
            if d_loo and d_aoi and d_loo['delta'] is not None and d_aoi['delta'] is not None:
                # removal effect measured at the top vs addition effect at the bottom: compare magnitudes
                if src_loo != src_aoi:
                    gap_cell = 'n/a (different hosts)'
                else:
                    gap = abs(d_loo['delta']) - abs(d_aoi['delta'])
                    hw = (np.hypot(d_loo['halfwidth'], d_aoi['halfwidth'])
                          if d_loo['halfwidth'] is not None and d_aoi['halfwidth'] is not None else None)
                    gap_cell = f'{gap:+.0f}' + (f' ± {hw:.0f}' if hw is not None else '') + tag(src_loo)
        rq3 = ''
        if resolved:
            # rq3b.resolved_arms was fixed from the H100 sweep; an A100 arm whose RQ1 interval excludes 0 gets RQ3a under the same criterion
            a100_resolved = (src_aoi == 'A100' and d_aoi is not None and d_aoi['halfwidth'] is not None
                             and abs(d_aoi['delta']) > d_aoi['halfwidth'])
            if (a['id'] in resolved or a100_resolved) and fp11 and aoi:
                cells = [fmt(delta_ci(per_seed_steps(fp11, t), per_seed_steps(aoi, t))) for t in RQ3A_THRESHOLDS[:3]]
                rq3 = ' ' + ' / '.join(cells) + (' [A100, resolved in RQ1; not in rq3b.resolved_arms]' if a100_resolved and a['id'] not in resolved else '') + ' |'
            else:
                rq3 = ' (not in rq3b.resolved_arms) |'
        aoi_cell = fmt(d_aoi) + (extrap_note(fp11, aoi) if d_aoi else '') + (tag(src_aoi) if aoi else '')
        if not aoi and aoi_id in configs:
            aoi_cell = 'not run'
        if a['id'] == 'A8' and a8_row(g):
            r8 = a8_row(g)
            loo_cell = (f"AdamW +{r8['gap']:.4f} loss at the budget" + (f"; needs ≥ {r8['ratio']:.2f}× the steps" if 'ratio' in r8 else '')
                        + ' (one run, lower bound; DEVIATIONS.md 3.3)')
            aoi_cell = 'not run (ctrl_A8_r11, D13)'
        rows.append(f"| {a['id']} {a['name']} | {pred} | {loo_cell} | {aoi_cell} | {tps_cell} | {tps_aoi_cell} | {split_cell} | "
                    f"{split_aoi_cell} | {gap_cell} |" + rq3)
    text = '\n'.join(rows) + '\n\nΔ steps: interpolated crossing of 3.28 on full-validation points; intervals are 95% Welch t ' \
           'intervals across seeds. A run that never crosses 3.28 counts as budget step + step-equivalent shortfall ' \
           '(loss at budget − 3.28) / local slope, the persistence.py fallback; [extrap. k/n] says how many seeds of the ' \
           'baseline and the arm that is. "not run" = the configuration is in the matrix but has no completed learning run; ' \
           'such arms are throughput-only, and their Δ tokens/s columns still apply. ' \
           '[A100] = measured on the lab A100 (lab_a100/PROTOCOL.md) against the A100\'s own baseline (fp_r11 for add-one-in, ' \
           'fp_r18 for leave-one-out), never against the H100 one; its tokens/s are the H100\'s, so its wall-clock split ' \
           'needs the transfer check below to pass. Δ tokens/s and the splits use the H100 throughput sweep (300 steps ' \
           'each); see the timing table for whole-run step times. The add-one-in rows of A2, A4b and A6 carry prerequisite flags ' \
           '(DEVIATIONS.md 3.1). Split fractions blow up when the total time change is near zero (A4b, A5b, A6b); read the signed ' \
           'Δ steps and Δ tokens/s there. No p-values are reported.\n'
    text += '\n' + sections(results_root, g, ga, study, tr, tp_host, tok, bool(a100_root))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, 'w').write(text)
    return text

def final_mean(runs):
    return float(np.mean([final_loss(r) for r in runs]))

def rq3b_table(ga):
    """Budget sweep (RQ3b), add-one-in side, on the A100: Δ final validation loss (arm − fp_r11) at 0.5×, 0.75× and 1.0× of the
    3,000-step budget with the LR schedule rescaled. Seed 0 against seed 0 at every scale (same initialisation and data order);
    the 1.0× row repeats it with every seed. 3.28 is out of reach at the short budgets, so the matched quantity is the loss."""
    arms = ('aoi_A3', 'aoi_A6')
    def pt(scale, name, seeds0=True):
        runs = ga.get(name if scale == 1.0 else f"rq3b_s{scale}_{name}", [])
        runs = [r for r in runs if r.seed == 0] if seeds0 else runs
        return runs
    L = ['| budget | fp_r11 final loss | ' + ' | '.join(f'{x} final loss | Δ {x}' for x in arms) + ' |',
         '|---|---|' + '---|---|' * len(arms)]
    deltas = {x: {} for x in arms}
    for scale, seeds0, label in ((0.5, True, '0.5× (1,500 steps)'), (0.75, True, '0.75× (2,250 steps)'),
                                 (1.0, True, '1.0× (3,000 steps), seed 0'), (1.0, False, '1.0× (3,000 steps), all seeds')):
        base = pt(scale, 'fp_r11', seeds0)
        if not base:
            continue
        cells = []
        for x in arms:
            r = pt(scale, x, seeds0)
            if r:
                d = final_mean(r) - final_mean(base)
                if seeds0:
                    deltas[x][scale] = d
                cells += [f'{final_mean(r):.4f} (n={len(r)})', f'{d:+.4f}']
            else:
                cells += ['not run', 'n/a']
        L.append(f'| {label} | {final_mean(base):.4f} (n={len(base)}) | ' + ' | '.join(cells) + ' |')
    return '\n'.join(L) + '\n', deltas

def a8_row(g):
    """A8 positive control (DEVIATIONS.md 3.3): loss gap at the budget and the lower bound on the step ratio."""
    ctrl, base = g.get('ctrl_A8_r18', []), g.get('fp_r18', [])
    if not ctrl or not base:
        return None
    adam = float(np.mean([final_loss(r) for r in ctrl]))
    muon = float(np.mean([final_loss(r) for r in base]))
    crossing = [steps_to_threshold(*r.val_curve(full_only=True), adam) for r in base]
    reached = [c['interpolated'] for c in crossing if c['crossed']]
    budget = float(ctrl[0].val_curve(full_only=True)[0][-1])
    out = dict(adamw=adam, muon=muon, gap=adam - muon, n_ctrl=len(ctrl), n_base=len(base), budget=budget)
    if reached:
        out['muon_step'] = float(np.mean(reached))
        out['ratio'] = budget / out['muon_step']
    return out

def timing_table(results_root, g, tp_host, tok):
    """Whole-run mean step time of the H100 learning runs next to the 300-step throughput run of the same configuration."""
    L = ['| configuration | learning runs (n) | whole-run mean step time (ms) | seed SD (%) | throughput run, first 300 steps (ms) | whole run vs 300 steps |',
         '|---|---|---|---|---|---|']
    for name in sorted(g):
        if name.startswith('ctrl'):
            continue
        t = [mean_step_time(r.time_step, r.time_ms, r.time_warmup) for r in g[name] if len(r.time_step)]
        tp = throughput_tokens_per_s(results_root, name, tp_host, tok)
        if not t:
            continue
        m = float(np.mean(t))
        sd = f'{100 * np.std(t, ddof=1) / m:.2f}' if len(t) > 1 else 'n/a'
        L.append(f"| {name} | {len(t)} | {m:.0f} | {sd} | {tp[1]:.0f} | {100 * (m / tp[1] - 1):+.1f}% |" if tp[1]
                 else f'| {name} | {len(t)} | {m:.0f} | {sd} | n/a | n/a |')
    return '\n'.join(L) + '\n'

def sections(results_root, g, ga, study, tr, tp_host, tok, with_a100=True):
    out = []
    if with_a100:
        out.append('## Transfer check: lab A100 against the H100\n')
        out.append(format_transfer(tr) if tr else 'Not computed: a baseline or a bridge arm is missing from one of the two datasets.\n')
        out.append('\n## RQ3b: budget sweep (add-one-in side, A100)\n')
        out.extend(rq3b_sections(ga))
    out.append('\n## A8 positive control (AdamW instead of Muon)\n')
    r = a8_row(g)
    if r:
        out.append(f"AdamW ends at {r['adamw']:.4f} after {r['budget']:.0f} steps ({r['n_ctrl']} run) against {r['muon']:.4f} for Muon "
                   f"(fp_r18, {r['n_base']} seeds): a loss gap of {r['gap']:+.4f}. " +
                   (f"Muon reaches AdamW's final loss at step {r['muon_step']:.0f}, so AdamW needs at least {r['ratio']:.2f}× "
                    f"the steps (lower bound; one run, borrowed learning rate 0.0018, no sweep: an upper bound on Muon's advantage)."
                    if 'ratio' in r else 'Muon never reaches that loss.') + '\n')
    else:
        out.append('Not computed: ctrl_A8_r18 or fp_r18 missing.\n')
    out.append('\n## Whole-run step time (H100 learning runs) next to the 300-step throughput runs\n')
    out.append(timing_table(results_root, g, tp_host, tok))
    out.append('\nThe whole-run mean covers every logged step except compile warm-up; the throughput run covers the first 300 steps '
               '(290 after warm-up) of one run. They differ most where the attention window grows during training (window '
               'warm-up). The A100 timings are not reported (DEVIATIONS.md 7).\n')
    return '\n'.join(out)

def rq3b_sections(ga):
    out = []
    t, deltas = rq3b_table(ga)
    out.append(t)
    out.append('\nΔ = final validation loss of the arm minus fp_r11 at the same budget (negative = the arm is better); every point is one '
               'run of seed 0, so Δ has no interval of its own. With the fp_r11 seed SD of 0.0042 the difference of two single runs has '
               'SD about 0.006 (not paired); pairing by seed index may shrink it, as the A100−H100 pairs of the transfer check '
               'suggest, and it is not assumed here. The budget is a multiple of the 3,000-step add-one-in budget with the learning-rate '
               'schedule rescaled (absolute-step hyper-parameters listed in study.yaml are rescaled with it). The leave-one-out side of RQ3b, '
               'aoi_A4b and the arms A2 and A7 were not run (lab_a100/PROTOCOL.md section 2).\n')
    for x, d in deltas.items():
        if {0.5, 0.75, 1.0} <= set(d):
            out.append(f'\n{x}: Δ loss {d[0.5]:+.4f} (0.5×), {d[0.75]:+.4f} (0.75×), {d[1.0]:+.4f} (1.0×).')
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--results-a100', default=None, help='the separate A100 dataset (never merged with --results)')
    ap.add_argument('--out', default=os.path.join(HERE, 'reports', 'table.md'))
    a = ap.parse_args()
    print(build(a.results, a.out, a.results_a100))

if __name__ == '__main__':
    main()
