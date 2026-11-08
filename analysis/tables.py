"""Primary deliverable table: one row per arm (proposal 5, 'primary deliverable').

Columns: registered prediction + provenance (prereg/predictions.yaml), Δ steps to 3.28 for LOO (r18 → r18−arm) and
AOI (r11 → r11+arm) with 95% intervals, Δ tokens/s (throughput runs, same host only), additive wall-clock split as
fractions (Shapley), and the LOO-vs-AOI gap reported as an upper bound on interaction (co-tuning bias cannot be
separated from interaction). RQ3a columns (other thresholds) only for arms listed as resolved (study.yaml
rq3b.resolved_arms). Arms without a defined LOO (OPEN_DECISIONS.md D1) show 'undefined (D1)'.
Usage: python -I analysis/tables.py [--results results] [--out reports/table.md]
"""
import argparse, json, os, sys
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_runs, group, same_host
from persistence import steps_to_threshold, delta_ci, RQ3A_THRESHOLDS
from decompose import decompose, mean_step_time

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

def build(results_root, out_path):
    study = yaml.safe_load(open(os.path.join(HERE, 'experiments', 'study.yaml')))
    arms = yaml.safe_load(open(os.path.join(HERE, 'experiments', 'arms.yaml')))['arms']
    prereg = yaml.safe_load(open(os.path.join(HERE, 'prereg', 'predictions.yaml')))
    preds = {p['ref']: p for p in prereg['predictions']}
    matrix = json.load(open(os.path.join(HERE, 'experiments', 'matrix.json')))
    undefined = {o['arm'] for o in matrix['open_decisions']}
    resolved = set(study['rq3b'].get('resolved_arms') or [])
    g = group(load_runs(results_root))
    tok = 524288
    rows = ['| arm | registered prediction (provenance) | Δ steps LOO (r18 → −arm) | Δ steps AOI (r11 → +arm) | '
            'Δ tokens/s LOO | wall-clock split LOO (sample-eff / throughput / cross) | LOO−AOI gap (upper bound on interaction) |'
            + (' RQ3a Δ steps @3.40/3.35/3.30 |' if resolved else ''),
            '|---|---|---|---|---|---|---|' + ('---|' if resolved else '')]
    for a in arms:
        p = preds.get(a.get('prediction_ref'), {})
        pred = f"{p.get('registered_prediction', '?')} ({p.get('basis', '?')})"
        fp11, fp18 = g.get('fp_r11', []), g.get('fp_r18', [])
        aoi = g.get(f"aoi_{a['id']}", [])
        loo = g.get(f"loo_{a['id']}", []) if a['id'] not in undefined else None
        d_aoi = delta_ci(per_seed_steps(fp11), per_seed_steps(aoi)) if fp11 and aoi else None
        if aoi and fp11:
            same_host(fp11, aoi)
        if loo is None:
            loo_cell, split_cell, tps_cell, gap_cell = 'undefined (D1)', 'n/a', 'n/a', 'n/a'
        else:
            d_loo = delta_ci(per_seed_steps(fp18), per_seed_steps(loo)) if fp18 and loo else None
            if fp18 and loo:
                host = same_host(fp18, loo)
            loo_cell = fmt(d_loo) + (extrap_note(fp18, loo) if d_loo else '')
            split_cell = tps_cell = 'n/a'
            if fp18 and loo and d_loo and d_loo['delta'] is not None:
                t0 = throughput_tokens_per_s(results_root, 'fp_r18', study['throughput']['host'], tok)
                t1 = throughput_tokens_per_s(results_root, f"loo_{a['id']}", study['throughput']['host'], tok)
                if t0[0] and t1[0]:
                    tps_cell = f'{t1[0] - t0[0]:+.0f}'
                    S0 = np.mean([x for x in per_seed_steps(fp18) if x is not None])
                    dec = decompose(S0, t0[1], S0 + d_loo['delta'], t1[1])
                    split_cell = (f"{dec['sample_efficiency_fraction']:.0%} / {dec['throughput_fraction']:.0%} / "
                                  f"{dec['cross_term_fraction']:.1%}")
            gap_cell = 'n/a'
            if d_loo and d_aoi and d_loo['delta'] is not None and d_aoi['delta'] is not None:
                # removal effect measured at the top vs addition effect at the bottom: compare magnitudes
                gap = abs(d_loo['delta']) - abs(d_aoi['delta'])
                hw = (np.hypot(d_loo['halfwidth'], d_aoi['halfwidth'])
                      if d_loo['halfwidth'] is not None and d_aoi['halfwidth'] is not None else None)
                gap_cell = f'{gap:+.0f}' + (f' ± {hw:.0f}' if hw is not None else '')
        rq3 = ''
        if resolved:
            if a['id'] in resolved and fp11 and aoi:
                cells = [fmt(delta_ci(per_seed_steps(fp11, t), per_seed_steps(aoi, t))) for t in RQ3A_THRESHOLDS[:3]]
                rq3 = ' ' + ' / '.join(cells) + ' |'
            else:
                rq3 = ' (not resolved in RQ1) |'
        aoi_cell = fmt(d_aoi) + (extrap_note(fp11, aoi) if d_aoi else '')
        rows.append(f"| {a['id']} {a['name']} | {pred} | {loo_cell} | {aoi_cell} | {tps_cell} | {split_cell} | {gap_cell} |" + rq3)
    text = '\n'.join(rows) + '\n\nΔ steps: interpolated crossing of 3.28 on full-validation points; intervals are 95% Welch t ' \
           'intervals across seeds. A run that never crosses 3.28 counts as budget step + step-equivalent shortfall ' \
           '(loss at budget − 3.28) / local slope, the persistence.py fallback; [extrap. k/n] says how many seeds of the ' \
           'baseline and the arm that is. No p-values are reported.\n'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, 'w').write(text)
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--out', default=os.path.join(HERE, 'reports', 'table.md'))
    a = ap.parse_args()
    print(build(a.results, a.out))

if __name__ == '__main__':
    main()
