"""Analysis tests on synthetic data (numpy, CPU)."""
import json, os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mde, persistence, decompose
from load import load_runs, group, same_host, HostPoolingError

# ---------------------------------------------------------------- MDE: the proposal's 5.1 table at sigma = 0.0013
@pytest.mark.parametrize('n,rq1,rq2', [(3, 46, 65), (5, 36, 51)])
def test_proposal_mde_table(n, rq1, rq2):
    t = mde.proposal_table(0.0013)
    assert round(t[n]['rq1']) == rq1 and round(t[n]['rq2']) == rq2

def test_mde_arithmetic_and_t_interval_is_wider():
    assert mde.rq1_halfwidth(0.0013, 3) == pytest.approx(1.959964 * 0.0013 * (2 / 3) ** 0.5, rel=1e-6)
    assert mde.rq2_halfwidth(0.0013, 5) == pytest.approx(1.959964 * 2 * 0.0013 / 5 ** 0.5, rel=1e-6)
    assert mde.to_steps(mde.rq1_halfwidth(0.0013, 3, use_t=True)) == pytest.approx(65.5, abs=0.1)

def curve(n=1400, a=3.20, b=7.6, c=180.0, every=25, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    s = np.arange(0, n + 1, every, dtype=float)
    return s, a + b * np.exp(-s / c) + noise * rng.standard_normal(len(s))

def test_local_slope_matches_analytic():
    s, l = curve(every=5)
    thr = 3.28
    s_star = -180.0 * np.log((thr - 3.20) / 7.6)
    assert mde.local_slope(s, l, thr, window=2) == pytest.approx((7.6 / 180.0) * np.exp(-s_star / 180.0), rel=0.05)

# ---------------------------------------------------------------- steps to threshold
def test_crossing_grid_and_interpolated():
    s, l = curve()
    r = persistence.steps_to_threshold(s, l, 3.28)
    exact = -180.0 * np.log((3.28 - 3.20) / 7.6)
    assert r['crossed'] and r['grid'] >= exact and r['grid'] - exact < 25
    assert abs(r['interpolated'] - exact) < 2.0          # linear interpolation inside a 25-step cell

def test_never_crossed_fallback():
    s, l = curve(n=600)
    r = persistence.steps_to_threshold(s, l, 3.28, budget_step=600)
    assert not r['crossed'] and r['loss_at_budget'] > 3.28 and r['step_equivalent_shortfall'] > 0

def test_rq3a_thresholds_ordered():
    s, l = curve()
    t = persistence.thresholds_table(s, l)
    vals = [t[x]['interpolated'] for x in persistence.RQ3A_THRESHOLDS]
    assert vals == sorted(vals)

def test_delta_ci_recovers_shift():
    a = [1000 + x for x in (-5, 0, 5, 3, -3)]
    b = [940 + x for x in (-4, 2, 4, -1, 1)]
    d = persistence.delta_ci(a, b)
    assert d['delta'] == pytest.approx(np.mean(b) - np.mean(a)) and d['halfwidth'] > 0

# ---------------------------------------------------------------- decomposition
def test_decomposition_identity_and_symmetric_split():
    d = decompose.decompose(S0=1490, T0=0.150, S1=1390, T1=0.160)
    assert d['sample_efficiency'] + d['throughput'] == pytest.approx(d['d_time'])
    dS, dT = -100, 0.010
    assert d['cross_term'] == pytest.approx(dS * dT)
    assert d['sample_efficiency'] == pytest.approx(dS * 0.150 + dS * dT / 2)
    assert d['throughput'] == pytest.approx(1490 * dT + dS * dT / 2)

def test_mean_step_time_excludes_warmup_and_uses_mean_not_median():
    steps = np.arange(20); ms = np.r_[np.full(10, 5000.0), np.linspace(100, 200, 10)]
    warm = steps < 10
    assert decompose.mean_step_time(steps, ms, warm) == pytest.approx(150.0)

# ---------------------------------------------------------------- loading and host pooling
def write_run(root, name, seed, host, vals):
    d = os.path.join(root, name, f'seed{seed}'); os.makedirs(d)
    json.dump(dict(host_label=host, seed=seed, exp_id=name), open(os.path.join(d, 'meta.json'), 'w'))
    with open(os.path.join(d, 'metrics.jsonl'), 'w') as f:
        for s, v in vals:
            f.write(json.dumps(dict(kind='val', step=s, tokens=s * 524288, val_loss=v, val_fraction=1.0, wall_clock_s=0)) + '\n')
    open(os.path.join(d, 'timing.jsonl'), 'w').close()
    open(os.path.join(d, 'DONE'), 'w').write('x')
    return d

def test_incomplete_runs_ignored_and_host_pooling_refused(tmp_path):
    write_run(tmp_path, 'fp_r18', 0, 'lab', [(0, 10.8), (100, 3.2)])
    write_run(tmp_path, 'fp_r18', 1, 'cloud', [(0, 10.8), (100, 3.2)])
    d = write_run(tmp_path, 'fp_r18', 2, 'lab', [(0, 10.8)])
    os.remove(os.path.join(d, 'DONE'))
    runs = load_runs(str(tmp_path))
    assert len(runs) == 2
    with pytest.raises(HostPoolingError):
        group(runs)
    with pytest.raises(HostPoolingError):
        same_host([runs[0]], [runs[1]])

# ---------------------------------------------------------------- primary table, end to end on synthetic results
def write_full_run(root, name, seed, host, c, timing_ms=None, a=3.20):
    d = os.path.join(root, name, f'seed{seed}'); os.makedirs(d)
    json.dump(dict(host_label=host, seed=seed, exp_id=name), open(os.path.join(d, 'meta.json'), 'w'))
    s, l = curve(a=a, c=c, every=25, noise=0.001, seed=seed)
    with open(os.path.join(d, 'metrics.jsonl'), 'w') as f:
        for st, v in zip(s, l):
            f.write(json.dumps(dict(kind='val', step=int(st), tokens=int(st) * 524288, val_loss=float(v), val_fraction=1.0, wall_clock_s=0)) + '\n')
    with open(os.path.join(d, 'timing.jsonl'), 'w') as f:
        for st in range(30):
            ms = 5000.0 if st < 10 else (timing_ms or 150.0)
            f.write(json.dumps(dict(step=st, gpu_ms=ms, compile_warmup=st < 10, wall_clock_s=0)) + '\n')
    open(os.path.join(d, 'DONE'), 'w').write('x')

def test_primary_table_end_to_end(tmp_path):
    import tables
    for k in range(3):
        write_full_run(tmp_path, 'fp_r11', k, 'lab', c=200.0)
        write_full_run(tmp_path, 'aoi_A7', k, 'lab', c=190.0)
        write_full_run(tmp_path, 'fp_r18', k, 'lab', c=180.0)
        write_full_run(tmp_path, 'loo_A7', k, 'lab', c=186.0)
    write_full_run(os.path.join(tmp_path, 'throughput', 'fp_r18', 'vast-h100', 't1'), 'run', 0, 'vast-h100', c=180.0, timing_ms=150.0)
    write_full_run(os.path.join(tmp_path, 'throughput', 'loo_A7', 'vast-h100', 't1'), 'run', 0, 'vast-h100', c=186.0, timing_ms=140.0)
    out = tmp_path / 'table.md'
    text = tables.build(str(tmp_path), str(out))
    row = next(l for l in text.splitlines() if l.startswith('| A7 '))
    cells = [c.strip() for c in row.strip('|').split('|')]
    assert cells[1].startswith('No sample-efficiency effect')
    assert cells[2].startswith('+') and cells[3].startswith('-')         # LOO slower to target, AOI faster
    assert cells[4].startswith('+')                                        # 140 ms/step vs 150 ms/step: more tokens/s
    assert '/' in cells[6] and '%' in cells[6]                              # cells[5] is the AOI Δ tokens/s
    assert 'undefined (D1)' in next(l for l in text.splitlines() if l.startswith('| A3 '))

def test_primary_table_uses_fallback_for_runs_that_never_cross(tmp_path):
    import tables
    for k in range(3):
        write_full_run(tmp_path, 'fp_r18', k, 'lab', c=180.0)
        write_full_run(tmp_path, 'loo_A7', k, 'lab', c=180.0, a=3.285)    # plateaus above 3.28: never crosses
    runs = group(load_runs(str(tmp_path)))
    base, arm = tables.per_seed_steps(runs['fp_r18']), tables.per_seed_steps(runs['loo_A7'])
    assert None not in arm and min(arm) > 1400 > max(base)               # extrapolated past the 1400-step budget
    row = next(l for l in tables.build(str(tmp_path), str(tmp_path / 't.md')).splitlines() if l.startswith('| A7 '))
    cell = [c.strip() for c in row.strip('|').split('|')][2]
    assert cell.startswith('+') and '[extrap. 0/3 base, 3/3 arm]' in cell

def test_primary_table_reports_arms_without_learning_runs_as_throughput_only(tmp_path):
    import tables
    for k in range(3):
        write_full_run(tmp_path, 'fp_r11', k, 'lab', c=200.0)
    write_full_run(os.path.join(tmp_path, 'throughput', 'fp_r11', 'vast-h100', 't1'), 'run', 0, 'vast-h100', c=200.0, timing_ms=100.0)
    write_full_run(os.path.join(tmp_path, 'throughput', 'aoi_A1b', 'vast-h100', 't1'), 'run', 0, 'vast-h100', c=200.0, timing_ms=125.0)
    text = tables.build(str(tmp_path), str(tmp_path / 't.md'))
    cells = [c.strip() for c in next(l for l in text.splitlines() if l.startswith('| A1b ')).strip('|').split('|')]
    assert cells[3] == 'not run' and cells[5].startswith('-')                 # no steps; 125 vs 100 ms/step: fewer tokens/s
    assert 'not run' not in next(l for l in text.splitlines() if l.startswith('| A8 '))   # ctrl arms have no aoi/loo config

# ---------------------------------------------------------------- the A100 dataset: separate root, own baselines, transfer gate
def populate_hosts(h_root, a_root, a100_offset=0.0):
    """Both roots hold fp_r11 and the bridge arms (the real datasets share these names, so they must never be pooled)."""
    for k in range(3):
        write_full_run(h_root, 'fp_r11', k, 'vast-h100', c=200.0)
        write_full_run(a_root, 'fp_r11', k, 'lab-a100', c=200.0, a=3.20 + a100_offset)
    for k in range(2):
        write_full_run(h_root, 'aoi_A6', k, 'vast-h100', c=170.0)
        write_full_run(a_root, 'aoi_A6', k, 'lab-a100', c=170.0, a=3.20 + a100_offset)
    write_full_run(h_root, 'aoi_A3', 0, 'vast-h100', c=170.0)
    write_full_run(a_root, 'aoi_A3', 0, 'lab-a100', c=170.0, a=3.20 + a100_offset)
    for root, host in ((h_root, 'vast-h100'),):       # throughput is the H100's: the split needs it
        write_full_run(os.path.join(root, 'throughput', 'fp_r11', host, 't1'), 'run', 0, host, c=200.0, timing_ms=100.0)
        write_full_run(os.path.join(root, 'throughput', 'aoi_A1b', host, 't1'), 'run', 0, host, c=200.0, timing_ms=125.0)

def test_a100_arm_uses_its_own_baseline_and_both_hosts_load_separately(tmp_path):
    import tables
    h, a = str(tmp_path / 'h100'), str(tmp_path / 'a100')
    populate_hosts(h, a)
    for k in range(2):
        write_full_run(a, 'aoi_A1b', k, 'lab-a100', c=150.0)
    with pytest.raises(HostPoolingError):                     # pooling both roots into one grouping is refused
        group(load_runs(h) + load_runs(a))
    text = tables.build(h, str(tmp_path / 't.md'), a)
    cells = [c.strip() for c in next(l for l in text.splitlines() if l.startswith('| A1b ')).strip('|').split('|')]
    assert cells[3].startswith('-') and cells[3].endswith('[A100]')
    assert cells[5].startswith('-')                           # H100 throughput: 125 vs 100 ms per step
    assert '/' in cells[7] and '%' in cells[7]                # the split is shown because the transfer check passes
    assert '| **result** | | all of the above | **pass** |' in text

def test_a100_split_is_withheld_when_the_transfer_check_fails(tmp_path):
    import tables
    h, a = str(tmp_path / 'h100'), str(tmp_path / 'a100')
    populate_hosts(h, a, a100_offset=0.03)                    # the A100 baseline sits 0.03 above the H100's
    for k in range(2):
        write_full_run(a, 'aoi_A1b', k, 'lab-a100', c=150.0, a=3.23)
    text = tables.build(h, str(tmp_path / 't.md'), a)
    cells = [c.strip() for c in next(l for l in text.splitlines() if l.startswith('| A1b ')).strip('|').split('|')]
    assert cells[3].endswith('[A100]') and cells[7].startswith('unsupported')
    assert '**FAIL**' in text

def test_welch_combination_matches_hand_calculation():
    from transfer import combo
    est, hw = combo([[1.0, 3.0], [0.0, 2.0]], [1, -1])        # means 2 and 1; var terms 1 and 1; df = 2; t = 4.303
    assert est == pytest.approx(1.0) and hw == pytest.approx(4.303 * 2 ** 0.5)
    assert combo([[1.0], [0.0, 2.0]], [1, -1])[1] is None     # a single value has no interval

def test_rq3b_table_pairs_seed_zero_at_every_scale(tmp_path):
    import tables
    a = str(tmp_path / 'a100')
    for name, loss in (('fp_r11', 3.30), ('aoi_A3', 3.27), ('rq3b_s0.5_fp_r11', 3.40), ('rq3b_s0.5_aoi_A3', 3.37)):
        write_full_run(a, name, 0, 'lab-a100', c=200.0, a=loss)
    ga = group(load_runs(a))
    text, deltas = tables.rq3b_table(ga)
    assert set(deltas['aoi_A3']) == {0.5, 1.0} and deltas['aoi_A3'][0.5] == pytest.approx(deltas['aoi_A3'][1.0], abs=0.01)
    assert 'not run' in text                                  # aoi_A6 has no runs in this toy dataset
