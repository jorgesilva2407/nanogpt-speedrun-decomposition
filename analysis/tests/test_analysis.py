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
