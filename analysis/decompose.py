"""Additive wall-clock decomposition with a symmetric (Shapley) split of the cross term (proposal 5.1, RQ1).

With S = steps to target and T = time per step, for a change (S0, T0) -> (S1, T1):
    dTime = S1*T1 - S0*T0 = dS*T0 + S0*dT + dS*dT          (exact identity)
    sample-efficiency share = dS*T0 + dS*dT/2
    throughput share        = S0*dT + dS*dT/2
The cross term dS*dT is split half/half (the average over both orders of expansion) and its magnitude is reported.
T is the mean of the per-step trace over a matched step range with compile warm-up excluded (not a median: arms 1-2
have a time-varying step cost by construction).
"""
import numpy as np

def mean_step_time(steps, ms, warmup, start=None, stop=None):
    """Mean per-step time (ms) over steps in [start, stop) excluding warm-up-marked steps."""
    steps, ms, warmup = np.asarray(steps, float), np.asarray(ms, float), np.asarray(warmup, bool)
    m = ~warmup
    if start is not None:
        m &= steps >= start
    if stop is not None:
        m &= steps < stop
    if not m.any():
        raise ValueError('empty step range for the timing mean')
    return float(ms[m].mean())

def decompose(S0, T0, S1, T1):
    dS, dT = S1 - S0, T1 - T0
    total = S1 * T1 - S0 * T0
    cross = dS * dT
    se = dS * T0 + cross / 2
    th = S0 * dT + cross / 2
    assert abs((se + th) - total) <= 1e-9 * max(1.0, abs(total))
    frac = (lambda x: x / total) if total != 0 else (lambda x: float('nan'))
    return dict(d_time=total, sample_efficiency=se, throughput=th, cross_term=cross,
                sample_efficiency_fraction=frac(se), throughput_fraction=frac(th),
                cross_term_fraction=frac(cross))
