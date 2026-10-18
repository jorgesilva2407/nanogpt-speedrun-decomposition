"""Steps-to-threshold from validation curves (RQ1 target 3.28; RQ3a thresholds 3.40/3.35/3.30/3.28).

For each curve two numbers are reported (proposal 5.1):
  grid          the first logged step whose validation loss is <= threshold
  interpolated  linear interpolation between the last logged point above the threshold and the first at/below it
If a curve never reaches the threshold: `crossed = False`, and the fallback is the loss at a fixed token budget (the
curve interpolated at that budget) plus a step-equivalent shortfall = (loss_at_budget - threshold) / local slope.
RQ3a is a screening instrument: thresholds are crossed at different points of the LR schedule.
"""
import numpy as np
from mde import local_slope

RQ3A_THRESHOLDS = (3.40, 3.35, 3.30, 3.28)

def steps_to_threshold(steps, losses, threshold, budget_step=None):
    steps, losses = np.asarray(steps, float), np.asarray(losses, float)
    order = np.argsort(steps)
    steps, losses = steps[order], losses[order]
    below = np.nonzero(losses <= threshold)[0]
    if len(below):
        i = int(below[0])
        grid = float(steps[i])
        if i == 0:
            interp = grid
        else:
            s0, s1, l0, l1 = steps[i - 1], steps[i], losses[i - 1], losses[i]
            interp = float(s0 + (l0 - threshold) * (s1 - s0) / (l0 - l1))
        return dict(crossed=True, grid=grid, interpolated=interp)
    b = float(steps[-1] if budget_step is None else budget_step)
    loss_b = float(np.interp(b, steps, losses))
    slope = local_slope(steps, losses, threshold)
    return dict(crossed=False, grid=None, interpolated=None, budget_step=b, loss_at_budget=loss_b,
                step_equivalent_shortfall=(loss_b - threshold) / slope if slope > 0 else float('inf'))

def thresholds_table(steps, losses, thresholds=RQ3A_THRESHOLDS, budget_step=None):
    return {t: steps_to_threshold(steps, losses, t, budget_step) for t in thresholds}

def mean_ci(values, use_t=True):
    """Mean and 95% half-width across seeds (t interval)."""
    from mde import _crit
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) == 0:
        return dict(n=0, mean=None, halfwidth=None)
    if len(v) == 1:
        return dict(n=1, mean=float(v[0]), halfwidth=None)
    return dict(n=len(v), mean=float(v.mean()), halfwidth=float(_crit(len(v) - 1, use_t) * v.std(ddof=1) / np.sqrt(len(v))))

def delta_ci(a, b, use_t=True):
    """Difference of means b - a with a Welch-style 95% half-width (independent seeds)."""
    from mde import _crit
    a = np.asarray([x for x in a if x is not None], float); b = np.asarray([x for x in b if x is not None], float)
    if len(a) < 2 or len(b) < 2:
        return dict(delta=(float(b.mean() - a.mean()) if len(a) and len(b) else None), halfwidth=None)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if va + vb > 0 else len(a) + len(b) - 2
    return dict(delta=float(b.mean() - a.mean()), halfwidth=float(_crit(int(round(df)), use_t) * se))
