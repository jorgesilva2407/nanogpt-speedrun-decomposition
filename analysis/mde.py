"""Minimum detectable effects (95% interval half-widths), in loss and in step-equivalents.

RQ1 (two-arm difference of means, n seeds each, common sigma):
    Var(mean_A - mean_B) = 2 sigma^2 / n      ->  half-width = z * sigma * sqrt(2 / n)
RQ2 (difference of differences, LOO gap minus AOI gap, four independent means of n seeds):
    Var = 4 sigma^2 / n                       ->  half-width = z * 2 sigma / sqrt(n)
Step equivalents divide a loss half-width by the local slope |dL/dstep| near the target. The proposal uses Track 3's
published sensitivity, 0.0045 loss per 100 steps (slope 4.5e-5 per step); `local_slope` estimates it from a curve.

Worked numbers at sigma = 0.0013, z = 1.96, slope 4.5e-5 (the proposal's 5.1 table):
    RQ1 n=3: 1.96 * 0.0013 * sqrt(2/3) = 0.0020804 loss -> 46.2 steps   (proposal: +-46)
    RQ1 n=5: 1.96 * 0.0013 * sqrt(2/5) = 0.0016115 loss -> 35.8 steps   (proposal: +-36)
    RQ2 n=3: 1.96 * 2 * 0.0013 / sqrt(3) = 0.0029422 loss -> 65.4 steps (proposal: +-65)
    RQ2 n=5: 1.96 * 2 * 0.0013 / sqrt(5) = 0.0022790 loss -> 50.6 steps (proposal: +-51)
These are z (normal) intervals, as in the proposal. With n this small a t interval is wider: e.g. RQ1 n=3 uses
df = 4, t = 2.776 -> 65.5 steps instead of 46.2 (`halfwidth(..., use_t=True)`).
"""
import math
import numpy as np

Z95 = 1.959963984540054
TRACK3_SLOPE = 0.0045 / 100   # loss per step near the target (Track 3's published sensitivity)

# two-sided 97.5% t quantiles for small df (exact values from tables)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 16: 2.120, 20: 2.086}

def _crit(df, use_t):
    if not use_t:
        return Z95
    return T975.get(df) or T975[min(T975, key=lambda k: abs(k - df))]

def rq1_halfwidth(sigma, n, use_t=False):
    """Loss half-width for a difference of two means of n seeds each."""
    return _crit(2 * n - 2, use_t) * sigma * math.sqrt(2.0 / n)

def rq2_halfwidth(sigma, n, use_t=False):
    """Loss half-width for (LOO gap) - (AOI gap): four means of n seeds."""
    return _crit(4 * n - 4, use_t) * 2.0 * sigma / math.sqrt(n)

def to_steps(loss_halfwidth, slope=TRACK3_SLOPE):
    return loss_halfwidth / abs(slope)

def local_slope(steps, losses, at_loss, window=3):
    """|dL/dstep| near the point where the curve reaches at_loss: least-squares slope over `window` points on each
    side of the first crossing (falls back to the last points if never crossed)."""
    steps, losses = np.asarray(steps, float), np.asarray(losses, float)
    below = np.nonzero(losses <= at_loss)[0]
    i = int(below[0]) if len(below) else len(steps) - 1
    lo, hi = max(0, i - window), min(len(steps), i + window + 1)
    if hi - lo < 2:
        lo, hi = max(0, len(steps) - 2), len(steps)
    a = np.polyfit(steps[lo:hi], losses[lo:hi], 1)[0]
    return abs(a)

def proposal_table(sigma=0.0013, slope=TRACK3_SLOPE):
    out = {}
    for n in (3, 5):
        out[n] = dict(rq1=to_steps(rq1_halfwidth(sigma, n), slope), rq2=to_steps(rq2_halfwidth(sigma, n), slope))
    return out

if __name__ == '__main__':
    for n, v in proposal_table().items():
        print(f'n={n}: RQ1 +-{v["rq1"]:.1f} steps, RQ2 +-{v["rq2"]:.1f} steps (z); '
              f'RQ1 with t: +-{to_steps(rq1_halfwidth(0.0013, n, use_t=True)):.1f}')
