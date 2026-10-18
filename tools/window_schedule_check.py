"""Pure-Python check that the window-schedule expressions used by different records give the same integers.

Forms (copied from the snapshots, with q=64, start=64, end=1792):
  r13: q*((step/N * (end - start) + start)//q)                          tokens  (float floor, then int tensor)
  r15: q * int((start + (end - start) * step / N) // q)                 tokens
Later records add block-count forms (see FORMS). Checked for every step 0..N and every N in the presets and
their RQ3b scalings (0.5, 0.75). Exit 1 if any two forms disagree where they are claimed equal.
Usage: python -I tools/window_schedule_check.py
"""
import sys

Q, S0, S1 = 64, 64, 1792

def r13(step, N):
    return int(Q * ((step / N * (S1 - S0) + S0) // Q))

def r15(step, N):
    return Q * int((S0 + (S1 - S0) * step / N) // Q)

def blocks_from_tokens_ceil(w):          # r15 manual block mask: (sliding_window + 127) // 128
    return (w + 128 - 1) // 128

def r16(step, N):                        # r16/r17: int(((1 - f) * 64 + f * 1792 + 64) // 128) blocks
    f = step / N
    return int(((1 - f) * 64 + f * 1792 + 64) // 128)

def r18(step, N):                        # r18: int(((1 - x) * 128 + x * 1856) // 128) blocks
    x = step / N
    return int(((1 - x) * 128 + x * 1856) // 128)

PRESET_N = [1875, 1750, 1530, 1480, 1490, 1390]
def all_N():
    out = set()
    for n in PRESET_N:
        out |= {n, int(n * 0.5), int(n * 0.75), round(n * 0.5), round(n * 0.75)}
    return sorted(out)

# (name, predicate, N values at which the replay relies on the claim = the records' own num_iterations)
CLAIMS = [
    ('r13 tokens == r15 tokens', lambda s, n: r13(s, n) == r15(s, n), [1750, 1530, 1480]),
    ('ceil(r15 tokens / 128) == r16 blocks', lambda s, n: blocks_from_tokens_ceil(r15(s, n)) == r16(s, n), [1480]),
    ('r16 blocks == r18 blocks', lambda s, n: r16(s, n) == r18(s, n), [1480, 1490, 1390]),
]

def main():
    bad = 0
    for name, claim, record_N in CLAIMS:
        fails = [(n, s) for n in record_N for s in range(n + 1) if not claim(s, n)]
        print(f'{name}: {"holds" if not fails else "FAILS"} at record N {record_N} (every step)')
        bad += bool(fails)
        other = {n: [s for s in range(n + 1) if not claim(s, n)] for n in all_N() if n not in record_N}
        other = {n: v for n, v in other.items() if v}
        print(f'    info, other N (presets/RQ3b scalings): ' + ('agrees everywhere' if not other else
              '; '.join(f'N={n} differs at steps {v}' for n, v in other.items())))
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
