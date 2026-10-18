"""Pure-Python: which (query, key) pairs does record #15's directly-built block mask drop?

Token-level rule (mask_mod, records #12-#15): causal and q - kv < w (document mask ignored here: it is applied
identically by both constructions inside selected blocks).
#15 block selection: query block a, key block b selected iff a >= b and a - b < ceil(w / 128).
A pair is DROPPED if the token rule allows it but its (a, b) block is not selected.
Counts pairs for one query block far from the sequence start (steady state), for every window the #15
schedule produces (multiples of 64 from 64 to 1792).
Usage: python -I tools/blockmask_check.py
"""
BS = 128

def counts(w):
    nb = (w + BS - 1) // BS
    allowed = dropped = 0
    a = 100                                   # a query block far from the start
    for qi in range(BS):
        q = a * BS + qi
        for kv in range(max(0, q - w + 1), q + 1):   # token rule: q - kv < w, causal
            allowed += 1
            if a - kv // BS >= nb:
                dropped += 1
    return allowed, dropped

if __name__ == '__main__':
    print('window  blocks  allowed_pairs_per_query_block  dropped  dropped_%  (dropped keys are at distance)')
    for w in range(64, 1793, 64):
        al, dr = counts(w)
        nb = (w + BS - 1) // BS
        lo = (nb - 1) * BS + 1
        print(f'{w:6d} {nb:6d} {al:12d} {dr:10d} {100*dr/al:8.2f}%   ({lo}..{w-1} for some queries)' if dr else f'{w:6d} {nb:6d} {al:12d} {dr:10d} {0:8.2f}%')
