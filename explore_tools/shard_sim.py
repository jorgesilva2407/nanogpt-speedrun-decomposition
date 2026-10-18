"""Pure-Python simulation of which shard/offset each micro-batch reads, for the record loaders.
Shards: 100,000,000 tokens each (all fineweb10B train shards; '3200000000 across 32 files' in the #17/#18 logs).
S = 65,536 tokens per micro-batch, G = 524,288 tokens per step."""
L, S, G = 100_000_000, 65536, 524288

def legacy(W, steps, strict=True, plus1=True):
    """#11-#14 (`> len` with +1) / #15-#17 (`>= len` with +1) loaders. Returns per step the sorted list of
    (shard, offset) read by all micro-batches. A = G // (S*W) accumulation calls per step."""
    A = G // (S * W)
    st = [dict(shard=0, pos=r * S) for r in range(W)]
    out = []
    for _ in range(steps):
        reads = []
        for _a in range(A):
            for r in range(W):
                s = st[r]
                reads.append((s['shard'], s['pos']))
                s['pos'] += S * W
                lim = s['pos'] + S * W + (1 if plus1 else 0)
                if (lim > L) if strict else (lim >= L):
                    s['shard'] += 1; s['pos'] = r * S
        out.append(sorted(reads))
    return out

def r18(W, steps):
    out = []; shard, cp = 0, 0
    for _ in range(steps):
        if cp + G + 1 >= L:
            shard += 1; cp = 0
        reads = [(shard, cp + v * S) for v in range(G // S)]
        cp += G
        out.append(sorted(reads))
    return out

N = 1500
a8, a1 = legacy(8, N), legacy(1, N)
b8, b1 = r18(8, N), r18(1, N)
first = lambda x, y: next((i for i in range(N) if x[i] != y[i]), None)
print('r11-style loader, W=8 vs W=1: first differing step =', first(a8, a1))
print('r18 loader, W=8 vs W=1: first differing step =', first(b8, b1))
print('r11-style W=8 vs r18 W=8: first differing step =', first(a8, b8))
i = first(a8, b8); print('  at that step r11-style W=8 reads', a8[i][:2], '...', a8[i][-2:]); print('  r18 reads', b8[i][:2], '...', b8[i][-2:])
for st in (188, 189, 190):
    print(f'  step {st}: r11 W=8 shards per micro-batch:', [s for s, _ in a8[st]])
c8 = legacy(8, N, strict=False)
print('#15-#17 rule (>=) W=8 vs #11-#14 rule (>) W=8: first differing step =', first(a8, c8))
