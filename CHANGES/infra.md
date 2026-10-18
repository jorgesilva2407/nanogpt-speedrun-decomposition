# Infrastructure commits (not records)

## infra: seeding

`--seed <int>` (required) seeds `random`, `numpy`, `torch` (CPU generator) and `torch.cuda` (all devices),
immediately after config parsing and before any model construction. Nothing else changes.

### Every randomness source in records #11–#18 (verified by grepping all eight snapshots and reading the code)

| source | where | controlled by `--seed`? | consequence for σ |
|---|---|---|---|
| **parameter init** | default module initialisers on the CPU generator: `nn.Embedding` → `normal_(0, 1)` (token embedding; value-embedding tables from #14), `nn.Linear`/`CastedLinear` → `kaiming_uniform_(a=√5)` (q, k, v, c_fc; then `c_proj`s and `lm_head` are zeroed), constants for `lamb`/`lambdas`/`skip_weights`. No record calls an RNG explicitly (`grep` for `rand`, `normal_`, `uniform_`, `manual_seed`, `np.random`, `random.`: 0 hits in all eight snapshots). | **yes** | the only seed-dependent source. Note: shapes differ between configurations (e.g. VE tables), so the *same seed* gives *different* initial weights in different configurations; seeds pair runs within a configuration, not across configurations. Unseeded record runs were still rank-consistent because DDP broadcasts rank 0's parameters at construction. |
| **data order** | sequential shards in sorted filename order, fixed offsets, no shuffling; validation always reads the first `val_tokens` of the val shard | **no** (seed-independent by construction; the speedrun rules forbid changing the token stream) | σ measured across seeds contains **no data-order variance**. Every run of a configuration sees the identical token stream. |
| **dropout** | none: 0 occurrences of `dropout`/`Dropout` in all eight snapshots | n/a | none |
| **GPU nondeterminism** | atomic accumulation in embedding backward, attention backward (SDPA cuDNN at #11, FlexAttention from #12), NCCL reduction order, cuBLAS/Triton reduction order | **no** | present in every run; two same-seed runs will **not** be bitwise identical. |
| **compiler autotuning** | `config.coordinate_descent_tuning = True` (#11–#17) selects Triton configs by timing; the choice can differ between runs/hosts, changing reduction orders | **no** | adds run-to-run numeric variation; absent at #18 (setting removed). |
| **uninitialised memory** | `torch.empty` update buffers in Muon (#15+): always overwritten before use; in #17's uneven split the stale buffer is gathered but never applied | n/a | none on the training path (by code reading). |

**What σ will measure:** variance across seeds = initialisation variance + GPU/compiler nondeterminism, with the
token stream held fixed. Same-seed reruns isolate the nondeterministic part; their agreement criterion in
`SPIKE.md` is therefore "within tolerance", never "bitwise".

**Not verified at runtime:** that the generators listed are the only ones consumed (e.g. by torch.compile or
NCCL internals). Listed in `DEFERRED_RUNTIME_CHECKS.md`.
