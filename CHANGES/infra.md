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

## infra: unified batch convention

**Convention (the #18 style, applied from the floor up).** `global_batch_tokens = 524,288` per optimizer step
(fixed), `micro_batch_tokens = 65,536` per device per forward/backward (asserted `== 65536`: never reduced),
`accum = global // (micro × world_size)` with `assert global % (micro × world_size) == 0`. `vranks = global //
micro = 8` is the number of micro-batches per step, i.e. the number of ranks of the record's 8-GPU run.

**Data addressing.** Micro-batch `a` of rank `r` is *virtual rank* `v = r·accum + a`. The loader keeps one
(shard, position) state per owned virtual rank and applies the record's own read/advance rule to it with the
record's 8-GPU constants (`pos₀ = v·S`, stride `S·vranks`, `S = micro`). Shard data is cached and shared
between the virtual ranks a device owns. The record's literal loader is the special case `vranks = world_size`,
`owned = [rank]` (this is exactly how the validation loader is built).

### Why this is a no-op for record #11 at world_size = 8 (on paper, step by step)

1. **Accumulation count.** Record: `train_accumulation_steps = batch_size // (B·W) = 512 // (64·8) = 1`.
   Flagged: `accum = 524288 // (65536·8) = 1`. Same.
2. **Micro-batch shape.** Record: `B = device_batch_size = 64`, `T = 1024`. Flagged: `B = micro // T =
   65536 // 1024 = 64`, `T = 1024`. Same `(64, 1024)` layout, `.view(B, T)` as before.
3. **Which tokens each rank reads.** With `accum = 1`, rank `r` owns exactly `v = r`. Its state starts at
   `pos = r·B·T` in shard 0 (record: `process_rank·B·T`), advances by `B·T·8` (record: `B·T·num_processes`,
   `num_processes = 8`) and switches shard when `pos + (B·T·8 + 1) > len(tokens)`, resetting to `r·B·T` (record:
   identical expression and reset). Same reads, same switch points.
4. **First batch twice.** Record: `x, y = next_batch()` before the loop, then `train_loader.reset()`, so step 0
   trains on the prefetched first batch and step 1 on the re-read first batch. Flagged
   (`loader_replay_first_round: true`): the first round over the owned virtual ranks is served without
   advancing, the loop's in-step prefetch then consumes the first batch, so steps 0 and 1 both train on it and
   step k ≥ 2 trains on batch k−1. Same. The explicit `train_loader.reset()` is removed because the replay flag
   replaces it (keeping both would replay twice).
5. **Validation.** `val_steps = val_tokens // (B·T·W)` is unchanged and the val loader is constructed with
   `vranks = world_size, owned = [rank]`, i.e. literally the record's loader. Same reads.
6. **Loss scaling across accumulation.** Unchanged code: each micro-step computes the mean cross-entropy over
   its 65,536 tokens and calls `backward()` (under `no_sync()` except on the last micro-step); after the loop
   every gradient is divided by `train_accumulation_steps`. With `accum = 1` the division is by 1, which is
   exact in IEEE arithmetic, and `no_sync()` is never entered — as in the record.
7. **Everything after the gradients** (Muon, Adam, schedule, momentum warmup) is untouched.

Hence at world_size 8 the flagged r11 path performs the same operations on the same data in the same order as
record #11. **This is an argument from code reading; it has not been executed** (`DEFERRED_RUNTIME_CHECKS.md`).

### What the convention guarantees at world_size ∈ {1, 2, 4}

* **Data.** Each step trains on the same 8 micro-batches as the 8-GPU record run (virtual rank `v` reads what
  rank `v` read), including across shard boundaries and the first-batch replay. This is *not* what the record's
  code would literally do on fewer GPUs: its loader's shard switch depends on `world_size`, and its
  prefetch/reset quirk becomes per-micro-step (see `EXPLORATION.md` §8: with 100M-token shards the literal
  1-GPU and 8-GPU streams diverge at step 189). Verified by executing only the loader classes on synthetic
  shards: `python -I tools/loader_equivalence.py <r11 snapshot> train_flagged.py configs/r11.yaml --protocol r11`
  → identical micro-batch offsets at world_size 1, 2, 4, 8 for 600 steps on 8 synthetic shards of ~6,100
  tokens (scaled-down geometry, same 8-micro-batch ratio): 73 steps at which some virtual rank switches shard,
  268 steps whose 8 micro-batches come from two different shards (under the record's per-rank rule, ranks switch
  at different steps and drift apart). Negative control: disabling the replay makes every world size differ
  from step 1.
* **Gradient.** DDP averages over `W` ranks the locally accumulated sum of `accum` micro-step gradients; the
  division by `accum` then gives `(1/(W·accum)) Σ_{r,a} ∇L_{r,a} = (1/8) Σ_v ∇L_v`, the same quantity the
  8-GPU run computes as `(1/8) Σ_r ∇L_r`. Equal in exact arithmetic; floating-point summation order differs
  (local bf16 accumulation of embedding/scalar gradients over `accum` steps vs NCCL's cross-rank sum), so
  results are equal only up to rounding. Not verified at runtime.
* **Muon.** Record #11's Muon assigns parameter `i` to rank `i % W` and all-reduces a bf16 buffer where only
  the owner writes non-zeros (x + 0 = x exactly); every parameter's update is computed from the same
  all-reduced gradient and its own momentum state, so the update does not depend on `W`.
* **Later presets.** Record #18's literal code *sums* micro-batch gradients without dividing (identical at
  `accum = 1`, i.e. on 8 GPUs). The flagged trainer always averages; on fewer GPUs this reproduces the 8-GPU
  computation rather than #18's literal fewer-GPU behaviour. Recorded again in `ledger/r18.yaml`.

## compat: drift established statically against the pinned stack

Pinned stack (unchanged, from upstream #18's Dockerfile): `nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04`, Python
3.12.7, `torch==2.6.0.dev20241231+cu126`. Sources read (not executed), saved under `ext/torch_src/`:
`sdp_utils.cpp` at main `09c950c` (last change before the nightly) and at `v2.6.0`;
`flex_attention.py`, `torch/compiler/__init__.py`, `torch/nn/parameter.py`, `torch/_inductor/kernel/flex_attention.py`
at main `8d9ff9c` (2024-12-31 23:45 UTC, the nightly's day).

| item | finding | action |
|---|---|---|
| `"""…S_{ii}' \sim…"""` in `zeropower_via_newtonschulz5` | invalid escape `\s` is a `SyntaxWarning` on Python ≥ 3.12 (docstring value unchanged: `\s` is kept literally) | **compat commit**: `r"""` prefix (same string value; this is also how `ff93d20b` writes it) |
| #11 forces cuDNN SDPA (`enable_cudnn_sdp(True)`, flash/mem-efficient/math off) | `check_cudnn_hardware_support` accepts sm80–sm90 and `check_cudnn_tensor_shapes` accepts head_dim 128, seq 1024 → A100 (sm80) and H100 (sm90) are both inside the gate; nothing else would be available, so a failing gate would raise | none; runtime availability (bundled cuDNN ≥ 8.9.3) deferred |
| FlexAttention API used by #12–#18 | `create_block_mask(..., _compile=False)` parameter exists; `BlockMask.from_kv_blocks(kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices, BLOCK_SIZE, mask_mod, seq_lengths)`; `flex_attention(..., enable_gqa=False, ...)` | none |
| `torch.compiler.set_stance(skip_guard_eval_unsafe=…)` (#16–#17), `nn.Buffer` (#17) | both present | none |
| FlexAttention kernels on A100 | `_a100_default_config` and an sm80 backward branch exist for head_dim 128; whether they fit A100 shared memory is a runtime question | deferred (spike item 4) |
| nightly wheel availability | PyTorch prunes old nightlies from its index; whether `2.6.0.dev20241231+cu126` can still be installed cannot be established statically | deferred; pins not changed. **Superseded 2026-10-18: the nightly is gone; see "compat: torch pin" below** |

## compat: torch pin changed to the stable 2.6.0 release (2026-10-18)

**Decision (user, 2026-10-18):** use `torch==2.6.0+cu126` from `https://download.pytorch.org/whl/cu126` instead of
upstream #18's `torch==2.6.0.dev20241231+cu126`. **Why:** on 2026-10-18 the cu126 nightly index only held
`2.15.0.dev` builds (Aug–Sep 2026), so the pinned nightly can no longer be installed (the SPIKE S4a failure case).
CUDA 12.6.2, cuDNN image and Python 3.12.7 are unchanged.

What was checked statically (sources at tag `v2.6.0` in `ext/torch_src/v2.6.0/`, compared with the nightly's day,
main `8d9ff9c`, in `ext/torch_src/`; `explore_tools/fetch_sources.sh` re-creates both):

| item | finding |
|---|---|
| resolution | `torch==2.6.0+cu126` resolves for linux x86_64 / Python 3.12 from the PyTorch index alone (`uv pip compile`): `triton==3.2.0`, `nvidia-cudnn-cu12==9.5.1.17`, `nvidia-nccl-cu12==2.21.5` |
| relative age | the 2024-12-31 nightly is **newer** than the v2.6.0 release: it contains code v2.6.0 lacks (`torch.compiler.is_exporting`, inductor `prologue_fusion`, FlexAttention's `WRITE_DQ` default) |
| every torch API the trainer calls | present with the arguments the trainer passes, in both: `tools/torch_api_check.py` → `gates/torch_api.txt` (12 checks, 0 problems) |
| `torch/nn/attention/flex_attention.py` | 2 lines differ: the nightly defaults `kernel_options["WRITE_DQ"] = True`, which concerns gradients of tensors captured by a `score_mod`. The trainer passes no `score_mod` (block masks only), so this does not apply |
| cuDNN SDPA gate (#11 dense path) | `check_cudnn_hardware_support` accepts [sm80, sm90] at v2.6.0 too: A100 and H100 both pass |
| FlexAttention **backward tile sizes on A100** (head_dim 128, every preset: 768 / 6 heads) | **differ**: nightly `(BLOCK_M, BLOCK_N, warps, stages) = (32, 128, 4, 3)`; v2.6.0 `(64, 128, 8, 3)`, a larger tile using more shared memory. H100 is `(64, 128, 8, 3)` in both. Tile sizes change summation order, not the computed function. Whether v2.6.0's tile fits A100 shared memory is a runtime question: added to `DEFERRED_RUNTIME_CHECKS.md` and SPIKE S4 |
| other inductor changes (32 lines of `config.py`, 136 of `kernel/flex_attention.py`) | refactors and options the trainer does not set; `coordinate_descent_tuning` (the one option it sets) exists in both |

**Consequences.**
* **Within the study**, every configuration runs on the same build, so comparisons between arms are unaffected by
  the choice.
* **Against the record logs**, results carry build drift. This was already the case: the eight record logs were
  produced by six different torch builds (`EXPLORATION.md` §2, item 7). For example, #12–#16 ran on cu124 nightlies
  from Nov–Dec 2024.
* **Not established without a GPU:** whether kernels, autotuning choices or numerics differ from the nightly's.

## compat: FlexAttention backport into torch 2.6.0 (2026-11-07)

**Correction to the table above.** Row "FlexAttention backward tile sizes" compared `_get_nv_config` only. v2.6.0's
`flex_attention_backward` also skips every config with `num_warps == 8` (`d35358b`, #141653), which drops the H100
default `(64, 128, 8, 3)` and leaves no choice: the first full-size run (smoke test 2, fp_r18 seed 0, H100 SXM) failed
in `loss.backward()` with `NoValidChoicesError` (target `flex_attention_backward`). The nightly contains the follow-up
`d2ec7f0756` (#143299) that removes the skip and also sets the A100 default to `(32, 128, 4, 3)`.

**Fix:** the image applies `d2ec7f0756` verbatim (`launch/patch_torch.py`, `launch/Dockerfile.NOTE.md`
deviation 2). Backward default configs now equal the nightly's on H100 and A100. The trainer is unchanged. Arms that
use SDPA (#11 dense path) never reached this code.

## compat: cuDNN SDPA meta-registration backport into torch 2.6.0 (2026-11-07)

**Second correction to "compat: torch pin".** The cuDNN SDPA gate was checked statically (row "cuDNN SDPA gate"), but
not the stride contract between the kernel and its meta function. v2.6.0 (and #18's nightly) contain `1565eba4b4`
(#138354: cuDNN SDPA output follows the query's memory layout) but not `ec93aa7f84` (#148921, March 2025: the matching
meta-registration fix). Under `torch.compile` the dense #11 path then fails `assert_size_stride` in the first forward
(throughput `fp_r11`, H100 SXM). **Fix:** the image applies `ec93aa7f84` verbatim (`launch/patch_torch.py`,
`launch/Dockerfile.NOTE.md` deviation 3). The trainer is unchanged; 8 configurations (all with `flex_kernel` off) were
affected.

## launch: p1 job order by priority (2026-11-07)

Order only: tier p1 runs the same 42 jobs, and no configuration, seed or trainer code changes. Seeds stay
breadth-first. Within a seed round the order is the five both-direction pairs (A2, A4b, A6, A6b, A7, each add-one-in
next to its leave-one-out), then the add-one-in-only H1 arms (A1a, A3), then `ctrl_A8_r18`. `ctrl_A8_r11` (D13's
first cut) runs after all three rounds. Before, the order was longest first, so a shortfall at the end of the sweep
cost `loo_*` seed 2. Now it costs `ctrl_A8_r11` first. `launch/make_jobs.py: PRIORITY, LAST`; the other tiers are
unchanged.

## analysis: primary table uses the never-crossed fallback (2026-11-08)

Analysis only: no run, configuration or trainer code changes, and every run already uploaded can be re-tabulated.
`analysis/tables.py: per_seed_steps` dropped every run that never reached 3.28, although `persistence.py` defines the
fallback for that case. With the first results (2 of 3 `fp_r18` seeds and every `loo_*` run end just above 3.28), the
leave-one-out column was empty and the `fp_r11` baseline mean kept only its two faster seeds. A run that never
crosses now counts as budget step + step-equivalent shortfall, (loss at budget − 3.28) / local slope, the
`persistence.py` rule; it is still dropped only if the local slope is not positive. Each Δ cell reports how many
seeds of the baseline and of the arm are extrapolated (`[extrap. k/n base, k/n arm]`). Test:
`analysis/tests/test_analysis.py: test_primary_table_uses_fallback_for_runs_that_never_cross`.

## analysis: arms without learning runs are throughput-only in the primary table (2026-11-09)

Analysis only; no run, configuration or trainer change. The P2 arms (A1b, A4, A5) and P3 (A5b) were not run (see
`DEVIATIONS.md`), so they have throughput but no learning curves. `analysis/tables.py` now (a) shows `not run` in a steps
column when the configuration is in the matrix but has no completed run, instead of `n/a`; (b) adds a `Δ tokens/s AOI`
column (r11 → +arm, same session and host as the leave-one-out column); (c) computes both throughput columns whenever the
throughput runs exist, not only when a steps estimate does. Wall-clock splits stay `n/a` for such arms: they need steps.
Test: `analysis/tests/test_analysis.py: test_primary_table_reports_arms_without_learning_runs_as_throughput_only`.

## lab A100: separate driver and protocol; RQ3b configurations (2026-11-10)

New directory `lab_a100/` with `run.sh` and `PROTOCOL.md`. `run.sh` runs the fixed job list of the A100 protocol (phases `anchor`, `gaps`,
`rq2`, `seeds2`, `rq3b`; default `all`, 24 runs) from one invocation inside the study image and uploads each finished run to its own
`RESULTS_REPO`, with `cloud_worker.sh`'s contract (skip runs already uploaded, restart incomplete ones from step 0, halt after 2
consecutive failures, `DRY_RUN=1`), plus upload retries and a compile cache in the mounted work directory. It shares nothing with the H100
tooling except `launch/run_one.py` and the trainer: `launch/` is not touched. The reason for a separate script: no `cloud_worker.sh`
tier selects an arbitrary list of (configuration, seed), and the lab's results must go to their own dataset (the names would collide).

`experiments/study.yaml: rq3b.resolved_arms` is filled with `[A2, A3, A4b, A6, A7]` (RQ3 "runs only on arms resolved in RQ1"; the
arms with an effect beyond seed noise in the H100 sweep) and `experiments/matrix.py` regenerated: 22 configurations `rq3b_s0.5_*` and
`rq3b_s0.75_*` (the two fixed points and both directions of the resolved arms, scaled budgets), added to `experiments/runs/` and
`matrix.json`. The 21 existing configurations, their config hashes and the throughput job list are unchanged. They live on the new tag
`a100-v1`; `study-freeze` and the trainer are unchanged. Only 6 of the 22 are scheduled (`fp_r11`, `aoi_A6`, `aoi_A3` at both scales, one seed).

## analysis: A100 dataset, transfer check, RQ3b, A8 and whole-run timing in the table (2026-11-14)

Analysis only; no run, configuration or trainer changes. `analysis/tables.py` takes `--results-a100` (the separate A100 dataset) and never
merges the two roots (both hold `fp_r11` and `fp_r18`; `load.group` still refuses mixed hosts). Arms the H100 never ran (A1b, A4, A5, A5b
add-one-in, A5b leave-one-out) are read from the A100 against the A100's own baselines and marked `[A100]`; throughput always comes from the H100
root. New `analysis/transfer.py` evaluates PROTOCOL.md section 3 as written (offset, curve, seeding, bridge; paired offset and `fp_r18` offset
as secondary) and gates the wall-clock split of the A100 arms on it. New columns/sections: add-one-in wall-clock split (arms without a
leave-one-out had none), A8 row (loss gap and the step-ratio lower bound, 1.55× from interpolated crossings of 3 seeds, replacing the 950-step /
1.46× grid reading in DEVIATIONS 3.3), RQ3b table (Δ final loss at 0.5×/0.75×/1.0×, seed 0 against seed 0), and the whole-run step time of the
H100 learning runs next to the 300-step throughput run. The H100-only output is unchanged except the new column and the A8 row (checked cell by
cell against the previous output). Tests: 18 (4 new: two-host loading, gate failure, Welch combination, RQ3b pairing).

## infra: run logging (Step 5; decided now because it cannot be added after the sweep)

* `--out-dir DIR` writes, on the master process:
  * `metrics.jsonl` — `kind=train` rows: step, tokens, train loss (mean over all 8 micro-batches of the step: per-rank
    sums gathered with **one batched all-reduce per flush**, so the value does not depend on world size), wall clock;
    `kind=val` rows: step, tokens, val loss, fraction of the validation set used, wall clock.
  * `timing.jsonl` — every training step: GPU time from a pair of CUDA events around the step (no per-step host sync),
    `compile_warmup` for steps < `measurement.timing_warmup_steps` (10), wall clock. Records are flushed at each
    validation and at the end (both already synchronise).
  * `meta.json` — config hash (sha256 of canonical JSON of flags + hparams + measurement), preset, resolved flags,
    hparams and measurement, seed, git SHA and dirty flag, docker tag and host label (from the launcher's `run:`
    section), hostname, torch/CUDA/cuDNN versions, GPU name, world size, accumulation steps; at the end peak
    `max_memory_allocated`/`max_memory_reserved` and the record-convention training time.
  * `DONE` — written last, after a barrier. No `DONE` means the run is re-run from scratch (files are truncated at
    start); there is no resume code.
* A run refuses to start if the launcher reports a dirty git tree (unless `allow_dirty` is set for a dry run), and if
  the config hash it was given does not match the config.
* `measurement:` config section (never touches training): `val_every_final_third` (v8: 25; adds validations in the
  last third, on top of `hparams.val_loss_every` = 125), `val_final_third_fraction` (1.0; a fraction < 1 evaluates a
  fixed subset — the first micro-batches of the val stream, the same for every run and world size), and
  `timing_warmup_steps` (10). The alternative "fixed 20% subset every 5 steps in the final third" is
  `val_every_final_third: 5, val_final_third_fraction: 0.2` — supported, not the default (open decision).
* `--save-checkpoint`: the final state is saved only when asked (it was always saved before; I/O happens with the clock
  stopped, after the last step). Checkpoints are never read back.
* `--throughput-steps N`: N training steps, no validation, then the same files (used by `launch/throughput.py`).
* With `--out-dir` the per-step console/log line (which called `.item()`, a host sync per step) is not printed; the
  record-style log file still receives the code header and the validation lines.
* Not verified at runtime: event timing overhead, the batched all-reduce, file outputs (`DEFERRED_RUNTIME_CHECKS.md`).
