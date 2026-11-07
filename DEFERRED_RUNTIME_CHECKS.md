# Deferred runtime checks

Nothing in this repository has been executed with torch or on a GPU (by design: the build task forbids it).
Every item below is **NOT RUN (by design)**. Each lists what would be run later and what would count as a pass.
Static evidence that *was* produced is in `gates/rNN.txt` and `CHANGES/rNN.md`.

Notation: `run_one` = `python launch/run_one.py`; "8-GPU" = `torchrun --nproc_per_node=8`.

## Cross-cutting (infra and compat)

| check | status | how / pass criterion |
|---|---|---|
| seeding controls all initial weights | NOT RUN (by design) | build the model twice with the same `--seed` in one process, compare `state_dict()` tensors bitwise → identical; different seeds → different |
| no other generator consumed before init (torch.compile, NCCL) | NOT RUN (by design) | as above but after DDP construction and one compiled forward |
| unified batch convention, world_size invariance of the token stream | NOT RUN (by design) on real shards; **verified statically** on synthetic shards (`tools/loader_equivalence.py`) | log the first token id of every micro-batch for 400 steps at world_size 1 and 8 → identical sequences |
| unified batch convention, gradient equivalence | NOT RUN (by design) | one step at world_size 1 (accum 8) vs 8 (accum 1), same seed and data → per-parameter max relative difference of gradients at bf16-rounding level (report the value; bitwise equality is not expected) |
| cuDNN SDPA available on the A100 for the dense (#11) path | NOT RUN (by design); statically the sm80–sm90 gate in `sdp_utils.cpp` admits it | one forward of preset r11 on the A100 without `RuntimeError: No available kernel` |
| FlexAttention kernels fit A100 shared memory (head_dim 128, fwd + bwd) | NOT RUN (by design); statically, unpatched torch 2.6.0 picks a **larger A100 backward tile** than the record's nightly: (64, 128, 8 warps, 3 stages) vs (32, 128, 4, 3) (CHANGES/infra.md "compat: torch pin"). **2026-11-07:** the image now backports d2ec7f0756, which restores the nightly's (32, 128, 4, 3) on A100 (launch/Dockerfile.NOTE.md, deviation 2) | one fwd+bwd of preset r12 and r18 on the A100 without an "out of resource: shared memory" error. If it fails: run those arms on H100 (same tile in both builds) or override the backward tile through `kernel_options` (a trainer change; group decision) |
| pinned `torch==2.6.0+cu126` (stable; replaces the pruned nightly, CHANGES/infra.md "compat: torch pin") installs | NOT RUN (by design) | `docker build` of `launch/Dockerfile` succeeds; else report and stop (do not change the pin again without a group decision) |

## r11 (record #11)

| check | status | how / pass criterion |
|---|---|---|
| state_dict load | NOT RUN (by design) | parameter names/shapes equal the snapshot's (static gate PASS: 99 tensors, 162,201,642 parameters); load a state_dict produced by the snapshot script into `train_flagged.py --config configs/r11.yaml` with `strict=True` |
| forward equivalence | NOT RUN (by design) | same weights + same batch through snapshot and flagged models → identical loss (bitwise, same process/device) |
| smoke train | NOT RUN (by design) | 20 steps, preset r11: val at step 0 = 10.8258 (= ln 50304, zero-initialised lm_head; every record log shows exactly this), train loss decreases |
| torch.compile behaviour | NOT RUN (by design) | compile succeeds with `coordinate_descent_tuning`; no graph breaks warned beyond the snapshot's |
| fixed point | NOT RUN (by design) | full run, 8-GPU: val-loss curve within seed noise of the committed log (final 3.2753 at step 3000) |

## r12 (record #12)

| check | status | how / pass criterion |
|---|---|---|
| state_dict load | NOT RUN (by design) | static gate PASS (99 tensors, names/shapes identical); load the snapshot's state_dict into preset r12, `strict=True` |
| forward equivalence | NOT RUN (by design) | same weights + batch: snapshot vs flagged r12 loss bitwise identical; flagged `flex_kernel` only (r11 + kernel) vs r11 loss equal up to rounding (kernel swap) |
| smoke train | NOT RUN (by design) | 20 steps preset r12; step-0 val 10.8258; `create_block_mask` compiles |
| torch.compile behaviour | NOT RUN (by design) | compiled `flex_attention`/`create_block_mask` inside the compiled model compile once (no recompiles per step with `dynamic=False`); also for the per-row causal mask (flex chunked path, B=64 rows) |
| FlexAttention on A100 | NOT RUN (by design) | fwd+bwd of r12 on the A100 without shared-memory errors |
| fixed point | NOT RUN (by design) | full run vs committed log (final 3.2783 at 1875) |

## r13 (record #13)

| check | status | how / pass criterion |
|---|---|---|
| state_dict load | NOT RUN (by design) | static gate PASS (names/shapes identical); `strict=True` load of a snapshot state_dict |
| forward equivalence | NOT RUN (by design) | same weights, batch and step: snapshot vs flagged r13 loss bitwise identical, at a step where the window is 64 and one where it is 1792 |
| smoke train | NOT RUN (by design) | 20 steps preset r13; window tensor follows 64, 64, … (print attn_blocksize) |
| torch.compile behaviour | NOT RUN (by design) | the window is a tensor captured by the mask_mod: no recompilation when its value changes (count recompiles over 200 steps) |
| fixed point | NOT RUN (by design) | full run vs committed logs (7 runs, mean final 3.2770 at 1750) |

## r14 (record #14)

| check | status | how / pass criterion |
|---|---|---|
| state_dict load | NOT RUN (by design) | static gate PASS (100 tensors + 12 buffers identical); `strict=True` load incl. `rotary.inv_freq` buffers |
| forward equivalence | NOT RUN (by design) | snapshot vs flagged r14 bitwise identical loss; additionally assert `rotary.inv_freq.dtype == torch.bfloat16` after `model.bfloat16()` (the v14 reading) |
| smoke train | NOT RUN (by design) | 20 steps preset r14; peak memory recorded (vte output 65,536 × 9,216 bf16 per micro-batch) |
| torch.compile behaviour | NOT RUN (by design) | Rotary's Python-side cache inside the compiled graph: no recompilation after step 1 |
| fixed point | NOT RUN (by design) | full run vs committed logs (38 runs, mean final 3.2776 at 1530) |

## r15 (record #15)

| check | status | how / pass criterion |
|---|---|---|
| state_dict load | NOT RUN (by design) | static gate PASS (100 tensors + 12 buffers); `strict=True` load (`attn.lambdas` shape (2,)) |
| forward equivalence | NOT RUN (by design) | snapshot vs flagged r15 bitwise identical loss |
| Muon distribution bit-identity (`code_opts_15` alone) | NOT RUN (by design) | one optimizer step from identical state with `code_opts_15` on vs off (`muon_ema_momentum` fixed): parameters bitwise identical |
| block mask construction (`blockmask_manual`) | NOT RUN (by design) | materialise both masks for a window of 128 and count selected pairs: flagged/record manual mask = create_block_mask count − 8,128 per query block (`tools/blockmask_check.py`) |
| smoke train | NOT RUN (by design) | 20 steps preset r15 at world_size 1 and 8 (Muon all_gather requires len(params) % W == 0: 48 and 24 → W ∈ {1,2,4,8}) |
| torch.compile behaviour | NOT RUN (by design) | uncompiled `flex_attention` inside the compiled model compiles to a fused kernel (no eager fallback warning) |
| fixed point | NOT RUN (by design) | full run vs committed logs (75 runs, mean final 3.2785 at 1480) |

## r16 (record #16)

| check | status | how / pass criterion |
|---|---|---|
| state_dict load | NOT RUN (by design) | static gate PASS after name map; load with the documented key rename (`embed.`→`transformer.wte.`, `blocks.`→`transformer.h.`), `strict=True` |
| forward equivalence | NOT RUN (by design) | snapshot vs flagged r16 bitwise identical loss; `blockmask_separate` on vs off: equal up to rounding |
| smoke train | NOT RUN (by design) | 20 steps preset r16; `set_stance(skip_guard_eval_unsafe=True)` raises no guard errors after step 5 |
| torch.compile behaviour | NOT RUN (by design) | no recompilation when the block count changes (count over 1480 steps) |
| fixed point | NOT RUN (by design) | full run vs committed logs (40 runs, mean final 3.2785 at 1480) |

## r17 (record #17)

| check | status | how / pass criterion |
|---|---|---|
| state_dict load | NOT RUN (by design) | static gate PASS; load with the key rename; rotary cos/sin are non-persistent buffers (absent from state_dict in both) |
| forward equivalence | NOT RUN (by design) | snapshot vs flagged r17 bitwise identical loss |
| Muon uneven split | NOT RUN (by design) | world_size 8 with drop_attn_layer: 44 square matrices → last round of 4; parameters identical to a world_size-1 step from the same state (up to all_gather exactness) |
| expandable_segments set before CUDA init | NOT RUN (by design) | `torch.cuda.memory._get_allocator_backend()`/allocator settings report expandable segments under preset r17 |
| smoke train | NOT RUN (by design) | 20 steps preset r17 |
| torch.compile behaviour | NOT RUN (by design) | `None` entries in the value-embedding list do not cause graph breaks per layer |
| fixed point | NOT RUN (by design) | full run vs the committed log (final 3.2794 at 1490) |

## r18 (record #18)

| check | status | how / pass criterion |
|---|---|---|
| state_dict load | NOT RUN (by design) | static gate PASS (97 tensors, 275,742,772); load with the key rename, `strict=True` |
| forward equivalence | NOT RUN (by design) | snapshot vs flagged r18 bitwise identical loss; dtypes: block lambdas fp32, rotary cos/sin fp32, embeddings bf16 |
| optimizer coverage | NOT RUN (by design) | the start-up assertion (every parameter in exactly one optimizer) passes for all eight presets |
| smoke train | NOT RUN (by design) | 20 steps preset r18; step-0 val 10.8258; LR multiplier 1.0 until step 834, then t/0.4 |
| torch.compile behaviour | NOT RUN (by design) | no coordinate-descent tuning: compile time shorter than r17's; no recompilation as the window grows |
| fixed point | NOT RUN (by design) | full run vs the committed log (final 3.2785 at 1390) |
| leave-one-out sanity | NOT RUN (by design) | r18 with `logit_softcap: 30` runs (this is arm 7's LOO configuration) |

## arm-muon (optimizer flag)

| check | status | how / pass criterion |
|---|---|---|
| AdamW path builds and steps | NOT RUN (by design) | presets r11 and r18 with `optimizer: adamw`: start-up assertion (every parameter in one optimizer) passes, 20 steps run |
| AdamW LR sweep sanity | NOT RUN (by design) | the three swept LRs do not diverge in 200 steps |

## infra: run logging

| check | status | how / pass criterion |
|---|---|---|
| files written | NOT RUN (by design) | 50-step run with `--out-dir`: metrics.jsonl has 50 train rows + val rows, timing.jsonl 50 rows, meta.json complete, DONE present |
| train loss world-size invariance | NOT RUN (by design) | same seed at world_size 1 and 8: per-step train loss equal up to rounding (mean over the same 8 micro-batches) |
| timing overhead | NOT RUN (by design) | step time with and without the CUDA-event pair differs by < 0.5% |
| subset validation | NOT RUN (by design) | `val_final_third_fraction: 0.2` evaluates 32 global micro-batches (4 per rank at 8 GPUs) |
