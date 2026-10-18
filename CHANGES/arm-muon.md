# arm-muon — optimizer flag (positive control) and the arm → flag → dependency table

## What changed
* Flag `optimizer: muon | adamw` (FLAGS default and every preset: `muon`). With `adamw`, the hidden matrices — exactly
  the parameters Muon receives — are optimised by `torch.optim.AdamW(lr=adamw_lr, betas=adamw_betas,
  weight_decay=adamw_weight_decay, fused=True)`; embeddings, head and scalars keep their Adam optimisers; the Muon
  momentum warmup is skipped (nothing to warm up). Works with both optimizer layouts (four optimizers at r11–r17,
  grouped Adam + matrix optimizer at r18 / `code_opts_18`).
* Config: `adamw_lr` 0.0018, `adamw_betas` [0.9, 0.95], `adamw_weight_decay` 0. Source: record #2's README ("was
  learning rate 0.0018, warmup=250, warmdown=2000, betas=(0.9, 0.95) IIRC") — a different architecture and batch, so
  these are starting points, not tuned values; `experiments/arms.yaml` carries the LR sweep (0.0009, 0.0018, 0.0036).
* `experiments/arms.yaml`: every arm as flag/config values r11 → r18, with separable sub-arms.
* `tools/arm_table.py`: derives the table below from `arms.yaml` and the trainer's `REQUIRES` (read with ast).
* `REQUIRES`: `block_window` now requires `flex_kernel, long_context` instead of `blockmask_manual`; its code
  path builds its own block mask, so the old prerequisite was not mechanical and wrongly chained A5 to A4b.
  "Superseded" relations are documented next to `REQUIRES` (`block_window` makes `blockmask_manual` inert;
  `ve_split`/`ve_sparsify` build their own layer→table mapping).

## Static checks (`gates/r11.txt`, `gates/r18.txt`)
* r11 (all record flags off, `optimizer: muon`) vs the r11 snapshot: constants, flag detectors (incl. `optimizer`),
  parameters (162,201,642) and loader stream — PASS.
* r18 (all record flags on, `optimizer: muon`) vs the r18 snapshot: constants, flag detectors, parameters
  (275,742,772), loader stream — PASS.
* Negative control across snapshots: r11 preset vs r18 snapshot 29 constant mismatches; r18 preset vs r11 snapshot 29.
* All eight presets re-gated after this change: PASS.
* The AdamW path changes no parameter (inventory identical); its runtime behaviour is unverified
  (`DEFERRED_RUNTIME_CHECKS.md`).

## Arm → flag → dependency table (`python -I tools/arm_table.py --markdown`)
| arm | name | record | switches | requires (direct) | requires (transitive) | prerequisite arms |
|---|---|---|---|---|---|---|
| A1a | FlexAttention kernel | #12 | flex_kernel | - | - | - |
| A1b | 64K-ctx sliding window + document masking | #12 | long_context, doc_masking | flex_kernel | flex_kernel | A1a |
| A2 | attention window warmup | #13 | window_warmup | flex_kernel | flex_kernel | A1a |
| A3 | value embeddings (VE rung 1) | #14 | value_embeds | - | - | - |
| A4 | U-net value embeddings (VE rung 2) | #15 | ve_unet, ve_lambdas2 | value_embeds | value_embeds | A3 |
| A4b | #15 assorted code optimizations | #15 | code_opts_15, muon_ema_momentum, blockmask_manual | flex_kernel, long_context | flex_kernel, long_context | A1a, A1b |
| A5 | split VE + block sliding window + separate block mask (VE rung 3) | #16 | ve_split, block_window, blockmask_separate | ve_unet, flex_kernel, long_context | ve_unet, value_embeds, flex_kernel, long_context | A1a, A1b, A3, A4 |
| A5b | #16 MFU tweaks | #16 | code_opts_16 | - | - | - |
| A6 | sparsified VE + improved rotary (VE rung 4) | #17 | ve_sparsify, rope_half_truncate | ve_split | ve_split, ve_unet, value_embeds | A3, A4, A5 |
| A6b | dropped attention layer (capacity change) | #17 | drop_attn_layer | - | - | - |
| A7 | logit softcap 30 -> 15 | #18 | config:logit_softcap | - | - | - |
| A8 | Muon -> AdamW (positive control) | #3 | optimizer | - | - | - |

Reading the table. "Prerequisite arms" are mechanical: a configuration with the arm on must have them on. For
add-one-in this makes A1b, A2, A4, A4b, A5 and A6 cumulative (the matrix adds the prerequisites and records it).
For leave-one-out it means that removing a prerequisite arm from r18 while its dependants stay on is undefined —
not only for the VE ladder (A3 ← A4 ← A5 ← A6) but also for the FlexAttention kernel (A1a ← A1b, A2, A4b, A5) and
the 64K layout (A1b ← A4b, A5). This is reported as an open decision (`experiments/matrix.py` lists the options
and does not pick one).
