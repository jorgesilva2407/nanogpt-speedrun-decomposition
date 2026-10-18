# Open decisions (for the group; none is resolved in code)

## D1 — Leave-one-out for an arm that another arm requires  (STOPPED HERE, per the build brief)

**What LOO means for such an arm.** "Full minus rung 3 while rung 4 requires 3" asks for a configuration the trainer
refuses: `ve_sparsify` (rung 4) builds on `ve_split` (rung 3), so r18 with rung 3 off and rung 4 on violates a
prerequisite (start-up assert `flag ve_sparsify requires flag ve_split`). The same structure holds beyond the VE
ladder, because several mechanisms are built on FlexAttention and on the 64K layout:

| arm | r18 switches that require it | dependent arms |
|---|---|---|
| A1a FlexAttention kernel | long_context, doc_masking, window_warmup, blockmask_manual, block_window | A1b, A2, A4b, A5 |
| A1b 64K layout (+doc mask) | blockmask_manual, block_window | A4b, A5 |
| A3 VE rung 1 | ve_unet | A4, A5, A6 |
| A4 VE rung 2 | ve_split | A5, A6 |
| A5 VE rung 3 | ve_sparsify | A6 |

`experiments/matrix.py` emits **no** configuration for these five LOOs (`study.yaml: loo_prerequisite_policy:
undecided`). Options (listed, not chosen):

* **O1 cumulative removal** — r18 minus the arm and every switch that transitively requires it. Always defined;
  measures the joint contribution. A per-arm number becomes a difference of two cumulative LOOs (variance of a
  difference of two estimates: RQ2's MDE grows accordingly). Adds 5 configurations.
* **O2 separable parts only** — LOO only for sub-switches nothing else requires (A1b's `doc_masking`, A4's
  `ve_lambdas2`, A5's `blockmask_separate` and `block_window`); the remainder of those arms has no LOO.
* **O3 re-implement dependants without the prerequisite** — e.g. sparsified VE on one chunked table, masks without
  FlexAttention. Not any record's code, changes the meaning of the dependent arm; not built.
* **O4 no LOO for prerequisite arms** — add-one-in only; RQ2 not estimable for A1a, A1b, A3, A4, A5.

Note: A4b's LOO is emitted, but its `blockmask_manual` part is inert in r18 (superseded by `block_window`); the LOO
then measures `code_opts_15` + `muon_ema_momentum`. Recorded in that configuration's provenance.

## D2 — How many configurations (the proposal's 14 vs 16)
The proposal's "six flags / 14 configurations" predates its own arm table (seven non-control arms → 7×2+2 = 16).
The code generates **25 configurations, 113 runs** today: fp 2, add-one-in 11, leave-one-out 6, control (A8 on both
fixed points) 2, AdamW LR sweep 4; seeds 5 for `rq2_arms` and fixed points, 3 otherwise. Differences from 16:
sub-arms the replay could separate (A1a/A1b, A4/A4b, A5/A5b, A6/A6b) are separate arms; the control and sweep are
counted; five LOOs are withheld (D1). `split_subarms: true` gives 38 configurations / 178 runs. Add-one-in runs
inherit r11's 3,000 steps (≈ 2.2× r18's 1,390), so the run count understates GPU time relative to the proposal's
"~1.5 h per run" assumption.

## D3 — Validation cadence
Default is v8 (every 125 steps, every 25 in the final third, full validation set). The alternative "fixed 20% subset
every 5 steps in the final third" (same cost, 5× resolution) was never folded into the proposal; it is supported
(`study.yaml: validation.mode: subset20`) but not selected.

## D4 — Window warmup without the 64K layout
`window_warmup` requires only `flex_kernel`; on 1,024-token rows it is inert once the window exceeds the row
(from ≈ 0.56 N). So add-one-in of A2 measures a truncated mechanism. Alternative: require `long_context`
(add-one-in becomes cumulative with A1b). Not changed.

## D5 — Pre-registered softcap prediction
`prereg/predictions.yaml` copies proposal §4 verbatim: softcap 30→15 → "No sample-efficiency effect". The group has
not signed off (project context: "needs group sign-off before the Oct 6 freeze").

## D6 — Schedule
The proposal's trainer freeze (Oct 6) and sweep window (Oct 6–20) have passed (today is 2026-10-18); nothing has
been run. The mid-report is 2026-10-23.

## D7 — RQ2 seed list and AdamW sweep bases
`rq2_arms` lists every non-control arm (5 seeds each); `adamw_lr_sweep.bases` runs the sweep on both fixed points.
Both are config fields chosen as defaults, not decisions.

## D8 — Torch build (DECIDED 2026-10-18)

Upstream #18 pins the nightly `torch==2.6.0.dev20241231+cu126`. PyTorch has pruned it from its index (on 2026-10-18
the cu126 nightly index only held `2.15.0.dev` builds), so the pinned image could not be built. **Decision (user):
use the stable `torch==2.6.0+cu126`.** Implemented in `launch/Dockerfile`; static consequences in `CHANGES/infra.md`
("compat: torch pin"). The one hardware-specific risk found: a larger FlexAttention backward tile on A100 (SPIKE S4).

