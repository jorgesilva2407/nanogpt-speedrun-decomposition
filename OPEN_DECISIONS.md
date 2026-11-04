# Decisions

All decisions below were **taken on 2026-11-04** (D8 on 2026-10-18) and are implemented in `experiments/study.yaml`,
`launch/make_jobs.py` and the regenerated matrix. The section "Background" further down is the original write-up of
each open question (2026-10-18), kept unchanged so the options and their trade-offs stay on record.

## Decided 2026-11-04

| | decision | why (short) | where |
|---|---|---|---|
| D1 | **O4**: no leave-one-out for A1a, A1b, A3, A4, A5; add-one-in only. RQ2 covers A2, A4b, A5b, A6, A6b, A7. | Free; the LOO is undefined for these arms. First extra purchase if money remains after P2: O1 for the VE ladder only (cumulative LOO of A3/A4/A5, 3 configurations, ~$5.5). Note: O1 does not inflate variance as the background text below says. The per-rung estimate is a difference of two configuration means against the same fp_r18, so it is 2σ²/n, like any RQ1 difference. | `study.yaml: loo_prerequisite_policy: aoi_only` |
| D2 | **21 configurations, 63 runs** (2 fp, 11 aoi, 6 loo, 2 ctrl). The proposal's "14" is superseded. | Unbundled sub-arms are part of the contribution; the report says so. | `experiments/matrix.json` |
| D3 | **v8 cadence**: full validation set every 125 steps, every 25 in the final third. | The subset20 path has never run and there is no spike to exercise it; the fixed-point check compares against full-set record logs. | `study.yaml: validation.mode: v8` |
| D4 | **Keep** window warmup as is (requires only `flex_kernel`). | On r11's 1,024-token rows the window stops restricting at ≈0.56 N, so A2's add-one-in measures a truncated mechanism. A2's LOO−AOI gap is reported in its own row as **structurally inflated** and does not count toward H2 on its own. | report |
| D5 | **Signed off**: softcap 30→15 registered as "no sample-efficiency effect". | A7 is softcap alone (the #18 rewrite's other changes are separate switches), so the test is clean either way; editing a pre-registration the day before the first run invites doubt. | `prereg/predictions.yaml` (unchanged, tag `prereg-v1`) |
| D6 | **All-cloud schedule** (revised 2026-11-04): freeze Wed 2026-11-04 evening; fixed points and throughput overnight; go/no-go Thu 2026-11-05; P1 Thu–Fri; P2 Fri–Sat; results freeze Sun 2026-11-08; writing Nov 9–11; read-through Nov 12; submit Nov 13. | The lab A100 is no longer available; the spike is dropped (fixed points are the first runtime validation; single-GPU runs everywhere make S3 moot). | `launch/cloud_worker.sh` |
| D7 | **3 seeds everywhere; no AdamW LR sweep.** | RQ2 MDE ±65 steps at σ=0.0013: H2 reported as underpowered below that, not as unsupported. Money left after P1 goes to P2 (more arms) before more seeds. Without the sweep, A8 compares AdamW at a borrowed LR (0.0018, record #2's) with co-tuned Muon: reported as an upper bound on Muon's advantage. | `study.yaml: rq2_arms: []`, `adamw_lr_sweep.bases: []` |
| D9 | **One GPU model for every run: H100 SXM 80GB** (rule: H100 if its price < 2.3× the A100's). | Avoids the A100-specific FlexAttention backward-tile risk under torch 2.6.0; 2.3× faster wall-clock; leaderboard hardware. Mixing GPU models would change autotuned kernels. | `study.yaml: throughput.host: vast-h100` |
| D10 | **Many rented machines of that model, running the pinned image, are one host class** (`vast-h100`), replacing proposal Rule 2's one-machine rule. | Steps-to-target depends on the token stream, the stack and the kernels, all pinned. Cross-machine nondeterminism lands in σ (seeds run on different machines), so it widens the CIs instead of biasing a Δ. Declared in the report. | `HOST_LABEL=vast-h100` |
| D11 | **Throughput**: all 21 jobs back to back on one rented instance in one session; only Δ tokens/s reported. | Replaces Rule 3's idle lab node; ratios within one session are comparable even if absolute tokens/s carry host noise. | `TIER=throughput` |
| D12 | **Fixed-point pass**: \|mean of 3 seeds − record final loss\| < 2.3σ̂ (σ̂ = SD of the 3 seeds), and the curve tracks the record's logged curve through the final third. Records: #11 3.2753 at step 3000, #18 3.2785 at step 1390. | A 3-seed mean against one logged run differs with SD σ√(1+1/3) ≈ 1.15σ, so the 95% band is ≈ ±2.3σ (±0.003 at σ=0.0013, ±0.012 at σ=0.005). On failure: debug until Fri 2026-11-06 at the latest, then sweep anyway and report the fixed points as unverified. | go/no-go |
| D13 | **Tiers and cut line, fixed in advance.** P1 = fixed points + A2, A4b, A6, A6b, A7 (both directions) + A1a, A3 (add-one-in; they carry primary-source predictions, which H1 needs) + A8 control. P2 = A1b, A4, A5 (add-one-in). P3 = A5b (throughput still measured). With r = measured / estimated fixed-point wall time (estimates: fp_r11 ≈ 4,250 s, fp_r18 ≈ 2,200 s on H100): r ≤ 1.0 → P1 + P2; 1.0 < r ≤ 1.3 → P1; r > 1.3 → P1 without `ctrl_A8_r11`. Always: remaining P1 cost ≈ 39 H100-h × r × $/h must fit the credit left with 15% spare for re-runs. | Deciding after seeing the spend invites sunk-cost choices. | `launch/make_jobs.py: TIERS` |

## Background (as written 2026-10-18; options and trade-offs)

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

