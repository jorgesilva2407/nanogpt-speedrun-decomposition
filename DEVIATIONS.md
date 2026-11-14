# DEVIATIONS.md — what was executed, and where it differs from the pre-registered plan

Written 2026-11-09, after the H100 sweep stopped; sections 1, 2.2, 3.1, 3.3, 3.4, 4 and 7 updated 2026-11-14 after the lab A100 finished. This is a record, not a decision: it has no D-number and changes nothing
in `OPEN_DECISIONS.md` or `prereg/predictions.yaml`. "Plan" means `OPEN_DECISIONS.md` D1–D13, `experiments/study.yaml` and
the README. Numbers are measured unless marked *estimate*.

## 1. What was executed

All runs: H100 SXM 80GB (D9), host class `vast-h100` (D10), commit `3706f29` (`study-freeze`), clean tree, except as noted in 5.

| | planned | executed |
|---|---|---|
| learning-curve runs (D2: 21 configurations × 3 seeds) | 63 | **28** on the H100, **+ 24** on the lab A100 (below, section 7) |
| throughput runs (D11: 21 jobs, one session) | 21 | **21** |
| spend (Vast charge records) | — | $78.59 |

Seeds executed per configuration (plan: 3 each):

| tier | configuration | seeds |
|---|---|---|
| fixed points | `fp_r11`, `fp_r18` | 3, 3 |
| P1, both directions | `aoi_` and `loo_` of A2, A4b, A6, A6b | 2, 2 each |
| P1 | `aoi_A7`, `loo_A7` | 1, 2 |
| P1, add-one-in only | `aoi_A1a`, `aoi_A3` | 1 each |
| P1, control | `ctrl_A8_r18`, `ctrl_A8_r11` | 1, **0** |
| P2 | `aoi_A1b`, `aoi_A4`, `aoi_A5` | **0** each |
| P3 | `aoi_A5b`, `loo_A5b` | **0** each |

Lab A100, a separate dataset (`speedrun-decomp/speedrun-results-a100`, tag `a100-v1`, host label `lab-a100`, 24 runs; only the transfer anchors
`fp_r11`, `fp_r18`, `aoi_A6` and `aoi_A3` repeat H100 configurations, to test whether Δ transfers, and are never pooled with the H100 runs): `fp_r11` ×3, `fp_r18` ×2, `aoi_A6` ×2, `aoi_A3` ×1 (the transfer check); `aoi_A1b`, `aoi_A4`, `aoi_A5`,
`aoi_A5b` and `loo_A5b` ×2 each (the gaps of RQ1 and RQ2); `rq3b_s0.5_*` and `rq3b_s0.75_*` of `fp_r11`, `aoi_A6`, `aoi_A3` ×1 each (RQ3b).

Fixed-point pass (D12): both passed (fp_r11: mean 3.2759 vs record 3.2753, SD 0.0042; fp_r18: 3.2802 vs 3.2791, SD 0.0015;
largest gap to the record curve in the final third 0.0017 and 0.0030, trigger 0.01). D13's ratio r = measured / estimated
fixed-point wall time came out at 0.90 (fp_r11) and 0.94 (fp_r18), both ≤ 1.0, which allows P1 + P2 subject to the budget clause.

## 2. Deviations from the plan

**2.1 Seeds: 1–2 instead of 3 (D7).** The credit available covered the fixed points, one seed of P1, and a second seed of
five configurations. Consequences: the detectable effects of the proposal's table (σ = 0.0013) do not apply. Our `fp_r11`
seed SD is 0.0042, about 3× that value, so one- and two-seed arms resolve only large effects. For one seed against the three-seed baseline the 95% half-width is about
±170 steps on r11 and ±35 on r18 (*estimate*: t interval with the SD pooled over every configuration of that preset, 59 and 13
steps, 6 and 7 degrees of freedom; two seeds give ±132 and ±27). With the baseline's three seeds alone the intervals would be
±293 and ±79, and a z interval, which ignores that the SD is estimated, would give ±142 and ±38: an earlier version of this
note quoted the optimistic ±134/±36. Arms with one seed get point estimates with no interval of their own.

**2.2 P2 and P3 learning runs not run on the H100 (D13); run on the lab A100 instead.** A1b, A4, A5 (P2) and A5b (P3) have H100
throughput but no H100 learning run. D13 allowed P2 at the measured r, but not within the credit. On the H100 alone RQ1's
decomposition is defined for 7 of 11 arms (A1a, A2, A3, A4b, A6, A6b, A7) and these four arms are throughput-only. The lab A100
(section 7) then measured them, two seeds each, against its own baseline: their Δ steps are marked `[A100]` in the table, and
their wall-clock split combines the A100 Δ steps with the H100 throughput, shown only because the transfer check passed. The
middle rungs (A4, A5) of the value-embedding ladder are therefore available; the ladder's rungs are bundles, so read them with 3.1.
Still not run: the context-only effect inside `aoi_A4b` (`A1b.ctx`, 3.1).

**2.3 `ctrl_A8_r11` not run.** This is D13's pre-registered first cut, so it follows the plan. The positive control is
`ctrl_A8_r18` only (one seed).

**2.4 Where the money went.** After round 0 (one seed of every P1 configuration except `ctrl_A8_r11`) the remaining budget
bought seed 1 of the four both-direction pairs (A2, A4b, A6, A6b) and of `loo_A7`, not the P2/P3 arms. D7 says "Money left
after P1 goes to P2 (more arms) before more seeds". P1 was not complete (D13 means 3 seeds per configuration), so P2 was not
yet next under D13, but the choice trades breadth for precision and is recorded as such. Seed 0 of A1b, A4, A5 and A5b would
have cost about $14, against about $16 spent on those second seeds (*estimates* from measured run times and $2.343/h).

**2.5 Schedule.** D6's dates (results freeze Sun 2026-11-08, submit Nov 13) were superseded; the dates used are results
freeze Tue 2026-11-10 and report due Sun 2026-11-15. D6's text has not been edited.

## 3. Interpretation limits (not deviations, but needed to read the results)

**3.1 The add-one-in configurations carry prerequisite flags.** `aoi_A2` also switches on `flex_kernel`; `aoi_A4b` also
switches on `flex_kernel` and `long_context` (without `doc_masking`); `aoi_A6` switches on the whole value-embedding ladder
(`value_embeds`, `ve_unet`, `ve_split`, `ve_sparsify`, `rope_half_truncate`). The matching leave-one-out removes only the arm's own
flags. So the A4b and A6 add-one-in numbers measure a bundle, and their gap to leave-one-out mixes interaction with the
bundle. Not run: A1b's context-only sub-arm (`A1b.ctx`), which would isolate the long-context part of `aoi_A4b`.
The cleanest order-dependence pairs are A2, A6b, A7 (A7 is a hyperparameter change only) and, on the A100, A5b (`code_opts_16` alone
in both directions). The A100 add-one-in arms are bundles too: `aoi_A1b` = `flex_kernel` + `long_context` + `doc_masking`;
`aoi_A4` = `value_embeds` + `ve_unet` + `ve_lambdas2`; `aoi_A5` = the A4 bundle + `ve_split` + `flex_kernel` + `long_context` +
`block_window` + `blockmask_separate` (no `doc_masking`), which is why A5 is not a step up from A6 on the same axis (A6 has the
ladder but no flex kernel or long context).

**3.2 Runs that never reach 3.28.** 12 of the 28 runs end above 3.28: 2 of 3 `fp_r18` seeds, 1 of 3 `fp_r11`, 7 of 10
leave-one-out runs (`loo_A2` 2/2, `loo_A6` 2/2, `loo_A7` 2/2, `loo_A4b` 1/2, `loo_A6b` 0/2), `aoi_A6b` 1/2 and
`ctrl_A8_r18` 1/1. `analysis/tables.py` counts them as budget step + (loss at budget − 3.28) / local slope, the `persistence.py`
fallback (commit 550ba45; before it, such runs were silently dropped). Each cell reports how many seeds are extrapolated. The
**final loss at the budget needs no extrapolation** and should be quoted next to every step count.

**3.3 A8 (positive control).** AdamW ends at 3.4624 against 3.2802 for Muon (fp_r18, mean of 3 seeds) and Muon reaches AdamW's final
loss at step 897 (interpolated; 896, 895 and 899 for the three seeds), so AdamW needs at least 1,390 / 897 = 1.55× the steps. AdamW
runs at the borrowed lr 0.0018 with no LR sweep (D7), so this is an upper bound on Muon's advantage. A step count to 3.28 would be
an extrapolation far past the data, so A8 is reported as the loss gap at the budget and the step-ratio lower bound; the table
row is filled in. (An earlier version of this note said step 950 and 1.46×; that was the first validation point at or below the loss
(the grid, 950 for all three seeds), not the interpolated crossing.)

**3.4 Throughput.** Each throughput run is one run (seed 0) of 300 steps, 290 after warm-up, so a throughput run has no
replicate of its own. Two other sources bound its noise on the H100. (a) Every learning run of the main sweep also logged the
time of every step, so most configurations have 2–3 independent runs on one machine: the seed-to-seed SD of the mean step time
is 0.16% (median over configurations; maximum 1.1%). (b) The main sweep and the throughput sweep ran on two different H100
machines, and the dense configurations (no window warm-up) differ by 0.5–1.9% between them, so step-time differences below
about 2% between arms measured on different machines are not interpreted. A throughput run covers only the first 300 steps,
so for configurations with window warm-up it understates the full-run step time: the r18-based configurations are 7–9% slower
over their whole training run (`fp_r18` 1,131 ms against 1,040 ms; the table output of `analysis/tables.py` now lists the whole-run
step time of every H100 learning configuration next to its 300-step throughput run, separately from the throughput columns). The A100
throughput sweep has no replicates and no second machine and is not reported. Wall-clock split fractions misbehave when steps and step time move in opposite directions
(A6b shows −74% / 174%); report signed amounts there.

## 4. Not run

RQ3b on the H100 and RQ3b's leave-one-out side everywhere. `rq3b.resolved_arms` was filled (A2, A3, A4b, A6, A7) after the H100 sweep, but
only the add-one-in side of A3 and A6 was run, on the A100, one seed each, at 0.5× and 0.75× (section 7); A2, A4b and A7 were not
rerun. RQ3a is read from the logged curves. Resume-from-checkpoint was assessed and not built.

## 5. Infrastructure that differs from the plan

* **One sequential worker on one on-demand instance** ran the main sweep, not many parallel workers. An interruptible attempt
  was outbid twice and finished no run ($1.13). The tooling still supports N workers (`make_jobs.py --workers`).
* **Throughput ran on a rented H100** (D9, D11), not the lab A100 that the README and `study.yaml` comments still name.
* **`fp_r18` seed 0** ran on an earlier machine and image (code `8d8da55`). Later images only added the cuDNN SDPA
  meta-registration backport and the p1 job order; its result lies within the spread of its other two seeds.
* **Torch:** stable 2.6.0+cu126 with two backports, FlexAttention `d2ec7f0756` and cuDNN SDPA `ec93aa7f84` (`CHANGES/infra.md`).
* **p1 job order:** priority order within each seed round, `ctrl_A8_r11` last (`CHANGES/infra.md`); same job set.
* **Job list edited while the worker ran.** Round 0 was cut to 13 jobs and extended by nine seed-1 jobs by editing
  `jobs.txt` in place, which this worker reads one line at a time. The original 42-line list and the edits are in the results
  repo under `_logs/59638827/`.
* **aoi_A4b and aoi_A5 compile slowly** (about 61 min for `aoi_A4b` seed 0 on the main host; later seeds hit the compile cache
  in about a minute). This costs wall time only.

## 6. Documents still saying the old thing

D6's dates; the README and `study.yaml` comments naming the lab A100 as the throughput host; `study.yaml` seeds (3 default).
Left as written, to be revised deliberately.

## 7. Lab A100

The lab A100 (80GB PCIe, host label `lab-a100`) fills only the gaps the H100 sweep left in RQ1 and RQ2 (learning runs for A1b, A4,
A5, A5b and the A5b pair) and, last and smallest, the add-one-in side of RQ3b, under `lab_a100/PROTOCOL.md` (written before any A100
learning run; driver `lab_a100/run.sh`; results in the separate HF dataset `speedrun-results-a100`; configurations from the tag
`a100-v1`). Its Δ are used next to the H100's, with the H100 throughput, only if the `anchor` phase of the protocol passes. That
deviates from D9/D10 and is declared as a limitation in the report and the presentation (disclaimer text in the protocol); no H100
experiment is repeated and the H100 measurements are not strengthened.

**Transfer check (PROTOCOL.md section 3, evaluated by `analysis/transfer.py`; numbers in `reports/table.md`).** It passed: the `fp_r11`
final-loss offset A100 − H100 is +0.0002 ± 0.0093 (3 against 3 seeds; limit ±0.010), the largest gap between the mean curves at the
final-third checkpoints 0.0012 (limit 0.01), the A100 seed SD 0.0040 (H100 0.0042; range [0.0002, 0.0057]); the bridge `aoi_A6` has Δ
−218 steps on the A100 against −219 on the H100 (difference +1 ± 128) and `aoi_A3` −246 against −247. Seed by seed the A100 and
H100 final losses of `fp_r11` differ by +0.0009, +0.0000, −0.0003, and the `fp_r18` offset (FlexAttention path, 2 against 3 seeds) is
−0.0005 ± 0.0069. The interval of the offset is wide, ±0.0093 of the ±0.010 allowed: the check shows no offset beyond about a
hundredth of a loss unit, not that there is none. The A100 Δ are therefore combined with the H100 throughput and reported next to
the H100 arms, flagged `[A100]`.

What the A100 arms resolve (95% half-widths in the table, two seeds each, Welch t): A1b ±116 steps, A4 ±145, A5 ±129, A5b add-one-in
±240 (one of its two seeds never crosses 3.28 and one of the baseline's never does, which widens it), `loo_A5b` ±94. Their Δ steps are
−489 (A1b), −221 (A4), −565 (A5), +17 (A5b add-one-in) and −5 (`loo_A5b`): A1b and A5 are resolved, A4 only just, and the A5b pair
is within the noise of zero in both directions (it was registered as throughput-dominant). The RQ3b points are single runs with no
interval, a difference of two single runs having an SD of about 0.006 loss (fp_r11 seed SD 0.0042).

A first sweep of the 21 throughput configurations on the A100 (13.3 h, one run each) was a compile-and-behaviour check, not a result
of the study. All 21 compile and run, peak memory matches the H100's within about 48 MiB, dense configurations run at 2.7–2.9× the
H100 step time, and the attention-window, long-context and block-mask configurations at 3.1–16.8× (`loo_A2` 12.2 s per step,
`aoi_A4b` 21.0 s; `aoi_A4b` and `aoi_A5` also warmed up for 68 and 111 min). The step-time effect of A1a, A1b, A2, A4b, A5 and A6b
differs in size, and for A6b in sign, from the H100's. A100 throughput is not reported.
