| arm | registered prediction (provenance) | Δ steps LOO (r18 → −arm) | Δ steps AOI (r11 → +arm) | Δ tokens/s LOO | Δ tokens/s AOI | wall-clock split LOO (sample-eff / throughput / cross) | wall-clock split AOI (sample-eff / throughput / cross) | LOO−AOI gap (upper bound on interaction) | RQ3a Δ steps @3.40/3.35/3.30 |
|---|---|---|---|---|---|---|---|---|---|
| A1a FlexAttention kernel | Throughput (Compiler paper; no modelling change) | undefined (D1) | -34 [extrap. 1/3 base, 0/1 arm] | n/a | -18960 | n/a | -42% / 142% / -1.6% | n/a | (not in rq3b.resolved_arms) |
| A1b 64K-ctx sliding window + document masking | Sample efficiency (Ours — unsupported) | undefined (D1) | -489 ± 116 [extrap. 1/3 base, 0/2 arm] [A100] | n/a | -52944 | n/a | 271% / -171% / 30.9% | n/a | -569 ± 312 / -383 ± 108 / -417 ± 64 [A100, resolved in RQ1; not in rq3b.resolved_arms] |
| A2 attention window warmup | Sample efficiency (Ours — unsupported) | +96 ± 41 [extrap. 2/3 base, 2/2 arm] | -9 ± 140 [extrap. 1/3 base, 0/2 arm] | -36683 | +5210 | 47% / 53% / 3.6% | 22% / 78% / -0.2% | +88 ± 146 | +0 ± 78 / -1 ± 64 / -2 ± 73 |
| A3 value embeddings (VE rung 1) | Sample efficiency, ~20% fewer tokens (Paper's headline, transferred naively) | undefined (D1) | -247 [extrap. 1/3 base, 0/1 arm] | n/a | -18360 | n/a | 177% / -77% / 6.8% | n/a | -190 / -165 / -197 |
| A4 U-net value embeddings (VE rung 2) | Mixed (Ours — unsupported) | undefined (D1) | -221 ± 145 [extrap. 1/3 base, 0/2 arm] [A100] | n/a | -9317 | n/a | 133% / -33% / 2.5% | n/a | -178 ± 45 / -154 ± 48 / -174 ± 53 [A100, resolved in RQ1; not in rq3b.resolved_arms] |
| A4b #15 assorted code optimizations | Mixed (Ours — unsupported) | -7 ± 66 [extrap. 2/3 base, 1/2 arm] | -473 ± 150 [extrap. 1/3 base, 0/2 arm] | -3201 | -71028 | -325% / 425% / -2.1% | 942% / -842% / 146.9% | -466 ± 164 | -565 ± 51 / -363 ± 49 / -403 ± 73 |
| A5 split VE + block sliding window + separate block mask (VE rung 3) | Throughput-dominant (Ours — unsupported) | undefined (D1) | -565 ± 129 [extrap. 1/3 base, 0/2 arm] [A100] | n/a | -60596 | n/a | 264% / -164% / 34.7% | n/a | -685 ± 520 / -538 ± 583 / -497 ± 120 [A100, resolved in RQ1; not in rq3b.resolved_arms] |
| A5b #16 MFU tweaks | Throughput-dominant (Ours — unsupported) | -5 ± 94 [extrap. 1/2 base, 1/2 arm] [A100] | +17 ± 240 [extrap. 1/3 base, 1/2 arm] [A100] | -2175 | +6003 | -416% / 516% / -1.8% | -86% / 186% / 1.0% | -12 ± 258 [A100] | (not in rq3b.resolved_arms) |
| A6 sparsified VE + improved rotary (VE rung 4) | Sample efficiency (Ours for the tweak — unsupported) | +60 ± 41 [extrap. 2/3 base, 2/2 arm] | -219 ± 116 [extrap. 1/3 base, 0/2 arm] | -4309 | +2134 | 83% / 17% / 0.7% | 95% / 5% / -0.4% | -159 ± 123 | -164 ± 103 / -146 ± 57 / -171 ± 63 |
| A6b dropped attention layer (capacity change) | Throughput, with a sample-efficiency cost (Ours — unsupported) | -28 ± 61 [extrap. 2/3 base, 0/2 arm] | +80 ± 1070 [extrap. 1/3 base, 1/2 arm] | -14811 | +24965 | -226% / 326% / -6.7% | -118% / 218% / 5.9% | -52 ± 1072 | (not in rq3b.resolved_arms) |
| A7 logit softcap 30 -> 15 | No sample-efficiency effect (The source claims stability, not efficiency) | +79 ± 32 [extrap. 2/3 base, 2/2 arm] | -82 [extrap. 1/3 base, 0/1 arm] | +9371 | -1191 | 150% / -50% / -2.8% | 109% / -9% / 0.3% | -2 | -60 / -58 / -56 |
| A8 Muon -> AdamW (positive control) | Sample efficiency, large (Positive control — a miss here indicts the instrument) | AdamW +0.1822 loss at the budget; needs ≥ 1.55× the steps (one run, lower bound; DEVIATIONS.md 3.3) | not run (ctrl_A8_r11, D13) | n/a | n/a | n/a | n/a | n/a | (not in rq3b.resolved_arms) |

Δ steps: interpolated crossing of 3.28 on full-validation points; intervals are 95% Welch t intervals across seeds. A run that never crosses 3.28 counts as budget step + step-equivalent shortfall (loss at budget − 3.28) / local slope, the persistence.py fallback; [extrap. k/n] says how many seeds of the baseline and the arm that is. "not run" = the configuration is in the matrix but has no completed learning run; such arms are throughput-only, and their Δ tokens/s columns still apply. [A100] = measured on the lab A100 (lab_a100/PROTOCOL.md) against the A100's own baseline (fp_r11 for add-one-in, fp_r18 for leave-one-out), never against the H100 one; its tokens/s are the H100's, so its wall-clock split needs the transfer check below to pass. Δ tokens/s and the splits use the H100 throughput sweep (300 steps each); see the timing table for whole-run step times. The add-one-in rows of A2, A4b and A6 carry prerequisite flags (DEVIATIONS.md 3.1). Split fractions blow up when the total time change is near zero (A4b, A5b, A6b); read the signed Δ steps and Δ tokens/s there. No p-values are reported.

## Transfer check: lab A100 against the H100

| check (PROTOCOL.md section 3) | measured | rule | |
|---|---|---|---|
| 1 baseline offset, fp_r11 final loss, A100 − H100 (3 vs 3 seeds) | +0.0002 ± 0.0093 | interval inside ±0.010 | pass |
| 2 curve, largest gap of the mean fp_r11 curve over the final-third 125-step checkpoints | 0.0012 (step 2125) | ≤ 0.01 | pass |
| 3 seeding, A100 fp_r11 SD of the final loss (H100: 0.0042) | 0.0040 | in [0.0002, 0.0057] | pass |
| 4 bridge aoi_A6: Δsteps A100 -218 (n=2) vs H100 -219 (n=2) | Δ(A100) − Δ(H100) = +1 ± 128 steps | same sign, interval contains 0 | pass |
| 4 bridge aoi_A3: Δsteps A100 -246 (n=1) vs H100 -247 (n=1) | one seed per GPU, no interval | same sign | pass |
| **result** | | all of the above | **pass** |

Not gates: the offset paired by seed index (A100 − H100 final loss, seeds 0, 1, 2) is +0.0009, +0.0000, -0.0003 (mean +0.0002). The fp_r18 offset (2 A100 vs 3 H100 seeds, the FlexAttention path) is -0.0005 ± 0.0069.


## RQ3b: budget sweep (add-one-in side, A100)

| budget | fp_r11 final loss | aoi_A3 final loss | Δ aoi_A3 | aoi_A6 final loss | Δ aoi_A6 |
|---|---|---|---|---|---|
| 0.5× (1,500 steps) | 3.3952 (n=1) | 3.3746 (n=1) | -0.0206 | 3.3777 (n=1) | -0.0175 |
| 0.75× (2,250 steps) | 3.3182 (n=1) | 3.2926 (n=1) | -0.0256 | 3.2970 (n=1) | -0.0212 |
| 1.0× (3,000 steps), seed 0 | 3.2743 (n=1) | 3.2445 (n=1) | -0.0297 | 3.2506 (n=1) | -0.0237 |
| 1.0× (3,000 steps), all seeds | 3.2761 (n=3) | 3.2445 (n=1) | -0.0316 | 3.2492 (n=2) | -0.0270 |


Δ = final validation loss of the arm minus fp_r11 at the same budget (negative = the arm is better); every point is one run of seed 0, so Δ has no interval of its own. With the fp_r11 seed SD of 0.0042 the difference of two single runs has SD about 0.006 (not paired); pairing by seed index may shrink it, as the A100−H100 pairs of the transfer check suggest, and it is not assumed here. The budget is a multiple of the 3,000-step add-one-in budget with the learning-rate schedule rescaled (absolute-step hyper-parameters listed in study.yaml are rescaled with it). The leave-one-out side of RQ3b, aoi_A4b and the arms A2 and A7 were not run (lab_a100/PROTOCOL.md section 2).


aoi_A3: Δ loss -0.0206 (0.5×), -0.0256 (0.75×), -0.0297 (1.0×).

aoi_A6: Δ loss -0.0175 (0.5×), -0.0212 (0.75×), -0.0237 (1.0×).

## A8 positive control (AdamW instead of Muon)

AdamW ends at 3.4624 after 1390 steps (1 run) against 3.2802 for Muon (fp_r18, 3 seeds): a loss gap of +0.1822. Muon reaches AdamW's final loss at step 897, so AdamW needs at least 1.55× the steps (lower bound; one run, borrowed learning rate 0.0018, no sweep: an upper bound on Muon's advantage).


## Whole-run step time (H100 learning runs) next to the 300-step throughput runs

| configuration | learning runs (n) | whole-run mean step time (ms) | seed SD (%) | throughput run, first 300 steps (ms) | whole run vs 300 steps |
|---|---|---|---|---|---|
| aoi_A1a | 1 | 1121 | n/a | 1111 | +0.9% |
| aoi_A2 | 2 | 1109 | 0.00 | 1057 | +4.9% |
| aoi_A3 | 1 | 1118 | n/a | 1110 | +0.7% |
| aoi_A4b | 2 | 1259 | 0.11 | 1249 | +0.8% |
| aoi_A6 | 2 | 1076 | 0.18 | 1064 | +1.2% |
| aoi_A6b | 2 | 1032 | 0.03 | 1017 | +1.5% |
| aoi_A7 | 1 | 1091 | n/a | 1071 | +1.9% |
| fp_r11 | 3 | 1074 | 0.14 | 1068 | +0.5% |
| fp_r18 | 3 | 1131 | 1.06 | 1040 | +8.7% |
| loo_A2 | 2 | 1140 | 0.08 | 1122 | +1.7% |
| loo_A4b | 2 | 1123 | 0.16 | 1047 | +7.3% |
| loo_A6 | 2 | 1136 | 0.16 | 1049 | +8.3% |
| loo_A6b | 2 | 1168 | 0.33 | 1071 | +9.0% |
| loo_A7 | 2 | 1104 | 0.30 | 1021 | +8.2% |


The whole-run mean covers every logged step except compile warm-up; the throughput run covers the first 300 steps (290 after warm-up) of one run. They differ most where the attention window grows during training (window warm-up). The A100 timings are not reported (DEVIATIONS.md 7).
