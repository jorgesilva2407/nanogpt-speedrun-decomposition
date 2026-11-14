# lab_a100/PROTOCOL.md — using the lab A100 to fill the gaps of the H100 sweep

Written 2026-11-10, **before any A100 learning run**. It fixes the pass rule and the pooling rule so neither can be tuned after the
data are seen. It is a record, not a decision: no D-number. A later change is made in a new commit, with the reason.

## 1. Purpose and constraint

The H100 sweep (28 learning runs) leaves gaps in the research questions. The A100 fills **only those gaps**; no H100 experiment is
repeated and the H100 measurements are not strengthened. Priority: RQ1 and RQ2 first, RQ3b last and small.

| question | answerable from the H100 data? | gap filled here |
|---|---|---|
| RQ1, decomposition (11 arms) | 7 of 11 arms | learning runs for A1b, A4, A5, A5b (throughput exists for all 21 configurations) |
| RQ2, order dependence (6 arms with a leave-one-out) | 5 of 6 (A2, A4b, A6, A6b, A7) | the A5b pair: `aoi_A5b` and `loo_A5b` |
| RQ3a, thresholds from logged curves | yes | nothing (`rq3b.resolved_arms` is filled) |
| RQ3b, reruns at 0.5× and 0.75× of the budget | no data | add-one-in side for A3 and A6 |
| H1, H2 | yes (A7 and A3 fail their sources' predictions; A2 differs between directions) | nothing |

Host: lab machine `tuna`, one **A100 80GB PCIe** (`GPU-9c95e339-…`), host label `lab-a100`, the same image
(`…modded-nanogpt-flagged:9ec04dbd…`) and the same trainer; configurations come from the tag `a100-v1`, which adds only the RQ3b
configurations to `study-freeze`. Results go to the separate HF dataset `speedrun-decomp/speedrun-results-a100`, never to the H100
one. The H100 tooling (`launch/cloud_worker.sh`, `launch/make_jobs.py`, the Vast templates) is not used.

## 2. Phases (`lab_a100/run.sh`, one invocation: `all`)

| phase | jobs (configuration: seeds) | serves | A100 time (estimate) |
|---|---|---|---|
| `anchor` | `fp_r11`: 0 1 2; `aoi_A6`: 0 1; `aoi_A3`: 0 | baseline for the new arms, transfer bridge (section 3), RQ3b's 1.0× point | ~15 h |
| `gaps` | `aoi_A1b`, `aoi_A4`, `aoi_A5`, `aoi_A5b`: 0 | RQ1 for the missing arms; half of RQ2's A5b pair | ~14 h |
| `rq2` | `fp_r18`: 0 1; `loo_A5b`: 0 | the other half of RQ2's A5b pair | ~11–14 h |
| `seeds2` | `aoi_A1b`, `aoi_A4`, `aoi_A5`, `aoi_A5b`, `loo_A5b`: 1 | a ~21% tighter interval for the new arms | ~16 h |
| `rq3b` | `rq3b_s0.5_` and `rq3b_s0.75_` of `fp_r11`, `aoi_A6`, `aoi_A3`: 0 | RQ3b, add-one-in side only | ~10 h |

About 66–69 h in total (estimate); a phase that is not finished by the report deadline is left out of the report and shown in the
presentation as work in progress. Not run: the leave-one-out side of RQ3b, `aoi_A4b` at scaled budgets, extra seeds for arms the H100
sweep already covers, `ctrl_A8_r11`.

## 3. Pass rule (evaluated on `anchor`; it never changes what runs, only how results are used)

The decomposition of a new arm uses its Δsteps from the A100 and the throughput from the H100 sweep, so the Δ has to transfer
between the two GPUs. Tested on `fp_r11` and one bridge arm, on full-validation points with the fallback of `analysis/tables.py` for runs
that never cross 3.28:

1. **Baseline offset.** Welch 95% interval of (mean over 3 A100 seeds − mean over 3 H100 seeds) of the `fp_r11` final validation loss
   lies inside **±0.010** (with 3 against 3 seeds it is about ±0.0095 if the A100 SD equals the H100's 0.0042, estimate: it can pass
   and cannot show smaller offsets).
2. **Curve.** The A100 mean `fp_r11` curve is within 0.01 of the H100 one at every 125-step checkpoint of the final third (D12's trigger).
3. **Seeding.** The A100 `fp_r11` SD is between 0.0002 and 0.0057 (D12).
4. **Bridge.** For `aoi_A6`, Δ(A100) has the sign of Δ(H100) (−219 steps) and the Welch 95% interval of Δ(A100) − Δ(H100) contains 0;
   `aoi_A3` (one seed on each GPU, no interval) must have the same sign.

Secondary, reported and not a gate: the same comparison paired by seed index (seed k fixes the initialisation and the data order on
both GPUs); the `fp_r18` offset from the two A100 seeds of `rq2` (the FlexAttention path is where the A100 differs most from the H100,
but the A5b pair is within-host, so nothing depends on it).

**If it passes**, the new arms' Δ (against the A100 baseline) are reported with the H100 arms and combined with the H100 throughput.
**If it fails**, they are reported next to the H100 table, flagged "A100 steps", and the decomposition for them is marked unsupported.
Either way the disclaimer of section 6 applies. The A5b pair compares its two directions on the A100 only.

## 4. What one seed lets us claim (estimate)

95% half-width of Δsteps for an arm against a three-seed baseline, t interval with the SD pooled over all configurations of the preset
(59 steps on r11 with 6 degrees of freedom, 13 on r18 with 7): **±167 steps (r11) and ±35 (r18) for one arm seed, ±132 and ±27 for two**.
So an effect below those is "not resolved", not "absent". For arms registered as throughput-dominant (A5, A5b) "no sample-efficiency
effect beyond ±167 steps" is a legitimate claim; the decomposition then reads as throughput.

## 5. Starting, stopping, resuming

One `docker run` with the image's own empty entrypoint overridden by `bash`; the script is fetched from the tag, so nothing is baked
into the image and nothing of the H100 start-up is reused. Finished runs are uploaded one by one; stopping the container loses only the
run in progress (it restarts from step 0, there is no resume), and restarting skips everything already finished.

The command (credentials come from the environment, never from the repository):

```bash
IMG=ghcr.io/jorgesilva2407/modded-nanogpt-flagged:9ec04dbd69e72372fba7fbd797030e46363b8662
COMMON="--shm-size=16g --gpus device=<GPU UUID> -v $HOME/a100-lab:/workspace -e GH_TOKEN -e HF_TOKEN \
  -e RESULTS_REPO=speedrun-decomp/speedrun-results-a100 -e STUDY_TAG=a100-v1 -e DOCKER_TAG=$IMG -e HOST_LABEL=lab-a100 --entrypoint bash"
CMD='curl -fsSL -H "Authorization: token $GH_TOKEN" -o /tmp/run.sh https://raw.githubusercontent.com/jorgesilva2407/nanogpt-speedrun-decomposition/a100-v1/lab_a100/run.sh && bash /tmp/run.sh all'
docker run --name <name> $COMMON $IMG -c "$CMD"        # restart after a stop: docker start -a <name>
```

## 6. Disclaimer for the report and the presentation

> Most experiments ran on rented NVIDIA H100 SXM 80GB GPUs (one host class, one image). The configurations that the budget did not
> cover were run on one NVIDIA A100 80GB PCIe in our lab, with the same trainer and image. We checked on a baseline and on a bridge arm
> that the quantity we transfer, the effect of an arm in steps, is preserved between the two GPU models within the stated margins, and
> every Δ uses a baseline from its own host; but this deviates from the pre-registered single-GPU-model rule (D9, D10), the margin is
> wide for the r11 baseline, and arms measured once resolve only large effects. Throughput (tokens/s) was measured on the H100 only.

## 7. Known limits

* The H100 side has 3 baseline seeds, so no amount of A100 data narrows the offset interval much below ±0.0076 (r11, 5 A100 seeds, estimate).
* A100 step times are 2.8× the H100's on dense configurations and 6–17× on the windowed and long-context ones; `aoi_A5` also compiles for
  about 2 h. This affects when results are ready, not what they mean.

## 8. Result (added 2026-11-14, after all 24 runs finished; sections 1 to 7 are unchanged)

All five phases ran from one `docker run` on one A100; systemd reloads on the host cut the container's GPU access twice and a watcher loop restarted it (the log is `~/a100-lab/worker-all.log` on the lab machine); all 24 runs finished their full schedule and are in the HF dataset. **The pass rule of section 3 passed**
(`analysis/transfer.py`, full table in `reports/table.md`): offset +0.0002 ± 0.0093 inside ±0.010; largest curve gap 0.0012; A100 SD 0.0040;
bridge `aoi_A6` −218 against −219 steps (difference +1 ± 128), `aoi_A3` −246 against −247. Secondary: seed-paired offsets +0.0009, +0.0000,
−0.0003; `fp_r18` offset −0.0005 ± 0.0069. So the new arms' Δ are reported with the H100 arms, flagged `[A100]`, with the disclaimer of section 6.
