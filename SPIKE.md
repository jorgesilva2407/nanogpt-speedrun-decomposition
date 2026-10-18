# Feasibility spike (proposal §3 step 0) — commands, outputs, pass/fail criteria

Nothing below has been run. Run in order on the lab A100, inside the pinned image, from a clean checkout:

```bash
docker build -f launch/Dockerfile -t modded-nanogpt-flagged:$(git rev-parse --short HEAD) .
export DOCKER_TAG=modded-nanogpt-flagged:$(git rev-parse --short HEAD)
python data/cached_fineweb10B.py 18     # val shard + 18 x 100M train tokens (3,000-step add-one-in runs need 15.7)
python -I experiments/matrix.py         # experiments/runs/<exp_id>/config.json
```
Spike runs use `--name spike_*` so they never mix with study results.

## S1 — Peak VRAM at stock settings (65,536-token micro-batch)

| | |
|---|---|
| commands | `python launch/run_one.py --exp experiments/runs/fp_r18 --seed 0 --host lab-a100 --nproc 1 --name spike_vram_fp_r18 --extra --throughput-steps 20` and the same for `aoi_A3` (r11 + value embeddings: the 463.6M-parameter 12-chunk table, the largest model in the matrix) and `fp_r11`; then repeat `fp_r18` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` exported (name `spike_vram_fp_r18_exp`) |
| output | `results/spike_vram_<exp>/seed0/meta.json` → `peak_memory_allocated_mib`, `peak_memory_reserved_mib`; `DONE` |
| pass | all four complete (DONE) and peak reserved < 81,920 MiB. Record whether peak reserved < 49,152 MiB (decides if 48 GB cloud cards are usable; proposal §6.1 derives 28–58 GB) |
| fail | OOM in any → stop; the PR #38 fallback (32,768-token sequences) changes the micro-batch and needs a group decision (never reduce it silently: the trainer asserts 65,536) |

## S2 — Seeding works, then measure σ

| | |
|---|---|
| commands | (a) `python launch/run_one.py --exp experiments/runs/fp_r11 --seed 0 --host lab-a100 --nproc 1 --name spike_seed_a` and again with `--name spike_seed_b` (same seed, full runs); (b) seeds 1 and 2: `--seed 1 --name spike_sigma`, `--seed 2 --name spike_sigma` (seed 0 = `spike_seed_a`) |
| output | `results/spike_seed_{a,b}/seed0/metrics.jsonl`, `results/spike_sigma/seed{1,2}/metrics.jsonl` |
| pass (a) | step-0 val loss = 10.8258 in both (= ln 50304, zero-initialised head; every record log shows this) and the two same-seed final val losses differ by less than the cross-seed SD from (b). Bitwise equality is NOT expected (GPU/compiler nondeterminism, CHANGES/infra.md) |
| pass (b) | σ = SD of the three final val losses (seeds 0, 1, 2). Report it next to the proposal's prior 0.0013 (Track 3) and the #12-era 0.005. `python -I analysis/mde.py` with the measured σ gives the MDEs; if σ ≫ 0.0013, revisit `study.yaml` seeds before the sweep |
| fail | same-seed runs differ by as much as different seeds → seeding does not control the dominant variance; stop and investigate (unseeded generator, autotuning) |

## S3 — GPU-count invariance (single-GPU legitimacy)

| | |
|---|---|
| commands | on one 4-GPU host: `python launch/run_one.py --exp experiments/runs/fp_r18 --seed 0 --host lab-a100 --nproc 1 --name spike_inv_w1 --extra --throughput-steps 400` and `--nproc 4 --name spike_inv_w4` (same seed); repeat both for `fp_r11` (names `spike_inv11_w1/_w4`) |
| output | `results/spike_inv_w{1,4}/seed0/metrics.jsonl` (per-step train loss, mean over the same 8 micro-batches at both world sizes) |
| pass | per-step train loss agrees at bf16-rounding level for the first steps and the traces stay within the cross-seed spread from S2 through step 400 (past the first shard boundary at step 189, where the records' own loaders would diverge; the unified loader must not). Data equivalence itself is already verified statically (`tools/loader_equivalence.py`) |
| fail | systematic divergence → the accumulation path is not equivalent; the proposal requires a redesign before any single-GPU result counts |

## S4 — Does the historical code run on the pinned stack?

| | |
|---|---|
| commands | (a) `docker build` above succeeds and `docker run --rm $DOCKER_TAG pip freeze > launch/env.lock.frozen`; (b) every preset for 20 steps: `for n in 11 12 13 14 15 16 17 18; do torchrun --standalone --nproc_per_node=1 train_flagged.py --config configs/r$n.yaml --seed 0 --out-dir results/spike_run_r$n/seed0 --throughput-steps 20; done`; (c) the unmodified snapshots of #12 and #18 for a few minutes: `timeout 900 torchrun --standalone --nproc_per_node=1 records/r12/log_source/train_gpt2.py` (same for `r18/log_source/train_gpt.py`) |
| output | `launch/env.lock.frozen`; `results/spike_run_r{11..18}/seed0/{timing.jsonl,metrics.jsonl,DONE}`; console of (c) |
| pass | (a) the pinned nightly `torch==2.6.0.dev20241231+cu126` installs; (b) all eight presets reach DONE with finite train loss and no start-up assertion (prerequisites, optimizer coverage, 65,536 micro-batch); (c) both snapshots compile and train (FlexAttention fits A100 shared memory; cuDNN SDPA is available for #11's dense path in (b)) |
| fail | (a) nightly no longer downloadable → stop and ask (pins are not changed by this build); (b)/(c) FlexAttention or SDPA errors on the A100 → fallback per proposal §8 (Track 3 trainer) is a group decision |
