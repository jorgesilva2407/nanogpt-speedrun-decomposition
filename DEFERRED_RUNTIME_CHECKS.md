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
| FlexAttention kernels fit A100 shared memory (head_dim 128, fwd + bwd) | NOT RUN (by design) | one fwd+bwd of preset r12 and r18 on the A100 |
| pinned nightly `torch==2.6.0.dev20241231+cu126` still installable | NOT RUN (by design) | `docker build` of `launch/Dockerfile` succeeds; else report and stop (pins are not to be changed by this build) |

## r11 (record #11)

| check | status | how / pass criterion |
|---|---|---|
| state_dict load | NOT RUN (by design) | parameter names/shapes equal the snapshot's (static gate PASS: 99 tensors, 162,201,642 parameters); load a state_dict produced by the snapshot script into `train_flagged.py --config configs/r11.yaml` with `strict=True` |
| forward equivalence | NOT RUN (by design) | same weights + same batch through snapshot and flagged models → identical loss (bitwise, same process/device) |
| smoke train | NOT RUN (by design) | 20 steps, 8 GPUs, preset r11: loss decreases; val at step 0 ≈ 10.83 (record log: `step:0/3000 val_loss:…`) |
| torch.compile behaviour | NOT RUN (by design) | compile succeeds with `coordinate_descent_tuning`; no graph breaks warned beyond the snapshot's |
| fixed point | NOT RUN (by design) | full run, 8-GPU: val-loss curve within seed noise of the committed log (final 3.2753 at step 3000) |
