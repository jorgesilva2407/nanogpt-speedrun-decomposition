`launch/Dockerfile` is upstream record #18's Dockerfile (`45544f5:Dockerfile`) with **two deviations: the torch
build, and a one-commit backport into it** (below). Everything else is unchanged: CUDA 12.6.2-cudnn-devel-ubuntu24.04, Python 3.12.7 built from source,
`pip install -r requirements.txt`.

| | upstream #18 | this repository |
|---|---|---|
| torch | `torch==2.6.0.dev20241231+cu126` (nightly index) | `torch==2.6.0+cu126` (stable index `whl/cu126`) |
| order | requirements, then torch | torch, then requirements (so pip does not first install the latest torch from PyPI, which would also leave it in an image layer) |

**Why.** PyTorch prunes old nightlies. On 2026-10-18 the nightly index for cu126 only held `2.15.0.dev` builds
(Aug–Sep 2026), so the upstream Dockerfile no longer builds. The user decided on 2026-10-18 to use the stable 2.6.0
release. What changes as a result, verified statically, is in `CHANGES/infra.md` ("compat: torch pin").

Build from the repository root so `requirements.txt` is in the context:
`docker build -f launch/Dockerfile -t modded-nanogpt-flagged:<git-sha> .`
`data/cached_fineweb10B.py` is upstream's download script (same commit), unchanged.

## Deviation 2: FlexAttention backport (2026-11-07)

`launch/patch_torch_flex.py` applies pytorch/pytorch `d2ec7f0756` (#143299, 2024-12-17, "[FlexAttention] Allow
num_warps 8 since when block size >=128") to the installed `torch/_inductor/kernel/flex_attention.py`, verbatim
(both hunks; the script asserts each old block matches exactly once, so a different torch fails the build).

**Why.** v2.6.0 was branched after `d35358b` (#141653, 2024-12-03), which skips every `num_warps == 8` FlexAttention
backward config, and before `d2ec7f0756`, which removes that skip. On H100, bf16 at head_dim 128 (every FlexAttention
preset: 768 / 6 heads), the only backward config without `max_autotune` is `(64, 128, 8, 3)`, so the choice list was
empty and `torch.compile` failed at the first `loss.backward()` with `NoValidChoicesError ... target:
flex_attention_backward` (smoke test 2, fp_r18 seed 0, 2026-11-07; log in the results repo at `_logs/`). Upstream
#18's nightly `2.6.0.dev20241231` contains `d2ec7f0756`, so with the backport the default backward configs are the
nightly's on H100 `(64, 128, 8, 3)` and on A100 `(32, 128, 4, 3)`. The static comparison of 2026-10-18 (CHANGES/infra.md)
compared the config tables but missed the skip.

