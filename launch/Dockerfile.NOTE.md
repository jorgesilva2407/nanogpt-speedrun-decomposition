`launch/Dockerfile` is upstream record #18's Dockerfile (`45544f5:Dockerfile`) with **one deviation: the torch
build**. Everything else is unchanged: CUDA 12.6.2-cudnn-devel-ubuntu24.04, Python 3.12.7 built from source,
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
