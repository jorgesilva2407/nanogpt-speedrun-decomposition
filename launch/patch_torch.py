# Backports two upstream pytorch commits into the installed torch 2.6.0. Run once at image build time; see
# launch/Dockerfile.NOTE.md (deviations 2 and 3). Each hunk is applied verbatim and must match exactly once, so a
# different torch fails the build instead of the rental.
#
# 1. d2ec7f0756 (#143299, 2024-12-17, "[FlexAttention] Allow num_warps 8 since when block size >=128"), in
#    torch/_inductor/kernel/flex_attention.py. v2.6.0 was branched before it: its FlexAttention backward lowering picks
#    (64, 128, 8, 3) as the only config on H100 for bf16 head_dim 128 (every preset: 768 / 6 heads) and then skips every
#    num_warps == 8 config, so torch.compile fails with NoValidChoicesError (smoke test 2, 2026-11-07). Upstream #18's
#    nightly (2.6.0.dev20241231) contains the commit.
# 2. ec93aa7f84 (#148921, 2025-03-13, "fix cuDNN SDPA meta registration"), in torch/_meta_registrations.py. Since
#    1565eba4b4 (#138354, 2024-11-04, in v2.6.0 and in #18's nightly) the cuDNN SDPA kernel returns its output in the
#    query's memory layout, but the v2.6.0 meta function still predicts a contiguous (B, H, S_Q, D_V) tensor, so
#    compiled code on the dense #11 path (cuDNN SDPA forced) fails assert_size_stride (throughput fp_r11, 2026-11-07).
import os
import sys

import torch

assert torch.__version__ == "2.6.0+cu126", torch.__version__
TORCH = os.path.dirname(torch.__file__)

PATCHES = {
    "d2ec7f0756": ("_inductor/kernel/flex_attention.py", [
        (
            "        elif capability >= (8, 0):  # A100\n"
            "            if head_dim == 64:\n"
            "                return (32, 128, 4, 3)\n"
            "            elif head_dim == 128:\n"
            "                return (64, 128, 8, 3)\n"
            "            else:\n"
            "                return (64, 64, 4, 2)\n",
            "        elif capability >= (8, 0):\n"
            "            if head_dim >= 64:\n"
            "                return (32, 128, 4, 3)\n"
            "            elif head_dim == 128:\n"
            "                # SM86/89 have smaller shared memory sizes\n"
            "                num_stages = 3 if capability[-1] == 0 else 2\n"
            "                return (64, 64, 4, num_stages)\n"
            "            else:\n"
            "                return (64, 64, 4, 2)\n",
        ),
        (
            "            continue\n"
            "        if num_warps == 8:\n"
            "            # Working around https://github.com/pytorch/pytorch/issues/141603\n"
            "            continue\n",
            "            continue\n",
        ),
    ]),
    "ec93aa7f84": ("_meta_registrations.py", [
        (
            "    res = torch.empty((B, H, S_Q, D_V), dtype=query.dtype, device=query.device)\n"
            "    logsum_exp = torch.empty(\n",
            "    res_shape = (B, H, S_Q, D_V)\n"
            "    if tuple(query.shape) == res_shape:\n"
            "        query_t = query.transpose(1, 2)\n"
            "        res = torch.empty_like(query_t).transpose(1, 2)\n"
            "    else:\n"
            "        dim_order = sorted(\n"
            "            [0, 1, 2, 3], key=lambda idx: query.stride()[idx], reverse=True\n"
            "        )\n"
            "        permuted_shape = [res_shape[idx] for idx in dim_order]\n"
            "        final_permute = [dim_order.index(i) for i in range(len(dim_order))]\n"
            "        res = torch.empty(\n"
            "            permuted_shape, dtype=query.dtype, device=query.device\n"
            "        ).permute(final_permute)\n"
            "\n"
            "    logsum_exp = torch.empty(\n",
        ),
    ]),
}

for commit, (rel, hunks) in PATCHES.items():
    path = os.path.join(TORCH, rel)
    src = open(path).read()
    for old, new in hunks:
        n = src.count(old)
        if n != 1:
            sys.exit(f"patch_torch {commit}: expected 1 match in {rel}, found {n}:\n{old}")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"patch_torch: applied {commit} to {path}")
