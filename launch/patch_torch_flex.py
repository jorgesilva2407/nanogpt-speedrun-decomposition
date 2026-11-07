# Backports pytorch/pytorch d2ec7f0756 (#143299, 2024-12-17, "[FlexAttention] Allow num_warps 8 since when block size
# >=128") into the installed torch 2.6.0. Run once at image build time; see launch/Dockerfile.NOTE.md.
#
# Why: v2.6.0 was branched before that commit. Its FlexAttention backward lowering picks (64, 128, 8, 3) as the only
# config on H100 for bf16 head_dim 128 (every preset: 768 / 6 heads) and then skips every num_warps == 8 config, so the
# choice list is empty and torch.compile fails with NoValidChoicesError (smoke test 2, 2026-11-07). Upstream #18's
# nightly (2.6.0.dev20241231) contains the commit. Both hunks are applied verbatim, so H100 and A100 get the nightly's
# default backward configs.
import os
import sys

import torch

assert torch.__version__ == "2.6.0+cu126", torch.__version__
path = os.path.join(os.path.dirname(torch.__file__), "_inductor", "kernel", "flex_attention.py")

HUNKS = [
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
]

src = open(path).read()
for old, new in HUNKS:
    n = src.count(old)
    if n != 1:
        sys.exit(f"patch_torch_flex: expected 1 match, found {n}:\n{old}")
    src = src.replace(old, new)
open(path, "w").write(src)
print(f"patch_torch_flex: applied d2ec7f0756 to {path}")
