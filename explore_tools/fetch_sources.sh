#!/bin/sh
# fetch_sources.sh — re-create the external material that is not versioned (see .gitignore), from pinned SHAs.
# Idempotent: every step is skipped when its output already exists. Read-only on everything it downloads.
#   .venv/                  CPU-only tooling (numpy, pyyaml, pytest; no torch)          requirements-tools.txt
#   upstream/               github.com/KellerJordan/modded-nanogpt, full history + PR heads (refs/remotes/origin/pr/*)
#   records/rNN/tree/       `git archive <tree SHA>` per record (MANIFEST.md); log_source/ and ANCHOR.txt are rewritten
#                           too and must come out identical to the committed ones (checked at the end)
#   ext/gist_r15, gist_r17  YouJiacheng's gists that corroborate the #15 and #17 snapshots
#   ext/torch_src/          torch sources read for CHANGES/infra.md: the record nightly's day (flat) and v2.6.0/
# Usage: sh explore_tools/fetch_sources.sh            (from anywhere; network needed for missing pieces)
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
cd "$R"

if [ ! -x .venv/bin/python ]; then
  echo "== .venv"
  python3 -m venv .venv
  .venv/bin/python -m pip install -q -r requirements-tools.txt
fi

if [ ! -d upstream/.git ]; then
  echo "== upstream (full history + PR heads)"
  git clone -q https://github.com/KellerJordan/modded-nanogpt upstream
  git -C upstream fetch -q origin '+refs/pull/*/head:refs/remotes/origin/pr/*'
fi

# NN  tree SHA (full)                             log path (read at 5deb47d)                                                  script        code SHA
SNAPSHOTS='
11 ff93d20bd0853e173a0977bd3aba3cc01750e117 records/111024_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt train_gpt2.py ff93d20b
12 ebce774d08b6646d8e80b4d2ac3d14c687534679 records/111924_FlexAttention/8384493d-dba9-4991-b16b-8696953f5e6d.txt train_gpt2.py ebce774
13 759b8fc74c2ef30e5c78ceec65bcb6a190a61399 records/112424_WindowWarmup/cf9e4571-c5fc-4323-abf3-a98d862ec6c8.txt train_gpt2.py 759b8fc
14 49c3afac4b9ebe2d5b541d12295e770a8bfeef94 records/120424_ValueEmbed/00008ea0-21dd-442a-82ee-d12799249d0f.txt train_gpt2.py 49c3afa
15 155afd523552de06abf76f88f9676cdec92ed084 records/120824_UNetValueEmbedsTweaks/0069607b-aa90-49fd-9766-4368bcd168c4.txt train_gpt2.py none
16 0690f2c7dfd368478ed15b867b329c4a8a9c6f71 records/121024_MFUTweaks/04df8c70-2462-4b6c-9e49-c7589b321a81.txt train_gpt2.py 0690f2c
17 e76ab996da9e761ab270e6e81317d75c1a5688a4 records/121724_SparsifyEmbeds/165384d5-5a41-4fba-9b79-8ecc372616ea.txt train_gpt2.py none
18 45544f533ae9528ce0f94f2416fd9bd59c2ad54c records/010425_SoftCap/31d6c427-f1f7-4d8a-91be-a67b5dcd13fd.txt train_gpt.py 45544f5
'
echo "$SNAPSHOTS" | while read -r NN TREE LOG SCRIPT CODE; do
  [ -n "$NN" ] || continue
  if [ ! -d "records/r$NN/tree" ]; then
    echo "== records/r$NN"
    sh explore_tools/export_snapshot.sh "$NN" "$TREE" "$LOG" "$SCRIPT" "$CODE" > /dev/null
  fi
done

gist() {  # gist <dir> <gist id> <pinned commit>
  if [ ! -d "ext/$1/repo/.git" ]; then
    echo "== ext/$1"
    mkdir -p "ext/$1"
    git clone -q "https://gist.github.com/$2.git" "ext/$1/repo"
    git -C "ext/$1/repo" checkout -q "$3"
  fi
}
gist gist_r15 1fed95270177aa04f389b46e8c7d7733 866bad1
gist gist_r17 dff723d9a362303f4dca6d71a6469555 ab98f29

T=ext/torch_src
NIGHTLY=8d9ff9c8a414c66809a2be63870a8f1c7ab7ea09   # pytorch main, 2024-12-31 23:45 UTC (the nightly's day)
SDP_MAIN=09c950cc872dfcee453307db47fa10553c3f5616  # last change to sdp_utils.cpp before the nightly
torchfile() {  # torchfile <out name> <ref> <path in pytorch/pytorch>
  if [ ! -s "$T/$1" ]; then
    mkdir -p "$(dirname "$T/$1")"
    curl -fsSL "https://raw.githubusercontent.com/pytorch/pytorch/$2/$3" -o "$T/$1"
    echo "== $T/$1"
  fi
}
torchfile sdp_utils_main_20241222.cpp "$SDP_MAIN" aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
torchfile sdp_utils_v2.6.0.cpp v2.6.0 aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
torchfile torch_nn_attention_flex_attention.py "$NIGHTLY" torch/nn/attention/flex_attention.py
torchfile torch__inductor_kernel_flex_attention.py "$NIGHTLY" torch/_inductor/kernel/flex_attention.py
torchfile torch_compiler___init__.py "$NIGHTLY" torch/compiler/__init__.py
torchfile torch_nn___init__.py "$NIGHTLY" torch/nn/__init__.py
torchfile torch_nn_parameter.py "$NIGHTLY" torch/nn/parameter.py
torchfile torch__inductor_config.py "$NIGHTLY" torch/_inductor/config.py
torchfile torch_backends_cuda___init__.py "$NIGHTLY" torch/backends/cuda/__init__.py
torchfile torch_backends_cudnn___init__.py "$NIGHTLY" torch/backends/cudnn/__init__.py
# the release actually pinned since 2026-10-18 (launch/Dockerfile.NOTE.md); checked by tools/torch_api_check.py
torchfile v2.6.0/sdp_utils.cpp v2.6.0 aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
torchfile v2.6.0/torch_nn_attention_flex_attention.py v2.6.0 torch/nn/attention/flex_attention.py
torchfile v2.6.0/torch__inductor_kernel_flex_attention.py v2.6.0 torch/_inductor/kernel/flex_attention.py
torchfile v2.6.0/torch_compiler___init__.py v2.6.0 torch/compiler/__init__.py
torchfile v2.6.0/torch_nn___init__.py v2.6.0 torch/nn/__init__.py
torchfile v2.6.0/torch_nn_parameter.py v2.6.0 torch/nn/parameter.py
torchfile v2.6.0/torch__inductor_config.py v2.6.0 torch/_inductor/config.py
torchfile v2.6.0/torch_backends_cuda___init__.py v2.6.0 torch/backends/cuda/__init__.py
torchfile v2.6.0/torch_backends_cudnn___init__.py v2.6.0 torch/backends/cudnn/__init__.py

if git status --porcelain -- records | grep -q .; then
  echo "WARNING: re-exported snapshots differ from the committed ones:"; git status --short -- records; exit 1
fi
echo "sources ready; committed snapshots reproduced unchanged"
