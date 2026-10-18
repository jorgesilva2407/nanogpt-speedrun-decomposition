`launch/Dockerfile` is upstream record #18's Dockerfile, byte-identical (`45544f5:Dockerfile`): CUDA
12.6.2-cudnn-devel-ubuntu24.04, Python 3.12.7, `pip install -r requirements.txt`, then
`torch==2.6.0.dev20241231+cu126` from the PyTorch nightly index. Pins are not changed by this repository.
Build from the repository root so `requirements.txt` is in the context:
`docker build -f launch/Dockerfile -t modded-nanogpt-flagged:<git-sha> .`
`data/cached_fineweb10B.py` is upstream's download script (same commit), unchanged.
