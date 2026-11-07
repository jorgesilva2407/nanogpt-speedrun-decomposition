FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7
ENV PATH=/usr/local/bin:$PATH

RUN apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12 /usr/local/bin/pip

COPY requirements.txt /modded-nanogpt/requirements.txt
WORKDIR /modded-nanogpt

# DEVIATION from upstream #18: the pinned nightly torch==2.6.0.dev20241231+cu126 is no longer on PyTorch's index;
# the stable 2.6.0+cu126 release is used instead (decided 2026-10-18; see launch/Dockerfile.NOTE.md). It is installed
# before requirements.txt so pip does not first pull the latest torch from PyPI.
RUN python -m pip install --upgrade pip && \
    pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126

RUN pip install -r requirements.txt

# DEVIATIONS 2 and 3: backport pytorch d2ec7f0756 (#143299; FlexAttention backward has no valid config on H100 at
# head_dim 128 without it) and ec93aa7f84 (#148921; cuDNN SDPA meta registration, without it the dense #11 path fails
# assert_size_stride under torch.compile) into torch 2.6.0. See launch/Dockerfile.NOTE.md. Applied after
# requirements.txt so nothing reinstalls torch over the patched files.
COPY launch/patch_torch.py /modded-nanogpt/patch_torch.py
RUN python /modded-nanogpt/patch_torch.py && \
    T=$(python -c "import os, torch; print(os.path.dirname(torch.__file__))") && \
    rm -f "$T"/_inductor/kernel/__pycache__/flex_attention*.pyc "$T"/__pycache__/_meta_registrations*.pyc && \
    python -m py_compile "$T/_inductor/kernel/flex_attention.py" "$T/_meta_registrations.py" && \
    ! grep -q "issues/141603" "$T/_inductor/kernel/flex_attention.py" && \
    grep -q "res = torch.empty_like(query_t).transpose(1, 2)" "$T/_meta_registrations.py"

CMD ["bash"]
ENTRYPOINT []
