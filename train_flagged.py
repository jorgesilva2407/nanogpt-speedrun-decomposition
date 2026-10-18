import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
def _peek_flags_before_torch(argv):
    # rec-17 (code_opts_17) sets the CUDA allocator before torch is imported, so read the config's flags first
    if '--config' not in argv:
        return {}
    path = argv[argv.index('--config') + 1]
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            import yaml
            return (yaml.safe_load(f) or {}).get('flags') or {}
        import json as _json
        return _json.load(f).get('flags') or {}
_early_flags = _peek_flags_before_torch(sys.argv)
if _early_flags.get('code_opts_17') and not _early_flags.get('code_opts_18'): # rec-17 only (rec-18 drops it)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import uuid
import glob
import time
import json
import contextlib
import argparse
from dataclasses import dataclass, fields

import numpy as np
import torch
if _early_flags.get('code_opts_18'): # rec-18 runs a dummy backward right after importing torch
    torch.empty(1, device='cuda', requires_grad=True).backward()
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
# Use of FlexAttention contributed by @KoszarskyB (rec-12; used only when flags.flex_kernel)
from torch.nn.attention.flex_attention import BlockMask, flex_attention as flex_attention_raw, create_block_mask as create_block_mask_raw
# rec-12..14 call compiled wrappers; rec-15 (code_opts_15) calls flex_attention directly inside the compiled model
flex_attention = torch.compile(flex_attention_raw, dynamic=False)
create_block_mask = torch.compile(create_block_mask_raw, dynamic=False)

# -----------------------------------------------------------------------------
# Flags: one per record mechanism. FLAGS holds the r11 value (the floor). Every preset lists every flag
# (configs/rNN.yaml); REQUIRES lists mechanical prerequisites, enforced at start-up by name.
FLAGS = {
    # rec-12
    'flex_kernel': False,    # attention through FlexAttention instead of cuDNN SDPA (same attention pattern)
    'long_context': False,   # 1 row of 64K tokens with a sliding window, global positions (vs 64 rows of 1024 causal)
    'doc_masking': False,    # no attention across document boundaries (token 50256 starts a document)
    'rotary_variant': 'v11', # rotary: 'v11' inv_freq on CPU at init | 'v12' on device at first use | 'v14' bf16 buffer,
                             # rotates itself | 'v17' cos/sin tables precomputed in fp32 as buffers, rotation in fp32
    # rec-13
    'window_warmup': False,  # attention window grows linearly over training (window_warmup_* hparams) instead of fixed
    # rec-14
    'value_embeds': False,   # VE rung 1: token value embeddings (one vocab x n_layer*n_embd table) replace the layer-0 value residual
    # rec-15
    've_unet': False,        # VE rung 2: n_layer/2 value-embedding chunks, decoder layers reuse them in mirrored order
    've_lambdas2': False,    # value mixing v = l0*v + l1*x with two free scalars (init 0.5, 0.5) instead of (1-lamb)*v + lamb*x
    'code_opts_15': False,   # rec-15 throughput changes with identical numerics: Muon all_gather distribution, pinned
                             # non-blocking loader, persistent window tensor, uncompiled flex_attention call
    'muon_ema_momentum': False, # Muon momentum as an EMA (buf.lerp_(g, 1-m); g.lerp_(buf, m)) instead of buf = m*buf + g
    'blockmask_manual': False,  # block mask built directly at 128-token-block granularity (BlockMask.from_kv_blocks)
    # rec-16
    've_split': False,       # VE rung 3a: one nn.Embedding per value-embedding chunk (module value_embeds, built after the blocks)
    'block_window': False,   # VE rung 3b's "block sliding window": window counted in whole 128-token blocks, no token-level window
    'blockmask_separate': False, # "separate block mask": fully-unmasked blocks passed as full blocks (no mask_mod evaluation)
    'code_opts_16': False,   # rec-16 throughput changes with identical numerics: DDP gradient_as_bucket_view and no buffer
                             # broadcast, set_stance(skip_guard_eval_unsafe) from step 5, unbuffered shard reads, enable_gqa
    # rec-17
    've_sparsify': False,    # VE rung 4: ve_sparse_tables tables used by the first and last layers only (none in between)
    'rope_half_truncate': False, # "improved rotary" frequencies: (1/1024)**linspace(0,1,dim/4) on half the pairs, zero on the rest
    'drop_attn_layer': False,  # layer dropped_attn_layer has no attention (capacity change)
    'code_opts_17': False,   # rec-17 throughput changes: expandable_segments allocator; enable_gqa dropped again
    # rec-18
    'fp32_small_params': False, # model kept in fp32 with only the embeddings in bf16 (rec-18 bf16_embeds) instead of
                                # model.bfloat16() + fp32 linears: scalars (lambdas, skip weights) and rotary tables stay fp32
    'code_opts_18': False,   # rec-18 rewrite, throughput/implementation: one Adam with param groups, whole batch fetched
                             # at step start, no no_sync, window tensor copied every step, no coordinate-descent tuning,
                             # no set_stance/enable_gqa/expandable_segments, Muon normalises after the transpose,
                             # dummy backward after importing torch
}
FLAG_CHOICES = {'rotary_variant': ('v11', 'v12', 'v14', 'v17')}
REQUIRES = {'long_context': ['flex_kernel'], 'doc_masking': ['flex_kernel'], 'window_warmup': ['flex_kernel'],
            've_unet': ['value_embeds'], 'blockmask_manual': ['flex_kernel', 'long_context'],
            've_split': ['ve_unet'], 'block_window': ['blockmask_manual'], 'blockmask_separate': ['block_window'],
            've_sparsify': ['ve_split']}

class Flags:
    def __init__(self, d):
        self.__dict__.update(d)

def load_flags(cfg):
    given = dict(cfg.get('flags') or {})
    missing, unknown = set(FLAGS) - set(given), set(given) - set(FLAGS)
    assert not missing and not unknown, f"config flags mismatch: missing={sorted(missing)} unknown={sorted(unknown)}"
    for k, v in given.items():
        if k in FLAG_CHOICES:
            assert v in FLAG_CHOICES[k], f"flag {k}={v!r} not in {FLAG_CHOICES[k]}"
        else:
            assert isinstance(v, bool), f"flag {k} must be true/false, got {v!r}"
    for k, reqs in REQUIRES.items():
        if given[k]:
            for r in reqs:
                assert given[r], f"flag {k} requires flag {r} (set {r}: true)"
    return Flags(given)

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    r"""
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if flags.code_opts_18: # rec-18: transpose first, then normalise (same norm; reduction over a transposed view)
        if G.size(0) > G.size(1):
            X = X.T
        # Ensure spectral norm is at most 1
        X = X / (X.norm() + 1e-7)
    else:
        X /= (X.norm() + eps) # ensure top singular value <= 1
        if G.size(0) > G.size(1):
            X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        if flags.code_opts_15:
            # rec-15: parameters grouped by size, each rank orthogonalises one parameter per round and the updates are
            # exchanged with an async all_gather overlapped with the next round
            self.num_process = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ["RANK"])
            params = list(params)
            assert all(isinstance(p, torch.Tensor) for p in params)
            sizes = {p.numel() for p in params}
            param_groups = [
                {
                    "params": [p for p in params if p.numel() == size],
                    "update_buffer": [
                        torch.empty(size, device="cuda", dtype=torch.bfloat16)
                        for _ in range(self.num_process)
                    ],
                }
                for size in sizes
            ]
            super().__init__(param_groups, defaults)
        else:
            super().__init__(params, defaults)

    def momentum_update(self, g, state, momentum, nesterov):
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(g)
        buf = state['momentum_buffer']
        if flags.muon_ema_momentum: # rec-15
            buf.lerp_(g, 1 - momentum)
            return g.lerp_(buf, momentum) if nesterov else buf
        buf.mul_(momentum).add_(g)
        if nesterov:
            g = g.add(buf, alpha=momentum)
        return g

    def step(self):
        if flags.code_opts_15:
            return self.step_all_gather()

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    g = self.momentum_update(g, state, momentum, group['nesterov'])
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

    def step_all_gather(self): # rec-15
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            zeropower_backend = zeropower_backends[group["backend"]]
            backend_steps = group["backend_steps"]
            update_buffers = group["update_buffer"]
            # generate weight updates in distributed fashion
            params = group["params"]
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    p_world.data.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.num_process]:
                # rec-17: the last round may hold fewer parameters than ranks (44 square matrices once a layer has
                # no attention); a rank without a parameter sends its stale buffer, which no rank applies
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    g = self.momentum_update(g, state, momentum, nesterov)
                    g = zeropower_backend(g, steps=backend_steps).flatten()
                else:
                    g = update_buffers[self.rank]
                update_prev()
                handle = dist.all_gather(update_buffers, g, async_op=True)
                params_world = params[base_i : base_i + self.num_process]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def half_truncated_inv_freq(dim, device=None):
    # rec-17 "improved rotary" (half-truncate RoPE by @YouJiacheng): base-1024 frequencies on dim/4 pairs, zero on the rest
    inv_freq = (1 / 1024) ** torch.linspace(0.0, 1.0, steps=dim // 4, dtype=torch.float32, device=device)
    return torch.cat([inv_freq, inv_freq.new_zeros(dim // 4)])

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000, max_seq_len=65536):
        super().__init__()
        self.dim = dim
        self.base = base
        if flags.rotary_variant == 'v11':
            if flags.rope_half_truncate:
                self.inv_freq = half_truncated_inv_freq(dim)
            else:
                self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        elif flags.rotary_variant == 'v12':
            self.inv_freq = None
        elif flags.rotary_variant == 'v14': # a registered buffer, so model.bfloat16() casts it to bf16 (rec-14)
            if flags.rope_half_truncate:
                self.register_buffer('inv_freq', half_truncated_inv_freq(dim))
            else:
                self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        else: # 'v17': tables for every position precomputed in fp32; buffers, so a global cast also casts them (rec-17)
            if flags.rope_half_truncate:
                inv_freq = half_truncated_inv_freq(dim)
            else:
                inv_freq = (1 / base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            t = torch.arange(max_seq_len, dtype=torch.float32)
            theta = torch.einsum("i, j -> ij", t, inv_freq)
            self.cos = nn.Buffer(theta.cos(), persistent=False)
            self.sin = nn.Buffer(theta.sin(), persistent=False)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if flags.rotary_variant == 'v17': # rec-17
            cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
            x1, x2 = x.to(dtype=torch.float32).chunk(2, dim=-1)
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
            return torch.cat((y1, y2), 3).type_as(x)
        if flags.rotary_variant == 'v14':
            # rec-14: angles from the bf16 buffer, tables kept in that dtype, rotation applied here
            if seq_len != self.seq_len_cached:
                t = torch.arange(seq_len, device=x.device)
                freqs = torch.outer(t, self.inv_freq)
                self.seq_len_cached = seq_len
                self.cos_cached = freqs.cos()
                self.sin_cached = freqs.sin()
            cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
            # apply_rotary_emb(x, cos, sin)
            x1, x2 = x.chunk(2, dim=3)
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
            return torch.cat((y1, y2), 3).type_as(x)
        if seq_len != self.seq_len_cached:
            if flags.rotary_variant == 'v11':
                self.seq_len_cached = seq_len
                t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
                freqs = torch.outer(t, self.inv_freq).to(x.device)
            else: # 'v12'
                if flags.rope_half_truncate:
                    self.inv_freq = half_truncated_inv_freq(self.dim, device=x.device)
                else:
                    self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
                self.seq_len_cached = seq_len
                t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
                freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_v = CastedLinear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)
        if flags.ve_lambdas2:
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5])) # @Grad62304977 (rec-15)
        else:
            self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977

    def forward(self, x, v1, block_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None and flags.value_embeds:
            # rec-17 (ve_sparsify): this layer has no value embedding. Record: v = lambdas[0] * v. With a single lamb
            # (non-historical combination) the same rule reads (1 - lamb) * v.
            v = self.lambdas[0] * v if flags.ve_lambdas2 else (1 - self.lamb) * v
        elif v1 is None:
            v1 = v # This happens if we are in the first block. v needs to be accessed by subsequent blocks
        # with flags.value_embeds the caller passes this layer's value embedding as v1 (never None), which gives
        # rec-14's v = (1 - lamb) * v + lamb * vi.view_as(v)
        if v1 is None:
            pass # rec-17: handled above
        elif flags.ve_lambdas2:
            v = self.lambdas[0] * v + self.lambdas[1] * v1.view_as(v) # @Grad62304977 (rec-15)
        else:
            v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977
        if flags.rotary_variant in ('v14', 'v17'):
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
            q, k = self.rotary(q), self.rotary(k)
        else:
            cos, sin = self.rotary(q)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if flags.flex_kernel:
            fa = flex_attention_raw if flags.code_opts_15 else flex_attention
            if flags.code_opts_16 and not flags.code_opts_17 and not flags.code_opts_18: # rec-16 only passes enable_gqa=True
                y = fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, enable_gqa=True)
            else:
                y = fa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        else:
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = CastedLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = CastedLinear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        # rec-17: skip attention of layer dropped_attn_layer (by @YouJiacheng)
        self.use_attn = not (flags.drop_attn_layer and layer_idx == config.dropped_attn_layer)
        if self.use_attn:
            self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.use_attn:
            x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1, block_mask)
            x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    logit_softcap : int = 30
    attn_window : int = 1024 # sliding-window length in tokens (long_context only)
    embed_vocab_size : int = 50304 # rows of the token and value embedding tables (rec-17: 50257; lm_head keeps vocab_size)
    dropped_attn_layer : int = 7 # drop_attn_layer only
    ve_sparse_tables : int = 3 # ve_sparsify only

class ValueEmbedding(nn.Module):
    # rec-16 (ve_split): the value-embedding chunks as separate tables; the U-net mirror as a list
    def __init__(self, config):
        super().__init__()
        n_tables = config.ve_sparse_tables if flags.ve_sparsify else config.n_layer // 2
        self.embed = nn.ModuleList([
            nn.Embedding(config.embed_vocab_size, config.n_embd)
            for _ in range(n_tables)
        ])
        self.n_layer = config.n_layer

    def forward(self, inputs):
        ve = [emb(inputs).bfloat16() if flags.fp32_small_params else emb(inputs) for emb in self.embed]
        if flags.ve_sparsify:
            # rec-17: 012 ... 012 structure (first and last layers only), improved on the U-net structure
            k = len(ve)
            return ve[:k] + [None] * (self.n_layer - 2 * k) + ve[:k]
        ve += reversed(ve)
        return ve

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        modules = dict(wte = nn.Embedding(config.embed_vocab_size, config.n_embd))
        if flags.value_embeds and not flags.ve_split:
            # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning (rec-14)
            # U-net structure on token value embeddings by @leloykun (rec-15): n_layer // 2 chunks
            n_ve = config.n_layer // 2 if flags.ve_unet else config.n_layer
            modules['vte'] = nn.Embedding(config.embed_vocab_size, config.n_embd*n_ve)
        modules['h'] = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.transformer = nn.ModuleDict(modules)
        if flags.ve_split:
            # token value embeddings by @KoszarskyB; U-net structure by @leloykun; split into tables (rec-16).
            # Created after the blocks, as in the record (moves its draws after the blocks' in the init RNG stream).
            self.value_embeds = ValueEmbedding(config)

        # U-net design by @brendanh0gan
        self.encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.decoder_layers = config.n_layer - self.encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.decoder_layers))

        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, target, attn_blocksize=None):

        block_mask = None
        if flags.flex_kernel:
            # window: the step's warmed-up value (rec-13, an int tensor) or the fixed attn_window; in tokens, or in
            # 128-token blocks when flags.block_window (rec-16)
            if flags.window_warmup:
                window = attn_blocksize
            else:
                window = self.config.attn_window // 128 if flags.block_window else self.config.attn_window
            if flags.block_window:
                # rec-16: window in whole blocks (`window` is a block count); token-level mask_mod is causal [& document] only
                BLOCK_SIZE = 128
                assert idx.ndim == 1
                docs = (idx == 50256).cumsum(0)
                docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
                docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
                def document_causal(b, h, q_idx, kv_idx):
                    causal_mask = q_idx >= kv_idx
                    if flags.doc_masking:
                        document_mask = docs[q_idx] == docs[kv_idx]
                        return causal_mask & document_mask
                    return causal_mask
                def dense_to_ordered(dense_mask):
                    num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
                    indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
                    return num_blocks[None, None].contiguous(), indices[None, None].contiguous()
                def create_doc_swc_block_mask(sliding_window_num_blocks):
                    kv_idx = block_idx = torch.arange(len(idx) // BLOCK_SIZE, dtype=torch.int32, device="cuda")
                    q_idx = block_idx[:, None]
                    causal_bm = q_idx >= kv_idx
                    causal_full_bm = q_idx > kv_idx
                    window_bm = q_idx - kv_idx < sliding_window_num_blocks
                    window_full_bm = window_bm
                    if flags.doc_masking:
                        document_bm = (docs_low[:, None] <= docs_high) & (docs_low <= docs_high[:, None])
                        document_full_bm = (docs_low[:, None] == docs_high) & (docs_low == docs_high[:, None])
                        nonzero_bm = causal_bm & window_bm & document_bm
                        full_bm  = causal_full_bm & window_full_bm & document_full_bm
                    else:
                        nonzero_bm = causal_bm & window_bm
                        full_bm  = causal_full_bm & window_full_bm
                    if flags.blockmask_separate:
                        kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm ^ full_bm)
                        full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)
                        return BlockMask.from_kv_blocks(
                            kv_num_blocks,
                            kv_indices,
                            full_kv_num_blocks,
                            full_kv_indices,
                            BLOCK_SIZE=BLOCK_SIZE,
                            mask_mod=document_causal,
                        )
                    kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm)
                    return BlockMask.from_kv_blocks(kv_num_blocks, kv_indices, BLOCK_SIZE=BLOCK_SIZE, mask_mod=document_causal)
                block_mask = create_doc_swc_block_mask(window)
                idx = idx[None]
            elif flags.long_context and flags.blockmask_manual:
                # rec-15: block-level mask built directly; partial blocks get the token-level mask_mod
                BLOCK_SIZE = 128
                assert idx.ndim == 1
                docs = (idx == 50256).cumsum(0)
                docs_low = docs.reshape(-1, BLOCK_SIZE)[:, 0].contiguous()
                docs_high = docs.reshape(-1, BLOCK_SIZE)[:, -1].contiguous()
                def document_sliding_window_causal(b, h, q_idx, kv_idx):
                    causal_mask = q_idx >= kv_idx
                    window_mask = q_idx - kv_idx < window
                    if flags.doc_masking:
                        document_mask = docs[q_idx] == docs[kv_idx]
                        return causal_mask & document_mask & window_mask
                    return causal_mask & window_mask
                S = len(idx)
                def create_sliding_window_causal_mask(S, sliding_window):
                    kv_idx = block_idx = torch.arange(S // BLOCK_SIZE, dtype=torch.int32, device="cuda")
                    q_idx = block_idx[:, None]
                    causal_mask = q_idx >= kv_idx
                    window_mask = q_idx - kv_idx < ((sliding_window + BLOCK_SIZE - 1) // BLOCK_SIZE)
                    if flags.doc_masking:
                        document_mask = (docs_low[q_idx] <= docs_high[kv_idx]) & (docs_low[kv_idx] <= docs_high[q_idx])
                        dense_mask = causal_mask & document_mask & window_mask
                    else:
                        dense_mask = causal_mask & window_mask
                    dense_mask = dense_mask.to(torch.int32)
                    num_blocks = dense_mask.sum(dim=-1).to(torch.int32)
                    indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True).to(torch.int32)
                    num_blocks = num_blocks[None, None, :].contiguous()
                    indices = indices[None, None, :].contiguous()
                    return BlockMask.from_kv_blocks(num_blocks, indices, BLOCK_SIZE=BLOCK_SIZE, mask_mod=document_sliding_window_causal)
                block_mask = create_sliding_window_causal_mask(S, window)
                idx = idx[None]
            elif flags.long_context:
                # rec-12: one 64K-token row (idx is 1-D), causal & sliding window [& same document]
                docs = (idx == 50256).cumsum(0)
                def document_causal_mask(b, h, q_idx, kv_idx):
                    causal_mask = q_idx >= kv_idx
                    window_mask = q_idx - kv_idx < window
                    if flags.doc_masking:
                        document_mask = docs[q_idx] == docs[kv_idx]
                        return causal_mask & document_mask & window_mask
                    return causal_mask & window_mask
                S = len(idx)
                block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)
                idx = idx[None]
            else:
                # record #11's pattern (rows of sequence_length tokens, causal within the row) [& same document]
                # with window_warmup the window also applies inside a row (inert once it exceeds the row length)
                Bm, Tm = idx.size()
                if flags.doc_masking:
                    docs = (idx == 50256).cumsum(1)
                    def row_document_causal_mask(b, h, q_idx, kv_idx):
                        m = (q_idx >= kv_idx) & (docs[b, q_idx] == docs[b, kv_idx])
                        return m & (q_idx - kv_idx < window) if flags.window_warmup else m
                    block_mask = create_block_mask(row_document_causal_mask, Bm, None, Tm, Tm, device="cuda", _compile=True)
                else:
                    def row_causal_mask(b, h, q_idx, kv_idx):
                        m = q_idx >= kv_idx
                        return m & (q_idx - kv_idx < window) if flags.window_warmup else m
                    block_mask = create_block_mask(row_causal_mask, None, None, Tm, Tm, device="cuda", _compile=True)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if flags.fp32_small_params:
            x = x.bfloat16() # rec-18 (no-op: embeddings are bf16)
        x = F.rms_norm(x, (x.size(-1),)) # @Grad62304977
        x0 = x
        v1 = None
        if flags.ve_split:
            vi = self.value_embeds(idx) # 12 entries: chunks 0..5 then 5..0 (rec-16)
        elif flags.value_embeds:
            vi = self.transformer.vte(idx).chunk(self.encoder_layers if flags.ve_unet else self.config.n_layer, dim=-1)

        # Store outputs for U-Net skip connections
        skip_connections = []

        # Encoder pass - process only the first half of the blocks
        for i in range(self.encoder_layers):
            x, v1 = self.transformer.h[i](x, vi[i] if flags.value_embeds else v1, x0, block_mask)
            skip_connections.append(x)  # Store the output for skip connections

        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.decoder_layers):
            skip_connection = skip_connections.pop()  # Get the corresponding encoder output
            # Apply learnable weight to skip connection
            weighted_skip = self.skip_weights[i] * skip_connection
            if flags.ve_split:
                ve = vi[self.encoder_layers + i] # ve_dec[i] of the mirrored list (rec-16)
            elif flags.value_embeds:
                # U-net structure on token value embeddings by @leloykun (rec-15): decoder layer i reuses chunk enc-1-i
                ve = vi[self.encoder_layers - 1 - i] if flags.ve_unet else vi[self.encoder_layers + i]
            else:
                ve = v1
            x, v1 = self.transformer.h[self.encoder_layers + i](x + weighted_skip, ve, x0, block_mask)

        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x)
        logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss.float()

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

def _load_data_shard_torch(filename, ntok):
    # rec-15 (code_opts_15): read straight into pinned memory, no numpy round trip; unbuffered from rec-16 (code_opts_16)
    with open(filename, "rb", buffering=0 if flags.code_opts_16 else -1) as f:
        tokens = torch.empty(ntok, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    # infra (unified batch convention): serves B x T-token micro-batches for a list of *virtual ranks*.
    # Virtual rank v reads exactly what rank v of a `vranks`-GPU run of the record reads (same offsets, same
    # shard-advance rule), so the token stream of the 8-GPU record run is reproduced on any world_size.
    # The record's own loader is the special case vranks == world_size, owned == [rank]. See CHANGES/infra.md.
    def __init__(self, filename_pattern, B, T, vranks, owned, shard_rule, replay_first_round=False, rows_2d=True, torch_io=False):
        self.B = B
        self.T = T
        self.rows_2d = rows_2d # False: return 1-D token tensors (rec-12 layout)
        self.torch_io = torch_io # rec-15: pinned uint16 shards, int32 inputs / int64 targets copied non_blocking
        self.vranks = vranks
        self.owned = list(owned)
        assert shard_rule in ('gt_plus1', 'ge', 'ge_plus1', 'global_ge_plus1'), f"unknown shard_rule {shard_rule}"
        self.shard_rule = shard_rule
        # record #11 consumed its first batch twice (prefetch, then train_loader.reset() before the loop);
        # replay_first_round reproduces that at step granularity: the first round over the owned virtual
        # ranks is served without advancing them.
        self.replay_first_round = replay_first_round

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        self.ntoks = []
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= vranks * B * T + 1
            ntok_total += int(shard_ntok)
            self.ntoks.append(int(shard_ntok))
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.shards = {} # shard index -> tokens, shared by the owned virtual ranks
        self.states = [dict(v=v, shard=0, pos=v * self.B * self.T) for v in self.owned]
        self.cursor = 0
        self.round = 0

    def shard_tokens(self, shard):
        if shard not in self.shards:
            live = {st['shard'] for st in self.states}
            for k in list(self.shards):
                if k not in live:
                    del self.shards[k]
            if self.torch_io:
                self.shards[shard] = _load_data_shard_torch(self.files[shard], self.ntoks[shard])
            else:
                self.shards[shard] = _load_data_shard(self.files[shard])
        return self.shards[shard]

    def next_batch(self):
        B = self.B
        T = self.T
        st = self.states[self.cursor]
        tokens = self.shard_tokens(st['shard'])
        if self.shard_rule == 'global_ge_plus1':
            # rec-18: one decision for the whole global batch, taken before reading: load next shard if necessary
            if (st['pos'] - st['v'] * B * T) + B * T * self.vranks + 1 >= len(tokens):
                st['shard'] = (st['shard'] + 1) % len(self.files)
                st['pos'] = st['v'] * B * T
                tokens = self.shard_tokens(st['shard'])
        buf = tokens[st['pos'] : st['pos']+B*T+1]
        if self.torch_io:
            # host side async is sufficient (rec-15)
            x = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # inputs
            y = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # targets
            if self.rows_2d:
                x, y = x.view(B, T), y.view(B, T)
        else:
            buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
            if self.rows_2d:
                x = (buf[:-1]).view(B, T) # inputs
                y = (buf[1:]).view(B, T) # targets
            else:
                x = buf[:-1] # inputs
                y = buf[1:] # targets
        if not (self.replay_first_round and self.round == 0):
            # advance current position and load next shard if necessary (rule of the record being replayed)
            st['pos'] += B * T * self.vranks
            if self.shard_rule == 'gt_plus1':
                advance = st['pos'] + (B * T * self.vranks + 1) > len(tokens) # record #11
            elif self.shard_rule == 'ge':
                advance = st['pos'] + B * T * self.vranks >= len(tokens) # records #12-#14
            elif self.shard_rule == 'ge_plus1':
                advance = st['pos'] + B * T * self.vranks + 1 >= len(tokens) # records #15-#17 (PR #43)
            else: # 'global_ge_plus1': decided before the read (rec-18)
                advance = False
            if advance:
                st['shard'] = (st['shard'] + 1) % len(self.files)
                st['pos'] = st['v'] * B * T
        self.cursor += 1
        if self.cursor == len(self.states):
            self.cursor = 0
            self.round += 1
        if self.torch_io:
            return x, y
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # Every field must be given by the config file (no defaults): see configs/rNN.yaml.
    # data hyperparams
    input_bin : str # input .bin to train on
    input_val_bin : str # input .bin to eval validation loss on
    # optimization hyperparams
    global_batch_tokens : int # tokens per optimizer step, across all devices (fixed; independent of world_size)
    micro_batch_tokens : int # tokens per device per forward/backward (65,536 in every record; never reduced)
    sequence_length : int # attention row length of the dense layout, in tokens
    loader_shard_rule : str # shard-advance rule of the record's loader (replayed per virtual rank)
    loader_replay_first_round : bool # record #11 trained on its first batch twice
    num_iterations : int # number of iterations to run
    warmup_iters : int
    warmdown_iters : int # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    lr_schedule_form : str # 'iters': warmup_iters/warmdown_iters (records #11-#17) | 'frac': cooldown_frac (record #18)
    cooldown_frac : float # lr_schedule_form 'frac': fraction of training spent cooling down (record #18: 0.4)
    weight_decay : float
    # evaluation and logging hyperparams
    val_loss_every : int # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int # every how many steps to save the checkpoint? 0 for only at the end
    # constants that the record hard-codes inline, lifted here unchanged
    vocab_size : int
    n_layer : int
    n_head : int
    n_embd : int
    logit_softcap : int
    attn_window : int # sliding-window length in tokens when flags.long_context (record #12: 1024)
    embed_vocab_size : int # rows of the token and value embedding tables (record #17: 50257, else vocab_size)
    dropped_attn_layer : int # flags.drop_attn_layer: index of the layer without attention (record #17: 7)
    ve_sparse_tables : int # flags.ve_sparsify: number of value-embedding tables (record #17: 3)
    lr_embed : float
    lr_head : float
    lr_muon : float
    lr_scalar : float
    adam_betas : list
    muon_momentum : float
    muon_momentum_warmup_start : float
    muon_momentum_warmup_end : float
    muon_momentum_warmup_steps : int
    window_warmup_start : int # flags.window_warmup: window at step 0, in tokens (record #13: 64)
    window_warmup_end : int # window at the last step, in tokens (record #13: 1792)
    window_warmup_quantum : int # window is a multiple of this many tokens (record #13: 64)
    window_block_formula : str # flags.block_window: block-count expression 'r16' (records #16-#17) | 'r18' (record #18)

def load_config(path):
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            import yaml # only needed when a .yaml config is passed directly; launch/run_one.py passes resolved JSON
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
    hp = cfg['hparams']
    names = {f.name for f in fields(Hyperparameters)}
    missing, unknown = names - set(hp), set(hp) - names
    assert not missing and not unknown, f"config hparams mismatch: missing={sorted(missing)} unknown={sorted(unknown)}"
    return cfg, Hyperparameters(**hp)

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='preset or resolved config (configs/rNN.yaml or resolved .json)')
parser.add_argument('--seed', type=int, required=True, help='seeds python, numpy, torch CPU and all CUDA generators')
cli = parser.parse_args()
run_config, args = load_config(cli.config)
flags = load_flags(run_config)

# seeding (infra; see CHANGES/infra.md for what this does and does not control). Every rank uses the same
# seed, so every rank builds the same initial weights before DDP's rank-0 broadcast.
import random
random.seed(cli.seed)
np.random.seed(cli.seed)
torch.manual_seed(cli.seed)
torch.cuda.manual_seed_all(cli.seed)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables (infra: unified batch convention, see CHANGES/infra.md)
assert args.micro_batch_tokens == 65536, "micro-batch is 65,536 tokens per device in every record; never reduce it"
if flags.long_context: # rec-12 layout: one row holding the whole micro-batch
    B, T = 1, args.micro_batch_tokens
else: # record #11 layout: rows of sequence_length tokens
    T = args.sequence_length
    assert args.micro_batch_tokens % T == 0
    B = args.micro_batch_tokens // T
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.global_batch_tokens % (args.micro_batch_tokens * ddp_world_size) == 0, "world_size must divide global/micro"
train_accumulation_steps = args.global_batch_tokens // (args.micro_batch_tokens * ddp_world_size)
vranks = args.global_batch_tokens // args.micro_batch_tokens # micro-batches per step = ranks of the 8-GPU record run
assert train_accumulation_steps * ddp_world_size == vranks

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, vranks,
                                     range(ddp_rank * train_accumulation_steps, (ddp_rank + 1) * train_accumulation_steps),
                                     args.loader_shard_rule, args.loader_replay_first_round, rows_2d=not flags.long_context,
                                     torch_io=flags.code_opts_15)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_world_size, [ddp_rank], args.loader_shard_rule,
                                   rows_2d=not flags.long_context, torch_io=flags.code_opts_15)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
if not flags.code_opts_18: # rec-18 fetches each step's batch at the start of the step instead
    x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = args.vocab_size
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      logit_softcap=args.logit_softcap, attn_window=args.attn_window,
                      embed_vocab_size=args.embed_vocab_size, dropped_attn_layer=args.dropped_attn_layer,
                      ve_sparse_tables=args.ve_sparse_tables))
if flags.fp32_small_params: # rec-18 (bf16_embeds = True)
    model = model.cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
else:
    model = model.cuda().bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()

if hasattr(config, "coordinate_descent_tuning") and not flags.code_opts_18:
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
if flags.code_opts_16:
    model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
else:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(True)
enable_flash_sdp(False)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)

# init the optimizer(s)
adam_betas = tuple(args.adam_betas)
if flags.ve_split:
    embed_params = [raw_model.transformer.wte.weight, *raw_model.value_embeds.parameters()] # rec-16
elif flags.value_embeds:
    embed_params = [raw_model.transformer.wte.weight, raw_model.transformer.vte.weight] # rec-14
else:
    embed_params = [raw_model.transformer.wte.weight]
params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2]+[raw_model.skip_weights]
if flags.code_opts_18:
    # rec-18: one Adam with three param groups, then Muon
    optimizer1 = torch.optim.Adam([dict(params=embed_params, lr=args.lr_embed),
                                   dict(params=[raw_model.lm_head.weight], lr=args.lr_head),
                                   dict(params=scalar_params, lr=args.lr_scalar)],
                                  betas=adam_betas, fused=True)
    muon_optimizer = Muon(matrix_params, lr=args.lr_muon, momentum=args.muon_momentum)
    optimizers = [optimizer1, muon_optimizer]
else:
    optimizer1 = torch.optim.Adam(embed_params, lr=args.lr_embed, betas=adam_betas, fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=args.lr_head,  betas=adam_betas, fused=True)
    optimizer3 = Muon(matrix_params, lr=args.lr_muon, momentum=args.muon_momentum)
    optimizer4 = torch.optim.Adam(scalar_params, lr=args.lr_scalar, betas=adam_betas, fused=True) # note that this learning rate is neither sensitive nor tuned
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
    muon_optimizer = optimizer3
# infra: every parameter is optimised by exactly one optimizer (fails at start-up instead of silently skipping one)
_opt_params = [p for opt in optimizers for g in opt.param_groups for p in g['params']]
assert len(_opt_params) == len({id(p) for p in _opt_params}), "a parameter is in two optimizers"
assert {id(p) for p in _opt_params} == {id(p) for p in raw_model.parameters()}, "a parameter is in no optimizer"

# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    if args.lr_schedule_form == 'frac': # rec-18: stable then decay, cooldown as a fraction of training
        t = 1 - it / args.num_iterations # time remaining in training
        assert 1 >= t > 0
        # 1) constant lr for first part of training
        if t >= args.cooldown_frac:
            return 1.0
        # 2) then linear cooldown
        else:
            return t / args.cooldown_frac
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
if flags.window_warmup and (flags.code_opts_15 or flags.code_opts_18):
    # rec-15: one persistent window tensor, updated in place only when the window changes (rec-18: every step)
    if flags.block_window: # rec-16: the window is a block count; its step-0 value (1 for 64 -> 1792, q 64)
        sw_size_prev = int((args.window_warmup_start + args.window_warmup_quantum) // 128)
    else:
        sw_size_prev = args.window_warmup_start
    sliding_window_size = torch.tensor(sw_size_prev, dtype=torch.int32, device="cuda")
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    attn_blocksize = None
    if flags.window_warmup:
        # Set the attention blocksize for the current step, in chunks of window_warmup_quantum (rec-13, @fernbear.bsky.social)
        q, w0, w1 = args.window_warmup_quantum, args.window_warmup_start, args.window_warmup_end
        if flags.block_window and args.window_block_formula == 'r18':
            # rec-18: linear increase over training in chunks of 128 blocks ((1-x)*128 + x*1856 for 64 -> 1792, q 64)
            x_progress = step / args.num_iterations # training progress
            assert 0 <= x_progress <= 1
            sw_size = int(((1 - x_progress) * (w0 + q) + x_progress * (w1 + q)) // 128)
        elif flags.block_window:
            # rec-16: linearly increase the window over training in chunks of q from w0 -> w1, counted in 128-token blocks
            frac_done = step / args.num_iterations # training progress
            sw_size = int(((1 - frac_done) * w0 + frac_done * w1 + q) // 128)
        else:
            sw_size = q * int((w0 + (w1 - w0) * step / args.num_iterations) // q)
        if flags.code_opts_18: # rec-18: copied every step
            sliding_window_size.copy_(sw_size)
            attn_blocksize = sliding_window_size
        elif flags.code_opts_15:
            if sw_size != sw_size_prev:
                sliding_window_size.copy_(sw_size, non_blocking=True)
                sw_size_prev = sw_size
            attn_blocksize = sliding_window_size
        elif flags.block_window:
            attn_blocksize = torch.tensor(sw_size, dtype=torch.int32, device='cuda')
        else:
            attn_blocksize = torch.tensor(q*((step/args.num_iterations * (w1 - w0) + w0)//q), dtype=torch.int, device='cuda')
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val, attn_blocksize=attn_blocksize)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    if flags.code_opts_18:
        # rec-18: the step's micro-batches fetched first; every micro-step backward all-reduces (no no_sync)
        micro_batches = [train_loader.next_batch() for _ in range(train_accumulation_steps)]
        for x, y in micro_batches:
            loss = model(x, y, attn_blocksize=attn_blocksize)
            loss.backward()
            train_loss = loss.detach()
            del loss
    for i in range(1, train_accumulation_steps+1):
        if flags.code_opts_18:
            break
        if flags.code_opts_15:
            # rec-15: forward and backward, then fetch the next micro-batch (same data, different overlap)
            with contextlib.ExitStack() as stack:
                if i < train_accumulation_steps: # there's no need to sync gradients every accumulation step
                    stack.enter_context(model.no_sync())
                if flags.code_opts_16 and not flags.code_opts_18 and step >= 5: # rec-16, rec-17
                    stack.enter_context(torch.compiler.set_stance(skip_guard_eval_unsafe=True))
                loss = model(x, y, attn_blocksize=attn_blocksize)
                loss.backward()
                train_loss = loss.detach()
                del loss
                x, y = train_loader.next_batch()
            continue
        # forward pass
        loss = model(x, y, attn_blocksize=attn_blocksize)
        train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    if not ((flags.code_opts_16 or flags.code_opts_18) and train_accumulation_steps == 1): # rec-16+ skip dividing by 1 (exact either way)
        for p in model.parameters():
            p.grad /= train_accumulation_steps
    # momentum warmup for Muon
    frac = min(step/args.muon_momentum_warmup_steps, 1)
    for group in muon_optimizer.param_groups: # one group, or one per parameter size with code_opts_15
        group['momentum'] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum_warmup_end
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        if not (args.lr_schedule_form == 'frac' and step == args.num_iterations - 1): # rec-18 skips the final sched.step
            sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
