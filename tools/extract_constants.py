"""Static gate: the preset's hparams equal the constants written in the snapshot (AST + source patterns).

For each config key the extractor either finds the snapshot's value (literal, Hyperparameters default, keyword
argument, or a pattern in the source) or reports it as NOT EXTRACTABLE. Mechanism detectors check that the
preset's flags agree with what the snapshot code does. Nothing is imported or executed.

Usage: python -I tools/extract_constants.py <snapshot.py> <configs/rNN.yaml>   (exit 1 on any mismatch)
"""
import ast, re, sys
import yaml

def lit(node, env=None):
    env = env or {}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        l, r = lit(node.left, env), lit(node.right, env)
        return {ast.Mult: l * r if None not in (l, r) else None, ast.Add: l + r if None not in (l, r) else None,
                ast.Sub: l - r if None not in (l, r) else None,
                ast.FloorDiv: l // r if None not in (l, r) else None}.get(type(node.op))
    if isinstance(node, (ast.Tuple, ast.List)):
        return [lit(e, env) for e in node.elts]
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = lit(node.operand, env)
        return -v if v is not None else None
    return None

class Snapshot:
    def __init__(self, path):
        self.src = open(path).read()
        self.tree = ast.parse(self.src)
        self.hp = {}
        for n in self.tree.body:
            if isinstance(n, ast.ClassDef) and n.name == 'Hyperparameters':
                for st in n.body:
                    if isinstance(st, ast.AnnAssign) and st.value is not None:
                        self.hp[st.target.id] = lit(st.value)
                    elif isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name):
                        self.hp[st.targets[0].id] = lit(st.value)
        self.consts = {}
        for n in self.tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                v = lit(n.value, self.consts)
                if v is not None:
                    self.consts[n.targets[0].id] = v
        self.calls = {}
        for n in self.tree.body:
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.targets[0], ast.Name):
                self.calls[n.targets[0].id] = n.value

    def kw(self, var, key):
        call = self.calls.get(var)
        if call is None:
            return None
        for k in call.keywords:
            if k.arg == key:
                return lit(k.value, self.consts)
        return None

    def gpt_config(self):
        """Model dims: GPTConfig defaults overridden by the model = GPT(...) call."""
        out = {}
        for n in self.tree.body:
            if isinstance(n, ast.ClassDef) and n.name == 'GPTConfig':
                for st in n.body:
                    if isinstance(st, ast.AnnAssign) and st.value is not None:
                        out[st.target.id] = lit(st.value)
        call = next((n.value for n in self.tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                     and ast.unparse(n.value.func) == 'GPT'), None)
        if call is not None:
            inner = call.args[0] if call.args and isinstance(call.args[0], ast.Call) else call
            for k in inner.keywords:
                out[k.arg] = lit(k.value, self.consts)
        alias = dict(n_layer='num_layers', n_head='num_heads', n_embd='model_dim')
        for a, b in alias.items():
            if b in out and a not in out:
                out[a] = out[b]
        return out

def find(src, pattern, group=1, cast=float):
    m = re.search(pattern, src)
    return cast(m.group(group)) if m else None

def extract(s: Snapshot):
    """Return dict key -> value (None = not extractable) for every key we know how to look for."""
    hp, src, g = s.hp, s.src, s.gpt_config()
    out = {}
    out['input_bin'] = hp.get('input_bin', hp.get('train_bin'))
    out['input_val_bin'] = hp.get('input_val_bin', hp.get('val_bin'))
    seq = hp.get('sequence_length')
    if 'device_batch_size' in hp:            # r11-r13: batch sizes in sequences
        out['global_batch_tokens'] = hp['batch_size'] * seq
        out['micro_batch_tokens'] = hp['device_batch_size'] * seq
    elif 'max_device_batch_size' in hp:      # r18: tokens
        out['global_batch_tokens'] = hp['batch_size']
        out['micro_batch_tokens'] = hp['max_device_batch_size']
    else:                                    # r14-r17: batch_size sequences of sequence_length, 1 per device
        out['global_batch_tokens'] = hp['batch_size'] * seq
        out['micro_batch_tokens'] = seq
    out['sequence_length'] = seq if 'device_batch_size' in hp and hp['device_batch_size'] > 1 else None
    out['num_iterations'] = hp.get('num_iterations')
    out['warmup_iters'] = hp.get('warmup_iters')
    out['warmdown_iters'] = hp.get('warmdown_iters', hp.get('cooldown_iters'))
    out['cooldown_frac'] = hp.get('cooldown_frac')
    out['lr_schedule_form'] = 'frac' if 'cooldown_frac' in hp else 'iters'
    out['weight_decay'] = hp.get('weight_decay')
    out['val_loss_every'] = hp.get('val_loss_every')
    out['val_tokens'] = hp.get('val_tokens')
    out['save_every'] = hp.get('save_every')
    out['vocab_size'] = s.consts.get('num_vocab', g.get('vocab_size'))
    out['embed_vocab_size'] = out['vocab_size']
    if 'vocab_size_next_multiple_of(128)' in src:   # record #17: embeddings 50257 rows, lm_head next multiple of 128
        out['embed_vocab_size'] = g.get('vocab_size')
        v = out['embed_vocab_size']
        out['vocab_size'] = next(x for x in range(v + 128)[::128] if x >= v)
    out['n_layer'], out['n_head'], out['n_embd'] = g.get('n_layer'), g.get('n_head'), g.get('n_embd')
    out['logit_softcap'] = find(src, r'logits = (\d+) \* torch\.tanh\(logits / \1\)', cast=int)
    if out['logit_softcap'] is None and 'self.lm_head_softcap * torch.tanh(logits / self.lm_head_softcap)' in src:
        out['logit_softcap'] = g.get('lm_head_softcap')   # record #15: GPTConfig field
    # optimizers: r11-r17 four optimizers; r18 one Adam with param groups + Muon
    if 'optimizer4' in s.calls:
        out['lr_embed'], out['lr_head'] = s.kw('optimizer1', 'lr'), s.kw('optimizer2', 'lr')
        out['lr_muon'], out['lr_scalar'] = s.kw('optimizer3', 'lr'), s.kw('optimizer4', 'lr')
        betas = {tuple(s.kw(o, 'betas')) for o in ('optimizer1', 'optimizer2', 'optimizer4')}
        out['adam_betas'] = list(betas.pop()) if len(betas) == 1 else None
        out['muon_momentum'] = s.kw('optimizer3', 'momentum')
    else:
        groups = {}
        call = s.calls.get('optimizer1')
        for d in call.args[0].elts:
            kws = {k.arg: k.value for k in d.keywords}
            groups[ast.unparse(kws['params'])] = lit(kws['lr'])
        out['lr_embed'], out['lr_head'], out['lr_scalar'] = groups.get('embed_params'), groups.get('head_params'), groups.get('scalar_params')
        out['lr_muon'] = s.kw('optimizer2', 'lr')
        out['adam_betas'] = s.kw('optimizer1', 'betas')
        out['muon_momentum'] = s.kw('optimizer2', 'momentum')
    out['muon_momentum_warmup_steps'] = find(src, r'frac = min\(step\s*/\s*(\d+), 1\)', cast=int)
    m = re.search(r"\(1 - frac\) \* ([\d.]+) \+ frac \* ([\d.]+)", src)
    out['muon_momentum_warmup_start'], out['muon_momentum_warmup_end'] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    # loader semantics
    if 'self.current_position + (B * T * self.num_processes + 1) > len(self.tokens)' in src:
        out['loader_shard_rule'] = 'gt_plus1'
    elif re.search(r'self\.current_position \+ batch_size >= len\(self\.tokens\)', src):
        out['loader_shard_rule'] = 'ge'
    elif re.search(r'self\.current_position \+ batch_size \+ 1 >= len\(self\.tokens\)', src):
        out['loader_shard_rule'] = 'ge_plus1' if 'def next_batch(self):' in src else 'global_ge_plus1'
    else:
        out['loader_shard_rule'] = None
    reset_before_loop = re.search(r'\ntrain_loader\.reset\(\)\n', src) is not None
    out['loader_replay_first_round'] = reset_before_loop
    out['attn_window'] = find(src, r'window_mask = q_idx - kv_idx < (\d+)\n', cast=int)
    out['dropped_attn_layer'] = find(src, r'(?:if layer_idx != |use_attn=\(i != )(\d+)', cast=int)
    out['ve_sparse_tables'] = None
    if 'None, None, None' in src:
        out['ve_sparse_tables'] = find(src, r'nn\.Embedding\([^)]*\)\s*for _ in range\((\d+)\)', cast=int)
    m = re.search(r'torch\.tensor\((\d+)\*\(\(step/args\.num_iterations \* \((\d+) - (\d+)\) \+ (\d+)\)//(\d+)\)', src)
    out['window_warmup_quantum'] = out['window_warmup_end'] = out['window_warmup_start'] = None
    if m:   # record #13: q*((step/N * (end - start) + start)//q)
        q1, end, s0, s1, q2 = map(int, m.groups())
        assert q1 == q2 and s0 == s1, 'unexpected window schedule shape'
        out['window_warmup_quantum'], out['window_warmup_end'], out['window_warmup_start'] = q1, end, s0
    m = re.search(r'sw_size =\s+(\d+) \* int\(\((\d+) \+ \((\d+) - (\d+)\) \* step / args\.num_iterations\) // (\d+)\)', src)
    if m:   # record #15: q * int((start + (end - start) * step / N) // q)
        q1, s0, end, s1, q2 = map(int, m.groups())
        assert q1 == q2 and s0 == s1, 'unexpected window schedule shape'
        out['window_warmup_quantum'], out['window_warmup_end'], out['window_warmup_start'] = q1, end, s0
    m = re.search(r'int\(\(\(1 - frac_done\) \* (\d+) \+ frac_done \* (\d+) \+ (\d+)\) // 128\)', src)
    if m:   # records #16-#17: blocks = int(((1 - f) * start + f * end + q) // 128)
        s0, end, q1 = map(int, m.groups())
        out['window_warmup_quantum'], out['window_warmup_end'], out['window_warmup_start'] = q1, end, s0
    out['window_block_formula'] = None
    if re.search(r'int\(\(\(1 - frac_done\) \* \d+ \+ frac_done \* \d+ \+ \d+\) // 128\)', src):
        out['window_block_formula'] = 'r16'
    m = re.search(r'int\(\(\(1 - x\) \* (\d+) \+ x \* (\d+)\) // 128\)', src)
    if m:   # record #18: blocks = int(((1 - x) * (start + q) + x * (end + q)) // 128); q not separable -> derived check
        out['window_block_formula'] = 'r18'
        out['_window_r18_endpoints'] = (int(m.group(1)), int(m.group(2)))
    return out

def detect_flags(s: Snapshot):
    """Mechanism detectors: what the snapshot code does, expressed as the flag values that reproduce it.
    A detector returns None when it cannot decide (reported, not a mismatch)."""
    src, hp = s.src, s.hp
    f = {}
    f['flex_kernel'] = 'flex_attention(q' in src
    seq = hp.get('sequence_length', hp.get('max_device_batch_size'))
    f['long_context'] = seq == 65536 and ('idx[None]' in src or 'inputs[None]' in src)
    f['doc_masking'] = 'docs[q_idx] == docs[kv_idx]' in src
    f['window_warmup'] = ('attn_blocksize' in src) or ('sliding_window' in src)
    if 'self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))' in src:
        f['rotary_variant'] = 'v11'
    elif 'self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))' in src:
        f['rotary_variant'] = 'v12'
    elif "self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))" in src:
        f['rotary_variant'] = 'v14'
    else:
        f['rotary_variant'] = None
    f['value_embeds'] = ('vte = nn.Embedding' in src) or ('self.value_embeds' in src)
    f['ve_unet'] = ('vi[self.num_encoder_layers-1-i]' in src) or ('ve += reversed(ve)' in src) or ('ve[0], ve[1], ve[2]' in src)
    f['ve_lambdas2'] = 'self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))' in src
    f['code_opts_15'] = 'dist.all_gather(update_buffers' in src
    f['muon_ema_momentum'] = 'buf.lerp_(g, 1 - momentum)' in src
    f['blockmask_manual'] = 'BlockMask.from_kv_blocks' in src
    f['ve_split'] = 'class ValueEmbedding' in src
    f['block_window'] = 'sliding_window_num_blocks' in src
    f['blockmask_separate'] = 'full_kv_num_blocks' in src
    f['code_opts_16'] = 'gradient_as_bucket_view=True' in src
    f['ve_sparsify'] = 'None, None, None' in src
    f['rope_half_truncate'] = '(1 / 1024) ** torch.linspace' in src
    f['drop_attn_layer'] = re.search(r'(?:if layer_idx != |use_attn=\(i != )\d+', src) is not None
    f['code_opts_17'] = 'expandable_segments:True' in src
    f['fp32_small_params'] = 'bf16_embeds' in src
    f['code_opts_18'] = "torch.empty(1, device='cuda', requires_grad=True).backward()" in src
    if 'nn.Buffer(theta.cos()' in src:
        f['rotary_variant'] = 'v17'
    return f

def main():
    snap_path, preset_path = sys.argv[1:3]
    s = Snapshot(snap_path)
    vals = extract(s)
    cfg = yaml.safe_load(open(preset_path))
    hp = cfg['hparams']
    bad = 0
    rows = []
    for k in sorted(hp):
        if k not in vals:
            rows.append((k, '-', hp[k], 'NOT EXTRACTABLE (no extractor)'))
            continue
        v = vals[k]
        if v is None:
            rows.append((k, '-', hp[k], 'NOT EXTRACTABLE (absent in snapshot)'))
            continue
        ok = (v == hp[k]) or (isinstance(v, float) and isinstance(hp[k], (int, float)) and abs(v - hp[k]) < 1e-12)
        if isinstance(v, (list, tuple)) and isinstance(hp[k], (list, tuple)):
            ok = list(v) == list(hp[k])
        rows.append((k, v, hp[k], 'match' if ok else 'MISMATCH'))
        bad += not ok
    if '_window_r18_endpoints' in vals:
        want = vals.pop('_window_r18_endpoints')
        got = (hp['window_warmup_start'] + hp['window_warmup_quantum'], hp['window_warmup_end'] + hp['window_warmup_quantum'])
        ok = tuple(want) == got
        rows.append(('derived:start+q, end+q', want, got, 'match' if ok else 'MISMATCH'))
        bad += not ok
        for k in ('window_warmup_start', 'window_warmup_end', 'window_warmup_quantum'):
            rows[:] = [r if r[0] != k else (k, '-', hp[k], 'checked only as start+q / end+q (record #18 form)') for r in rows]
    for k in sorted(set(vals) - set(hp)):
        if vals[k] is not None:
            rows.append((k, vals[k], '-', 'in snapshot, not a config key of this preset'))
    det = detect_flags(s)
    for k in sorted(cfg.get('flags') or {}):
        if k not in det:
            rows.append(('flag:' + k, '-', cfg['flags'][k], 'NO DETECTOR'))
        elif det[k] is None:
            rows.append(('flag:' + k, '-', cfg['flags'][k], 'UNDECIDED by detector'))
        else:
            ok = det[k] == cfg['flags'][k]
            rows.append(('flag:' + k, det[k], cfg['flags'][k], 'match' if ok else 'MISMATCH'))
            bad += not ok
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f'{r[0]:{w}s}  snapshot={str(r[1]):38s} preset={str(r[2]):38s} {r[3]}')
    print(f'{preset_path}: {bad} mismatches')
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
