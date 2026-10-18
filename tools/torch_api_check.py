"""Static check that every torch API train_flagged.py calls exists, with the arguments it passes, in a torch source
snapshot (files saved by explore_tools/fetch_sources.sh). Reads source with ast; never imports torch.
Usage: python -I tools/torch_api_check.py ext/torch_src/v2.6.0 [ext/torch_src]   (several dirs: one column each)
"""
import ast, os, sys

# (file in the snapshot, kind, dotted name, required parameter names, where the trainer uses it)
CHECKS = [
    ('torch_nn_attention_flex_attention.py', 'func', 'create_block_mask', ['mask_mod', 'B', 'H', 'Q_LEN', 'KV_LEN', 'device', '_compile'], 'block masks #12-#18'),
    ('torch_nn_attention_flex_attention.py', 'func', 'flex_attention', ['query', 'key', 'value', 'block_mask', 'enable_gqa'], 'attention #12-#18 (enable_gqa: #16)'),
    ('torch_nn_attention_flex_attention.py', 'func', 'BlockMask.from_kv_blocks', ['kv_num_blocks', 'kv_indices', 'full_kv_num_blocks', 'full_kv_indices', 'BLOCK_SIZE', 'mask_mod'], 'blockmask_manual / blockmask_separate'),
    ('torch_compiler___init__.py', 'func', 'set_stance', ['skip_guard_eval_unsafe'], 'code_opts_16 (#16-#17)'),
    ('torch_nn_parameter.py', 'class', 'Buffer', [], 'rotary_variant v17 (#17-#18)'),
    ('torch_nn___init__.py', 'import', 'Buffer', [], 'nn.Buffer'),
    ('torch__inductor_config.py', 'assign', 'coordinate_descent_tuning', [], 'all presets'),
    ('torch_backends_cuda___init__.py', 'func', 'enable_cudnn_sdp', ['enabled'], '#11 dense path'),
    ('torch_backends_cuda___init__.py', 'func', 'enable_flash_sdp', ['enabled'], '#11 dense path'),
    ('torch_backends_cuda___init__.py', 'func', 'enable_math_sdp', ['enabled'], '#11 dense path'),
    ('torch_backends_cuda___init__.py', 'func', 'enable_mem_efficient_sdp', ['enabled'], '#11 dense path'),
    ('torch_backends_cudnn___init__.py', 'func', 'version', [], 'meta.json'),
]

def find(tree, kind, dotted):
    parts = dotted.split('.')
    scope = tree.body
    for i, name in enumerate(parts):
        last = i == len(parts) - 1
        hit = None
        for node in scope:
            if kind == 'import' and last and isinstance(node, (ast.Import, ast.ImportFrom)):
                if any((a.asname or a.name).split('.')[-1] == name for a in node.names):
                    return node
            if kind == 'assign' and last and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    return node
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                hit = node
        if hit is None:
            return None
        if last:
            return hit
        scope = hit.body
    return None

def params(fn):
    a = fn.args
    return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]

def check(src_dir):
    out = []
    for fname, kind, dotted, need, use in CHECKS:
        path = os.path.join(src_dir, fname)
        if not os.path.exists(path):
            out.append((dotted, 'MISSING FILE', use)); continue
        node = find(ast.parse(open(path).read()), kind, dotted)
        if node is None:
            out.append((dotted, 'ABSENT', use)); continue
        if kind == 'func':
            missing = [p for p in need if p not in params(node)]
            out.append((dotted, 'ok' if not missing else f'missing params {missing}', use))
        else:
            out.append((dotted, 'ok', use))
    return out

def main():
    dirs = sys.argv[1:] or ['ext/torch_src/v2.6.0']
    cols = [check(d) for d in dirs]
    print('| API | ' + ' | '.join(dirs) + ' | used by |')
    print('|---|' + '---|' * len(dirs) + '---|')
    for i, (fname, kind, dotted, need, use) in enumerate(CHECKS):
        print(f'| `{dotted}` | ' + ' | '.join(c[i][1] for c in cols) + f' | {use} |')
    bad = sum(c[i][1] != 'ok' for c in cols for i in range(len(CHECKS)))
    print(f'\n{bad} problems')
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
