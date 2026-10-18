"""Tests for the static-gate tooling on synthetic inputs (pure Python; no torch)."""
import os, subprocess, sys, textwrap
import pytest, yaml
TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
from diffparse import hunks
import param_inventory as pinv
import check_ledger as cl

DIFF = textwrap.dedent('''\
    --- a/train.py
    +++ b/train.py
    @@ -1,3 +1,3 @@
     a
    -b
    +B
     c
    @@ -10 +10,2 @@
     x
    +y
    --- a/README.md
    +++ /dev/null
    @@ -1,2 +0,0 @@
    -doc
    -doc
    Binary files a/img/x.png and b/img/x.png differ
    ''')

def test_diffparse_counts_and_headers():
    hs = hunks(DIFF)
    assert [(h['file'], h['header']) for h in hs] == [
        ('train.py', '@@ -1,3 +1,3 @@'), ('train.py', '@@ -10 +10,2 @@'),
        ('README.md', '@@ -1,2 +0,0 @@'), ('img/x.png', 'Binary files differ')]
    assert hs[1]['old'] == [10, 1] and hs[1]['new'] == [10, 2]

def _write_case(tmp_path, entries, trainer_src):
    (tmp_path / 'diffs').mkdir(); (tmp_path / 'ledger').mkdir()
    (tmp_path / 'diffs' / 'd.diff').write_text(DIFF)
    (tmp_path / 'ledger' / 'r99.yaml').write_text(yaml.safe_dump(dict(record=99, diff='diffs/d.diff', hunks=entries)))
    (tmp_path / 'trainer.py').write_text(trainer_src)
    return str(tmp_path / 'ledger' / 'r99.yaml'), str(tmp_path / 'trainer.py')

TRAINER = 'FLAGS = {"f1": False}\nclass Hyperparameters:\n    lr: float\n'

def _entry(i, file, header, **kw):
    e = dict(id=f'H{i}', file=file, header=header, description='d', category='cosmetic', flag='none',
             flag_reason='r', provenance='direct', bundled=False, numerics='identical')
    e.update(kw); return e

def test_check_ledger_complete_passes(tmp_path):
    hs = hunks(DIFF)
    entries = [_entry(i, h['file'], h['header']) for i, h in enumerate(hs)]
    entries[0].update(flag='f1'); entries[1].update(flag='config:lr')
    led, tr = _write_case(tmp_path, entries, TRAINER)
    errs, nh, ne = cl.check(led, tr)
    assert errs == [] and nh == ne == 4

def test_check_ledger_missing_hunk_fails(tmp_path):
    hs = hunks(DIFF)
    entries = [_entry(i, h['file'], h['header']) for i, h in enumerate(hs)][:-1]
    led, tr = _write_case(tmp_path, entries, TRAINER)
    errs, _, _ = cl.check(led, tr)
    assert any('unclassified hunk' in e for e in errs)

def test_check_ledger_unknown_flag_and_todo_fail(tmp_path):
    hs = hunks(DIFF)
    entries = [_entry(i, h['file'], h['header']) for i, h in enumerate(hs)]
    entries[0].update(flag='nope'); entries[1].update(description='TODO'); entries[2].update(flag_reason='')
    led, tr = _write_case(tmp_path, entries, TRAINER)
    errs = cl.check(led, tr)[0]
    assert any('nope' in e for e in errs) and any('TODO' in e for e in errs) and any('without flag_reason' in e for e in errs)

SCRIPT = textwrap.dedent('''\
    import torch
    from torch import nn
    from dataclasses import dataclass
    class CastedLinear(nn.Linear):
        def __init__(self, i, o):
            super().__init__(i, o, bias=False)
    class Block(nn.Module):
        def __init__(self, d, use_attn=True):
            super().__init__()
            self.attn = CastedLinear(d, d) if use_attn else None
            self.mlp = nn.Linear(d, 4 * d)
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
            self.register_buffer('inv', torch.ones(d // 2))
    @dataclass
    class Cfg:
        vocab: int = 10
        d: int = 4
        def nxt(self, n):
            return next(x for x in range(self.vocab + n)[::n] if x >= self.vocab)
    class GPT(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.wte = nn.Embedding(cfg.vocab, cfg.d)
            self.blocks = nn.ModuleList([Block(cfg.d, use_attn=(i != 1)) for i in range(3)])
            self.head = CastedLinear(cfg.d, cfg.nxt(8))
            self.skip = nn.Parameter(torch.ones(3 // 2))
    model = GPT(Cfg(d=4))
    ''')

def test_param_inventory_on_synthetic_script(tmp_path):
    p = tmp_path / 's.py'; p.write_text(SCRIPT)
    params, buffers = pinv.inventory(str(p))
    d = dict(params)
    assert d['wte.weight'] == (10, 4)
    assert d['head.weight'] == (16, 4)                 # next multiple of 8 >= 10
    assert 'blocks.1.attn.weight' not in d and d['blocks.0.attn.weight'] == (4, 4)
    assert d['blocks.2.mlp.weight'] == (16, 4) and d['blocks.2.mlp.bias'] == (16,)
    assert d['blocks.0.lambdas'] == (2,) and d['skip'] == (1,)
    assert dict(buffers)['blocks.0.inv'] == (2,)

def test_loader_equivalence_negative_control(tmp_path):
    """The harness must report a difference when the replay flag is wrong."""
    root = os.path.dirname(TOOLS)
    snap = os.path.join(root, 'records', 'r11', 'log_source', 'train_gpt2.py')
    if not os.path.exists(snap):
        pytest.skip('records/r11 not exported')
    cfg = yaml.safe_load(open(os.path.join(root, 'configs', 'r11.yaml')))
    cfg['hparams']['loader_replay_first_round'] = False
    bad = tmp_path / 'bad.yaml'; bad.write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, '-I', '-W', 'ignore', os.path.join(TOOLS, 'loader_equivalence.py'), snap,
                        os.path.join(root, 'train_flagged.py'), str(bad), '--protocol', 'r11', '--steps', '50'],
                       capture_output=True, text=True)
    assert r.returncode == 1 and 'differ from the 8-GPU record stream: 1' in r.stdout

def test_param_inventory_plain_tensor_attribute_is_not_a_parameter(tmp_path):
    src = textwrap.dedent('''\
        import torch
        from torch import nn
        def helper(d):
            return torch.cat([torch.linspace(0.0, 1.0, steps=d // 4), torch.zeros(d // 4)])
        class R(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))   # plain attribute
                t = torch.arange(16)
                th = torch.einsum("i, j -> ij", t, helper(d))
                self.cos = nn.Buffer(th.cos(), persistent=False)
                self.w = nn.Parameter(torch.ones(3))
        class GPT(nn.Module):
            def __init__(self):
                super().__init__()
                self.r = R(8)
        model = GPT()
        ''')
    p = tmp_path / 's.py'; p.write_text(src)
    params, buffers = pinv.inventory(str(p))
    assert params == [('r.w', (3,))]
    assert buffers == [('r.cos', (16, 4))]


def test_torch_api_check_detects_missing_param_and_symbol(tmp_path):
    import torch_api_check as t
    (tmp_path / 'torch_nn_attention_flex_attention.py').write_text(
        'def create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device="cuda"):\n    pass\n'   # no _compile
        'class BlockMask:\n    @classmethod\n    def from_kv_blocks(cls, kv_num_blocks, kv_indices, full_kv_num_blocks=None,'
        ' full_kv_indices=None, BLOCK_SIZE=128, mask_mod=None):\n        pass\n')            # no flex_attention
    res = {name: status for name, status, _ in t.check(str(tmp_path))}
    assert 'missing params' in res['create_block_mask'] and "_compile" in res['create_block_mask']
    assert res['flex_attention'] == 'ABSENT'
    assert res['BlockMask.from_kv_blocks'] == 'ok'
    assert res['set_stance'] == 'MISSING FILE'
