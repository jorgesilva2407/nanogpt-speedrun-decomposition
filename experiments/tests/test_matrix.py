"""Tests for experiments/matrix.py (pure Python; reads presets, arms.yaml, study.yaml and the trainer via ast)."""
import ast, copy, hashlib, json, os, sys
import pytest, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import matrix as M

def build(**study_over):
    st = yaml.safe_load(open(os.path.join(ROOT, 'experiments/study.yaml')))
    st.update(study_over)
    m = M.Matrix(st, yaml.safe_load(open(os.path.join(ROOT, 'experiments/arms.yaml')))['arms'],
                 yaml.safe_load(open(os.path.join(ROOT, 'configs/r11.yaml'))),
                 yaml.safe_load(open(os.path.join(ROOT, 'configs/r18.yaml'))), M.trainer_registries())
    m.build()
    return m

def by_id(m):
    return {c['exp_id']: c for c in m.configs}

def test_config_hash_matches_trainer():
    tree = ast.parse(open(os.path.join(ROOT, 'train_flagged.py')).read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'config_hash')
    ns = {'json': json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'trainer', 'exec'), ns)
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs/r18.yaml')))
    assert ns['config_hash'](cfg) == M.config_hash(cfg)

def test_fixed_points_equal_presets():
    c = by_id(build())
    for pid, p in (('fp_r11', 'configs/r11.yaml'), ('fp_r18', 'configs/r18.yaml')):
        pre = yaml.safe_load(open(os.path.join(ROOT, p)))
        assert c[pid]['flags'] == pre['flags'] and c[pid]['hparams'] == pre['hparams']

def test_every_config_satisfies_prerequisites():
    m = build()
    for c in m.configs:
        assert M.violations(c['flags'], m.REQUIRES) == [], c['exp_id']

def test_undefined_loo_set_and_no_config_emitted():
    m = build()
    assert sorted(o['arm'] for o in m.open_items) == ['A1a', 'A1b', 'A3', 'A4', 'A5']
    ids = by_id(m)
    assert not any(f'loo_{a}' in ids for a in ('A1a', 'A1b', 'A3', 'A4', 'A5'))
    assert {'loo_A2', 'loo_A4b', 'loo_A5b', 'loo_A6', 'loo_A6b', 'loo_A7'} <= set(ids)

def test_aoi_ladder_is_cumulative_and_keeps_r11_hparams():
    c = by_id(build())
    a6 = c['aoi_A6']
    assert all(a6['flags'][f] for f in ('value_embeds', 've_unet', 've_split', 've_sparsify', 'rope_half_truncate'))
    added = {p['flag'] for p in a6['provenance']['prerequisites_added']}
    assert added == {'ve_split', 've_unet', 'value_embeds'}
    assert a6['hparams'] == yaml.safe_load(open(os.path.join(ROOT, 'configs/r11.yaml')))['hparams']

def test_loo_keeps_r18_hparams_and_softcap_arm_is_config():
    c = by_id(build())
    loo = c['loo_A7']
    r18 = yaml.safe_load(open(os.path.join(ROOT, 'configs/r18.yaml')))
    assert loo['hparams']['logit_softcap'] == 30 and loo['hparams']['num_iterations'] == r18['hparams']['num_iterations']
    assert loo['flags'] == r18['flags']
    assert c['aoi_A7']['hparams']['logit_softcap'] == 15 and c['aoi_A7']['hparams']['num_iterations'] == 3000

def test_seed_counts():
    c = by_id(build())
    assert len(c['fp_r18']['seeds']) == 5 and len(c['aoi_A3']['seeds']) == 5 and len(c['ctrl_A8_r18']['seeds']) == 3

def test_rq3b_scaling_rescales_absolute_step_hparams():
    m = build(rq3b=dict(budget_scales=[0.5], resolved_arms=['A7'], absolute_step_hparams=['warmdown_iters', 'warmup_iters', 'muon_momentum_warmup_steps']))
    c = by_id(m)
    r11 = c['rq3b_s0.5_aoi_A7']
    assert r11['hparams']['num_iterations'] == 1500 and r11['hparams']['warmdown_iters'] == 450
    assert r11['hparams']['muon_momentum_warmup_steps'] == 250
    r18 = c['rq3b_s0.5_loo_A7']
    assert r18['hparams']['num_iterations'] == 695 and r18['hparams']['warmdown_iters'] == 600   # frac schedule: unused, untouched
    assert 'warmdown_iters' not in r18['provenance']['scaled_hparams']

def test_validation_mode_subset20():
    c = by_id(build(validation=dict(mode='subset20')))
    assert c['fp_r11']['measurement']['val_every_final_third'] == 5
    assert c['fp_r11']['measurement']['val_final_third_fraction'] == 0.2
