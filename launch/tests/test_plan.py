"""Host-rule tests for launch/plan.py on synthetic state (pure Python)."""
import json, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'launch'))
import plan

MATRIX = dict(configurations=[
    dict(exp_id='fp_r11', family='fp', arm=None, seeds=[0, 1], hparams_from='r11'),
    dict(exp_id='fp_r18', family='fp', arm=None, seeds=[0, 1], hparams_from='r18'),
    dict(exp_id='aoi_A3', family='aoi', arm='A3', seeds=[0, 1], hparams_from='r11'),
    dict(exp_id='loo_A7', family='loo', arm='A7', seeds=[0, 1], hparams_from='r18'),
    dict(exp_id='aoi_A7', family='aoi', arm='A7', seeds=[0, 1], hparams_from='r11'),
])

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, 'HOSTS', str(tmp_path / 'hosts.json'))
    monkeypatch.setattr(plan, 'RESULTS', str(tmp_path / 'results'))
    monkeypatch.setattr(plan, 'matrix', lambda: MATRIX)
    return tmp_path

def test_first_host_claims_arm_and_plain_fixed_points(env):
    errors, jobs = plan.plan('lab', ['A7'], claim=True)
    assert errors == []
    names = [n for _, n, _ in jobs]
    assert names.count('fp_r11') == 2 and names.count('fp_r18') == 2 and names.count('loo_A7') == 2
    h = json.load(open(env / 'hosts.json'))
    assert h['arms'] == {'A7': 'lab'} and h['fixed_points'] == {'fp_r11': 'lab', 'fp_r18': 'lab'}

def test_second_host_refused_for_same_arm(env):
    plan.plan('lab', ['A7'], claim=True)
    errors, jobs = plan.plan('cloud', ['A7'], claim=True)
    assert errors and 'bound to host lab' in errors[0] and jobs == []

def test_other_arm_on_second_host_gets_baseline_replica(env):
    plan.plan('lab', ['A7'], claim=True)
    errors, jobs = plan.plan('cloud', ['A3'], claim=True)
    assert errors == []
    names = {n for _, n, _ in jobs}
    assert 'fp_r11__on_cloud' in names and 'fp_r11' not in names

def test_seed_already_run_elsewhere_is_refused(env):
    d = env / 'results' / 'aoi_A3' / 'seed1'
    d.mkdir(parents=True)
    (d / 'meta.json').write_text(json.dumps(dict(host_label='lab')))
    errors, _ = plan.plan('cloud', ['A3'])
    assert errors and 'already ran on lab' in errors[0]
