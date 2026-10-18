"""Print the arm -> flag -> dependency table.

Reads experiments/arms.yaml and the trainer's REQUIRES / FLAGS registries (via ast; nothing imported) and prints,
for each arm, its switches, their direct and transitive prerequisites, and which prerequisites belong to other
arms (those make add-one-in cumulative and leave-one-out ill-defined for the prerequisite arm).
Usage: python -I tools/arm_table.py [--arms experiments/arms.yaml] [--trainer train_flagged.py] [--markdown]
"""
import argparse, ast, sys
import yaml

def registries(path):
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('FLAGS', 'REQUIRES', 'FLAG_CHOICES'):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out['FLAGS'], out['REQUIRES']

def closure(flag, requires):
    seen, stack = [], list(requires.get(flag, []))
    while stack:
        f = stack.pop(0)
        if f not in seen:
            seen.append(f)
            stack += requires.get(f, [])
    return seen

def table(arms_path, trainer):
    flags, requires = registries(trainer)
    arms = yaml.safe_load(open(arms_path))['arms']
    owner = {}
    for a in arms:
        for f in (a.get('flags') or {}):
            owner[f] = a['id']
    rows = []
    for a in arms:
        sw = list(a.get('flags') or {}) + [f'config:{k}' for k in (a.get('config') or {})]
        direct, trans, ext = [], [], []
        for f in a.get('flags') or {}:
            assert f in flags, f'{f} not a trainer flag'
            direct += [r for r in requires.get(f, []) if r not in direct]
            trans += [r for r in closure(f, requires) if r not in trans]
        ext = sorted({owner[r] for r in trans if owner.get(r) not in (None, a['id'])})
        rows.append((a['id'], a['name'], a.get('record'), sw, [r for r in direct if r not in sw],
                     [r for r in trans if r not in sw], ext, a.get('ve_rung')))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='experiments/arms.yaml'); ap.add_argument('--trainer', default='train_flagged.py')
    ap.add_argument('--markdown', action='store_true')
    a = ap.parse_args()
    rows = table(a.arms, a.trainer)
    if a.markdown:
        print('| arm | name | record | switches | requires (direct) | requires (transitive) | prerequisite arms |')
        print('|---|---|---|---|---|---|---|')
        for r in rows:
            print(f"| {r[0]} | {r[1]} | #{r[2]} | {', '.join(r[3])} | {', '.join(r[4]) or '-'} | {', '.join(r[5]) or '-'} | {', '.join(r[6]) or '-'} |")
    else:
        for r in rows:
            print(f"{r[0]:5s} #{r[2]:<3} {r[1][:52]:52s} switches={r[3]}\n      requires={r[5] or '-'}  prerequisite_arms={r[6] or '-'}")

if __name__ == '__main__':
    main()
