"""Static parameter inventory: names and shapes of every nn.Parameter a training script would create.

An abstract interpreter over the script's AST. It evaluates only model *construction*: the module-level
`model = GPT(...)` call and, recursively, the `__init__` of every class it instantiates. Supported:
nn.Embedding, nn.Linear (and subclasses), nn.Parameter(torch.ones/zeros/tensor(...)), nn.ModuleList,
nn.ModuleDict, register_buffer / nn.Buffer (recorded as buffers), dataclass configs with methods, `if`,
conditional expressions, comprehensions, simple local arithmetic. Anything else evaluates to an opaque value
and is ignored unless assigned to `self` as a parameter (then it is an error). No code is executed with
Python's interpreter and torch is never imported.

For train_flagged.py the module-level names `args` (hparams) and `flags` come from a preset config.

Usage:
  python -I tools/param_inventory.py <script.py> [--preset configs/rNN.yaml] [--json out.json]
"""
import argparse, ast, json, math, sys
from collections import OrderedDict

class Opaque:
    def __repr__(self):
        return '<opaque>'
OPAQUE = Opaque()

class Tensor:
    """A plain tensor value (shape only). Assigned to `self` it is an ordinary attribute, not a parameter."""
    def __init__(self, shape):
        self.shape = tuple(int(s) for s in shape)
    def numel(self):
        return math.prod(self.shape)

class Param(Tensor):
    """Created only by nn.Parameter(...) (or nn.Embedding/nn.Linear weights)."""

class Buffer(Tensor):
    """Created only by register_buffer(...) or nn.Buffer(...)."""

class Module:
    def __init__(self, cls):
        self.cls = cls
        self.children = OrderedDict()  # name -> Module | Param | Buffer | None
        self.attrs = {}                # plain attributes (ints, configs, ...)

    def named_params(self, prefix=''):
        for k, v in self.children.items():
            if isinstance(v, Buffer):
                continue
            if isinstance(v, Param):
                yield prefix + k, v
            elif isinstance(v, Module):
                yield from v.named_params(prefix + k + '.')

    def named_buffers(self, prefix=''):
        for k, v in self.children.items():
            if isinstance(v, Buffer):
                yield prefix + k, v
            elif isinstance(v, Module):
                yield from v.named_buffers(prefix + k + '.')

class Obj:
    """Dataclass/config/namespace instance."""
    def __init__(self, cls, fields, methods=None):
        self.cls = cls
        self.fields = dict(fields)
        self.methods = methods or {}

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value

class Interp:
    def __init__(self, tree, globals_):
        self.classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        self.funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.g = dict(globals_)
        self.notes = []

    # ------------------------------------------------------------------ helpers
    def base_names(self, cls_node):
        out = []
        for b in cls_node.bases:
            out.append(ast.unparse(b))
        return out

    def is_dataclass(self, cls_node):
        return any(ast.unparse(d) in ('dataclass', 'dataclasses.dataclass') for d in cls_node.decorator_list)

    def linear_like(self, name, seen=()):
        if name in ('nn.Linear', 'torch.nn.Linear'):
            return True
        node = self.classes.get(name)
        if node is None or name in seen:
            return False
        return any(self.linear_like(b, seen + (name,)) for b in self.base_names(node))

    # ------------------------------------------------------------------ instantiation
    def instantiate(self, name, args, kwargs):
        if name in ('nn.Embedding', 'torch.nn.Embedding'):
            m = Module('Embedding')
            m.children['weight'] = Param((args[0], args[1]))
            return m
        if name in ('nn.Linear', 'torch.nn.Linear'):
            return self.make_linear(Module('Linear'), args, kwargs)
        if name in ('nn.ModuleList', 'torch.nn.ModuleList'):
            m = Module('ModuleList')
            for i, c in enumerate(args[0] if args else []):
                m.children[str(i)] = c
            return m
        if name in ('nn.ModuleDict', 'torch.nn.ModuleDict'):
            m = Module('ModuleDict')
            for k, c in (args[0] if args else {}).items():
                m.children[k] = c
            return m
        if name in ('nn.Parameter', 'torch.nn.Parameter'):
            t = args[0]
            if isinstance(t, Tensor):
                return Param(t.shape)
            raise ValueError(f'nn.Parameter of non-tensor {t!r}')
        if name in ('nn.Buffer', 'torch.nn.Buffer'):
            t = args[0]
            return Buffer(t.shape) if isinstance(t, Tensor) else Buffer(())
        node = self.classes.get(name)
        if node is None:
            return OPAQUE
        if self.is_dataclass(node):
            fields = OrderedDict()
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    fields[st.target.id] = self.eval(st.value, {}) if st.value is not None else None
            names = list(fields)
            for i, a in enumerate(args):
                fields[names[i]] = a
            fields.update(kwargs)
            methods = {st.name: st for st in node.body if isinstance(st, ast.FunctionDef)}
            return Obj(name, fields, methods)
        m = Module(name)
        init = next((st for st in node.body if isinstance(st, ast.FunctionDef) and st.name == '__init__'), None)
        if init is None:
            if self.linear_like(name):
                return self.make_linear(m, args, kwargs)
            return m
        env = self.bind(init, [m] + list(args), kwargs)
        env['__class__'] = name
        try:
            self.exec_block(init.body, env)
        except ReturnSignal:
            pass
        return m

    def make_linear(self, m, args, kwargs):
        i, o = args[0], args[1]
        bias = kwargs.get('bias', args[2] if len(args) > 2 else True)
        m.children['weight'] = Param((o, i))
        if bias:
            m.children['bias'] = Param((o,))
        return m

    def bind(self, fn, args, kwargs):
        env = {}
        params = fn.args.args
        defaults = fn.args.defaults
        nd = len(params) - len(defaults)
        for i, p in enumerate(params):
            if i < len(args):
                env[p.arg] = args[i]
            elif p.arg in kwargs:
                env[p.arg] = kwargs[p.arg]
            elif i >= nd:
                env[p.arg] = self.eval(defaults[i - nd], {})
            else:
                raise ValueError(f'missing argument {p.arg} for {fn.name}')
        for p, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
            env[p.arg] = kwargs.get(p.arg, self.eval(d, {}) if d is not None else None)
        return env

    # ------------------------------------------------------------------ statements
    def exec_block(self, body, env):
        for st in body:
            self.exec_stmt(st, env)

    def exec_stmt(self, st, env):
        if isinstance(st, ast.Expr):
            v = st.value
            if isinstance(v, ast.Call):
                f = ast.unparse(v.func)
                if f == 'super().__init__':
                    cls = env.get('__class__')
                    if cls and self.linear_like(cls):
                        args = [self.eval(a, env) for a in v.args]
                        kwargs = {k.arg: self.eval(k.value, env) for k in v.keywords}
                        self.make_linear(env['self'], args, kwargs)
                    return
                if f == 'self.register_buffer':
                    name = self.eval(v.args[0], env)
                    t = self.eval(v.args[1], env)
                    env['self'].children[name] = Buffer(t.shape if isinstance(t, Tensor) else ())
                    return
            return  # other expression statements (zero_(), docstrings, asserts-as-calls) do not create params
        if isinstance(st, ast.Assign):
            val = self.eval(st.value, env)
            for t in st.targets:
                self.assign(t, val, env)
            return
        if isinstance(st, ast.AnnAssign):
            if st.value is not None:
                self.assign(st.target, self.eval(st.value, env), env)
            return
        if isinstance(st, ast.If):
            cond = self.eval(st.test, env)
            if cond is OPAQUE:
                raise ValueError(f'cannot decide branch: {ast.unparse(st.test)}')
            self.exec_block(st.body if cond else st.orelse, env)
            return
        if isinstance(st, ast.For):
            it = self.eval(st.iter, env)
            for x in it:
                self.assign(st.target, x, env)
                self.exec_block(st.body, env)
            return
        if isinstance(st, ast.Return):
            raise ReturnSignal(self.eval(st.value, env) if st.value is not None else None)
        if isinstance(st, (ast.Assert, ast.Pass, ast.Import, ast.ImportFrom)):
            return
        raise ValueError(f'unsupported statement in __init__: {ast.unparse(st)[:80]}')

    def assign(self, target, val, env):
        if isinstance(target, ast.Name):
            env[target.id] = val
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
            m = env['self']
            if isinstance(val, (Module, Param, Buffer)) or val is None:
                m.children[target.attr] = val
            else:
                m.attrs[target.attr] = val
        elif isinstance(target, ast.Tuple):
            for t, v in zip(target.elts, val):
                self.assign(t, v, env)
        elif isinstance(target, ast.Subscript):
            base = self.eval(target.value, env)
            if isinstance(base, (dict, list)):
                base[self.eval(target.slice, env)] = val
        else:
            pass  # e.g. self.x.y = ... : not parameter-creating in these scripts

    # ------------------------------------------------------------------ expressions
    def eval(self, node, env):
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self.g:
                return self.g[node.id]
            if node.id in ('True', 'False', 'None'):
                return {'True': True, 'False': False, 'None': None}[node.id]
            return OPAQUE
        if isinstance(node, ast.Attribute):
            base = self.eval(node.value, env)
            if isinstance(base, Obj):
                if node.attr in base.fields:
                    return base.fields[node.attr]
                if node.attr in base.methods:
                    return ('method', base, base.methods[node.attr])
                return OPAQUE
            if isinstance(base, Module):
                if node.attr in base.children:
                    return base.children[node.attr]
                return base.attrs.get(node.attr, OPAQUE)
            if isinstance(base, dict):
                return base.get(node.attr, OPAQUE)
            return OPAQUE
        if isinstance(node, ast.BinOp):
            l, r = self.eval(node.left, env), self.eval(node.right, env)
            if isinstance(l, Tensor) or isinstance(r, Tensor):   # elementwise tensor op: shape of the tensor operand(s)
                if isinstance(l, Tensor) and isinstance(r, Tensor):
                    return Tensor(l.shape) if l.shape == r.shape else OPAQUE
                t, o = (l, r) if isinstance(l, Tensor) else (r, l)
                return Tensor(t.shape) if isinstance(o, (int, float)) else OPAQUE
            if l is OPAQUE or r is OPAQUE:
                return OPAQUE
            ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
                   ast.FloorDiv: lambda a, b: a // b, ast.Div: lambda a, b: a / b, ast.Mod: lambda a, b: a % b,
                   ast.Pow: lambda a, b: a ** b}
            return ops[type(node.op)](l, r)
        if isinstance(node, ast.UnaryOp):
            v = self.eval(node.operand, env)
            if v is OPAQUE:
                return OPAQUE
            return {ast.USub: lambda a: -a, ast.Not: lambda a: not a, ast.UAdd: lambda a: +a}[type(node.op)](v)
        if isinstance(node, ast.BoolOp):
            vals = [self.eval(v, env) for v in node.values]
            if any(v is OPAQUE for v in vals):
                return OPAQUE
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.Compare):
            left = self.eval(node.left, env)
            for op, c in zip(node.ops, node.comparators):
                right = self.eval(c, env)
                if left is OPAQUE or right is OPAQUE:
                    return OPAQUE
                f = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
                     ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
                     ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
                     ast.Is: lambda a, b: a is b, ast.IsNot: lambda a, b: a is not b}[type(op)]
                if not f(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            c = self.eval(node.test, env)
            if c is OPAQUE:
                raise ValueError(f'cannot decide: {ast.unparse(node.test)}')
            return self.eval(node.body if c else node.orelse, env)
        if isinstance(node, (ast.List, ast.Tuple)):
            out = []
            for e in node.elts:
                if isinstance(e, ast.Starred):
                    out.extend(self.eval(e.value, env))
                else:
                    out.append(self.eval(e, env))
            return out if isinstance(node, ast.List) else tuple(out)
        if isinstance(node, ast.Dict):
            return {self.eval(k, env): self.eval(v, env) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.ListComp):
            return list(self.comprehension(node, env))
        if isinstance(node, ast.GeneratorExp):
            return iter(list(self.comprehension(node, env)))
        if isinstance(node, ast.Subscript):
            base = self.eval(node.value, env)
            if isinstance(node.slice, ast.Slice):
                s = node.slice
                sl = slice(self.eval(s.lower, env), self.eval(s.upper, env), self.eval(s.step, env))
                return base[sl] if base is not OPAQUE else OPAQUE
            idx = self.eval(node.slice, env)
            if base is OPAQUE or idx is OPAQUE:
                return OPAQUE
            try:
                return base[idx]
            except Exception:
                return OPAQUE
        if isinstance(node, ast.Call):
            return self.call(node, env)
        if isinstance(node, ast.JoinedStr):
            return OPAQUE
        return OPAQUE

    def comprehension(self, node, env):
        gens = node.generators
        def rec(i, local):
            if i == len(gens):
                yield self.eval(node.elt, local)
                return
            g = gens[i]
            it = self.eval(g.iter, local)
            for x in it:
                l2 = dict(local)
                self.assign(g.target, x, l2)
                if all(self.eval(c, l2) for c in g.ifs):
                    yield from rec(i + 1, l2)
        return rec(0, dict(env))

    def call(self, node, env):
        fname = ast.unparse(node.func)
        args = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                args.extend(self.eval(a.value, env))
            else:
                args.append(self.eval(a, env))
        kwargs = {k.arg: self.eval(k.value, env) for k in node.keywords if k.arg}
        builtins = {'range': range, 'int': int, 'len': len, 'max': max, 'min': min, 'next': next, 'dict': dict,
                    'list': list, 'tuple': tuple, 'reversed': lambda x: list(reversed(x)), 'sum': sum, 'abs': abs,
                    'bool': bool, 'float': float, 'isinstance': lambda *a: OPAQUE, 'getattr': lambda *a: OPAQUE}
        if fname in builtins:
            if any(a is OPAQUE for a in args):
                return OPAQUE
            if fname == 'dict':
                return dict(*args, **kwargs)
            return builtins[fname](*args, **kwargs)
        if fname in ('torch.ones', 'torch.zeros', 'torch.empty', 'torch.randn'):
            shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
            if any(s is OPAQUE for s in shape):
                return OPAQUE
            return Tensor(shape)
        if fname == 'torch.arange':
            ints = [a for a in args if isinstance(a, int)]
            if len(ints) == len(args) and 1 <= len(args) <= 3:
                return Tensor((len(range(*args)),))
            return OPAQUE
        if fname == 'torch.linspace':
            steps = kwargs.get('steps', args[2] if len(args) > 2 else None)
            return Tensor((steps,)) if isinstance(steps, int) else OPAQUE
        if fname == 'torch.cat':
            parts = args[0] if args else []
            if parts and all(isinstance(t, Tensor) and len(t.shape) == 1 for t in parts):
                return Tensor((sum(t.shape[0] for t in parts),))
            return OPAQUE
        if fname == 'torch.einsum':
            if args and args[0].replace(' ', '') == 'i,j->ij' and all(isinstance(t, Tensor) for t in args[1:3]):
                return Tensor((args[1].shape[0], args[2].shape[0]))
            return OPAQUE
        if isinstance(node.func, ast.Attribute):
            base = self.eval(node.func.value, env)
            if isinstance(base, Tensor):
                m = node.func.attr
                if m in ('cos', 'sin', 'float', 'bfloat16', 'contiguous', 'clone', 'detach'):
                    return Tensor(base.shape)
                if m == 'new_zeros' and args and isinstance(args[0], int):
                    return Tensor((args[0],))
                return OPAQUE
        if fname == 'torch.tensor':
            v = args[0]
            if isinstance(v, (int, float)):
                return Tensor(())
            if isinstance(v, list):
                return Tensor((len(v),))
            return OPAQUE
        if fname in self.funcs:   # module-level helper function: interpret its body
            fn = self.funcs[fname]
            e2 = self.bind(fn, args, kwargs)
            try:
                self.exec_block(fn.body, e2)
            except ReturnSignal as r:
                return r.value
            return None
        f = self.eval(node.func, env)
        if isinstance(f, tuple) and f and f[0] == 'method':
            _, obj, fn = f
            e2 = self.bind(fn, [obj] + args, kwargs)
            try:
                self.exec_block(fn.body, e2)
            except ReturnSignal as r:
                return r.value
            return None
        if fname in self.classes or fname.startswith(('nn.', 'torch.nn.')):
            return self.instantiate(fname, args, kwargs)
        return OPAQUE

def module_constants(tree, extra):
    """Evaluate simple module-level constant assignments (e.g. num_vocab = 50304) in order."""
    g = dict(extra)
    it = Interp(tree, g)
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            name = st.targets[0].id
            if name in ('model', 'args', 'flags'):
                continue
            try:
                v = it.eval(st.value, dict(g))
            except Exception:
                v = OPAQUE
            if isinstance(v, (int, float, str, bool)) and not isinstance(v, Opaque):
                g[name] = v
    return g

def find_model_call(tree):
    for st in tree.body:
        if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'model' for t in st.targets):
            if isinstance(st.value, ast.Call) and ast.unparse(st.value.func) == 'GPT':
                return st.value
    raise ValueError('no module-level `model = GPT(...)`')

def inventory(path, preset=None):
    import warnings
    with open(path) as f:
        src = f.read()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)   # historical snapshots contain '\\s' in a docstring
        tree = ast.parse(src)
    extra = {}
    if preset is not None:
        import yaml
        with open(preset) as f:
            cfg = yaml.safe_load(f)
        extra['args'] = Obj('Hyperparameters', cfg['hparams'])
        extra['flags'] = Obj('Flags', cfg.get('flags_resolved', cfg.get('flags', {})))
    g = module_constants(tree, extra)
    if preset is not None:
        g['args'], g['flags'] = extra['args'], extra['flags']
    it = Interp(tree, g)
    call = find_model_call(tree)
    model = it.call(call, {})
    params = [(n, p.shape) for n, p in model.named_params()]
    buffers = [(n, b.shape) for n, b in model.named_buffers()]
    return params, buffers

def summarize(params):
    total = sum(math.prod(s) for _, s in params)
    groups = OrderedDict()
    for n, s in params:
        k = classify(n, s)
        groups[k] = groups.get(k, 0) + math.prod(s)
    return total, groups

def classify(name, shape):
    if name in ('lm_head.weight',):
        return 'lm_head'
    if name in ('transformer.wte.weight', 'embed.weight'):
        return 'embed'
    if 'vte' in name or 'value_embeds' in name:
        return 'value_embeddings'
    if name.startswith(('transformer.h.', 'blocks.')):
        return 'blocks'
    return 'other'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('script'); ap.add_argument('--preset'); ap.add_argument('--json')
    a = ap.parse_args()
    params, buffers = inventory(a.script, a.preset)
    total, groups = summarize(params)
    for n, s in params:
        print(f'{n:55s} {str(s):>18s} {math.prod(s):>12,d}')
    print('-' * 88)
    for k, v in groups.items():
        print(f'{k:20s} {v:>14,d}  ({v/1e6:.1f}M)')
    print(f'{"TOTAL":20s} {total:>14,d}  ({total/1e6:.1f}M)   buffers: {[(n, s) for n, s in buffers]}')
    if a.json:
        json.dump(dict(params=params, buffers=buffers, total=total, groups=groups), open(a.json, 'w'), indent=1)

if __name__ == '__main__':
    main()
