import numpy as np
# import sympy
# import sympy.core
# import sympy.core.symbol
from sympy.utilities.lambdify import lambdify as slambdify
from compiler.ast import flatten


# tensor lambdify
# returns a callable that returns a tensor
def lambdify(vars, expr):
    flags = ['buffered', 'delay_bufalloc', 'refs_ok']
    op_flags = [('readonly',), ('writeonly', 'allocate')]

    it = np.nditer([expr, None], flags=flags, op_flags=op_flags)
    it.reset()

    for (x, func) in it:
        func[...] = slambdify(vars, x)

    tensor_of_lambdas = it.operands[-1]

    def thread(*args):
        it = np.nditer([tensor_of_lambdas, None],
                       flags=flags, op_flags=op_flags,
                       op_dtypes=[np.object_, float])
        it.reset()

        for (func, out) in it:
            out[...] = func.item()(*args)

        return it.operands[-1]

    return thread


def diff(func, vars, out=None):
    func = np.array(func)
    vars = np.array(vars)
    flags = ['buffered', 'delay_bufalloc', 'reduce_ok', 'growinner', 'refs_ok']
    dtypes = [np.object_] * 3
    op_flags = [('readonly',)] * 2 + [('writeonly', 'allocate')]
    op_axes = [range(func.ndim) + [-1] * (vars.ndim),
               [-1] * (func.ndim) + range(vars.ndim), None]

    nditer = np.nditer([func, vars, None], flags=flags,
                       op_dtypes=dtypes, op_flags=op_flags, op_axes=op_axes)

    # why is this needed?
    nditer.reset()

    for (x, y, out) in nditer:
        from sympy import diff
        out[...] = diff(x, y)

    return nditer.operands[-1]


# a function to numerically evaluate numpy arrays
# containing sympy params using the corresponding values
# TODO might want to change the call to take a dict instead of params/vals
def eval(symb, params, vals):
    rule = {params[i]: vals[i] for i in range(len(vals))}

    flags = ['buffered', 'delay_bufalloc', 'refs_ok']
    it = np.nditer([symb, None], flags=flags, op_dtypes=[np.object_, float])
    it.operands[-1][...] = 0.0
    it.reset()

    for (el, out) in it:
        out[...] = el.item().subs(rule)

    return np.array(it.operands[-1])

# a function to perform symbolic substitutions on numpy arrays of
# sympy symbols


def subs(expr, rule):
    flags = ['buffered', 'delay_bufalloc', 'refs_ok']
    it = np.nditer([expr, None], flags=flags)
    it.operands[-1][...] = 0
    it.reset()

    for (el, out) in it:
        out[...] = el.item().subs(rule, simultaneous=True)

    return np.array(it.operands[-1])


class SymExpr():

    """
    wrapper around a tensor of symbolic expressions
    that behaves like one single expression
    """

    def __init__(self, expr):
        self.expr = np.array(expr)
        self.dims = self.expr.shape

    def callable(self, *args):
        self.func = lambdify(tuple(flatten(args)), self.expr)
        # thinking of adding cse compressed version of
        # expression here

    def subs(self, rule):
        return tensorSubs(self.expr, rule)

    def diff(self, params):
        return diff(self.expr, params)


def einsum(string, *arrays):
    """Simplified object einsum, not as much error checking

    does not support "..." or list input and will see "...", etc. as three
    times an axes identifier, tries normal einsum first!

    NOTE: This is untested, and not fast, but object type is
    never really fast anyway...
    """
    try:
        return np.einsum(string, *arrays)
    except TypeError:
        pass

    s = string.split('->')
    in_op = s[0].split(',')
    out_op = None if len(s) == 1 else s[1].replace(' ', '')

    in_op = [axes.replace(' ', '') for axes in in_op]
    all_axes = set()

    for axes in in_op:
        all_axes.update(axes)

    if out_op is None:
        out_op = set()
        for axes in in_op:
            for ax in axes:
                if ax in out_op:
                    out_op.remove(ax)
                else:
                    out_op.add(ax)
    else:
        all_axes.update(out_op)

    perm_dict = {_[1]: _[0] for _ in enumerate(all_axes)}

    dims = len(perm_dict)
    op_axes = []
    for axes in (in_op + list((out_op,))):
        op = [-1] * dims
        for i, ax in enumerate(axes):
            op[perm_dict[ax]] = i
        op_axes.append(op)

    op_flags = [('readonly',)] * len(in_op) + [('readwrite', 'allocate')]
    dtypes = [np.object_] * (len(in_op) + 1)  # cast all to object

    nditer = np.nditer(arrays + (None,),
                       op_axes=op_axes,
                       flags=['buffered', 'delay_bufalloc', 'reduce_ok',
                              'growinner', 'refs_ok'],
                       op_dtypes=dtypes,
                       op_flags=op_flags)

    nditer.operands[-1][...] = 0
    nditer.reset()

    for vals in nditer:
        out = vals[-1]
        prod = 1
        for value in vals[0:-1]:
            prod *= value
        out[...] += prod

    return nditer.operands[-1]


if __name__ == "__main__":
    pass
