import numpy as np
#import scipy
import sympy
import sympy.core
import sympy.core.symbol
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import xthreaded
import scipy
from compiler.ast import flatten
from scipy.interpolate import interp1d
import time

# tensor lambdify
# returns a callable that returns a tensor
def tLambdify(vars, expr):
    flags = ['buffered', 'delay_bufalloc', 'refs_ok']
    op_flags = [('readonly',), ('writeonly', 'allocate')]

    it = np.nditer([expr, None], flags=flags, op_flags=op_flags)
    it.reset()

    for (x, func) in it:
        func[...] = lambdify(vars, x)

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


def tdiff(func, vars, out=None):
    func = np.array(func)
    vars = np.array(vars)
    flags = ['buffered', 'delay_bufalloc', 'reduce_ok', 'growinner', 'refs_ok']
    dtypes = [np.object_]*3
    op_flags = [('readonly',)]*2 + [('writeonly', 'allocate')]
    op_axes = [range(func.ndim)+[-1]*(vars.ndim),
               [-1]*(func.ndim)+range(vars.ndim), None]

    nditer = np.nditer([func, vars, None], flags=flags,
                       op_dtypes=dtypes, op_flags=op_flags, op_axes=op_axes)

    # why is this needed?
    nditer.reset()

    for (x, y, out) in nditer:
        from sympy import diff
        out[...] = diff(x, y)

    return nditer.operands[-1]


def sysIntegrate(func, init,
                 control=None, phi=None,
                 tlimits=(0, 10), jac=None, method='bdf'):
    # func(t, x, u) returns xdot
    # control is parameter that gets passed to func, representing a controller
    # phi(x) returns the distance to the switching plane if any
    # init is the initial value of x at tlimits[0]
    from scipy.integrate import ode

    ti, tf = tlimits
    t, x = ([ti], [init])

    solver = ode(func, jac)
    solver.set_integrator('vode',
                          max_step=1e-1, min_step=1e-7,
                          method=method)
    solver.set_initial_value(init, ti)

    if control is not None:
        solver.set_f_params(control)

    dim = len(init)

    while solver.successful() and solver.t < tf:
        solver.integrate(tf, relax=True, step=True)
        x.append(solver.y)
        t.append(solver.t)
        if phi:
            dp, dn = map(phi, x[-2:])   # distance prev, distance next
            if dp*dn < 0:               # if a crossing occured
                # use interpolation (linear) to find the time
                # and config at the jump
                alpha = dp/(dn-dp)
                tcross = t[-2] - alpha*(t[-1] - t[-2])
                xcross = x[-2] - alpha*(x[-1] - x[-2])

                # replace the wrong values
                t[-1], x[-1] = (tcross, xcross)

                # reset integration
                solver.set_initial_value(xcross, tcross)
                print "found intersection at t=%f" % tcross

    # make the last point be exactly at tf
    #xf = x[-2] + (tf - t[-2])*(x[-1] - x[-2])/(t[-1] - t[-2])
    #x[-1] = xf
    #t[-1] = tf
    return (t[:-1], x[:-1])


# a function to numerically evaluate numpy arrays
# containing sympy params using the corresponding values
# TODO might want to change the call to take a dict instead of params/vals
def tensorEval(symb, params, vals):
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
def tensorSubs(expr, rule):
    flags = ['buffered', 'delay_bufalloc', 'refs_ok']
    it = np.nditer([expr, None], flags=flags)
    it.operands[-1][...] = 0
    it.reset()

    for (el, out) in it:
        out[...] = el.item().subs(rule, simultaneous=True)

    return np.array(it.operands[-1])


# from matutils import matmult
def matmult(*x):
    """
    Shortcut for standard matrix multiplication.
    matmult(A,B,C) returns A*B*C.
    """
    from functools import reduce
    return reduce(np.dot, x)


# function for finite differencing a callable that returns
# a tensor func(x), where x is a 1D array
# TODO rewrite this using nditer to make it work for any dimension tensors
def finite_diff(func, x, eps=1e-7):
    Ax = func(x)
    print(Ax)
    n = len(x)

    import copy
    output = []
    for i in range(n):
        xpertp = copy.deepcopy(x)
        xpertp[i] = x[i] + eps
        xpertm = copy.deepcopy(x)
        xpertm[i] = x[i] - eps
        h = xpertp[i]-xpertm[i]
        output.append((A(xpertp) - A(xpertm))/h)

    return np.rollaxis(np.array(output), 0, len(np.shape(output)))


# a wrapper around interp1d that also extrapolates
class interxpolate(scipy.interpolate.interp1d):
    def __call__(self, x):
        try:
            return super(interxpolate, self).__call__(x)
        except ValueError as e:
            # TODO make sure this is only triggered for the
            # proper exception. Maybe use error numbers?
            #print "WARNING: Interpolation called out of bounds, soldier!"
            xs, ys = (self.x, self.y)
            if x < xs[0]:
                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            elif x > xs[-1]:
                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
                raise


class tSymExpr():
    def __init__(self, expr):
        self.expr = np.array(expr)
        self.dims = self.expr.shape

    def callable(self, *args):
        self.func = tLambdify(tuple(flatten(args)), self.expr)
        # thinking of adding cse compressed version of
        # expression here

    def subs(self, rule):
        return tensorSubs(self.expr, rule)

    def diff(self, params):
        return tdiff(self.expr, params)


class trajectory:
    # a class to represent a trajectory, takes lists of points and
    # returns interpolation objects (callables)
    def __init__(self, *args):
        # takes as arguments the names of the fields it stores
        for name in args:
            setattr(self, '_' + name, [])
        self._t = []

    def addpoint(self, t, **kwargs):
        # keyword arguments in the form x=val
        self._t.append(t)
        for name, val in kwargs.iteritems():
            current = getattr(self, '_' + name)
            setattr(self, '_' + name, current + [val])

    def reset(self):
        # used for resetting all the args to []
        # does not delete interpolation objects already created
        for k in self.__dict__.keys():
            setattr(self, '_' + name, [])
            if k[0] is '_':
                setattr(self, k, [])
        self._t = []

    def interpolate(self):
        for k in self.__dict__.keys():
            if k[0] is '_' and k[1:] is not 't':
                ifunc = interxpolate(self._t, getattr(self, k), axis=0)
                setattr(self, k[1:], ifunc)


class Timer():
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print time.time() - self.start


class system():
    """
    a collection of callables, mostly """
    def __init__(self, func, tlims=(0, 10), **kwargs):
        # TODO add an option to pass a symSystem directly
        # skipping a bunch of middleman bullshit that might
        # otherwise have to happen

        self.tlims = tlims 
        keys = kwargs.keys()

        if 'xinit' in keys:
            self.xinit = kwargs['xinit']
            self.dim = len(self.xinit)
        elif 'dim' in keys:
            self.dim = kwargs['dim']
        else:
            self.dim = 1

        self.f = func
        self.u = kwargs['controller'] if 'controller' in keys else None
        self.dfdx = kwargs['dfdx'] if 'dfdx' in keys else None
        self.dfdu = kwargs['dfdu'] if 'dfdu' in keys else None
        self.phi = kwargs['phi'] if 'phi' in keys else None

    def set_phi(self, phi):
        self.phi = phi

    def set_u(self, controller):
        self.u = controller

    # TODO make sure this works in all combinations of linearization
    # or not, and controlled or not; first, though, get it working with
    # everything on
    def integrate(self, **kwargs):
        keys = kwargs.keys()
        xinit = kwargs['xinit'] if 'xinit' in keys else self.xinit
        use_jac = kwargs['use_jac'] if 'use_jac' in keys else False
        lin = kwargs['linearize'] if 'linearize' in keys else True

        if self.u is not None:
            func = lambda t, x: self.f(t, x, self.u(t, x))
            dfdx = lambda t, x: self.dfdx(t, x, self.u(t, x))
            dfdu = lambda t, x: self.dfdu(t, x, self.u(t, x))
        else:
            func = self.f
            dfdx = self.dfdx
            dfdu = self.dfdu

        jac = dfdx if use_jac else None

        (t, x) = sysIntegrate(func, self.xinit, tlimits=self.tlims,
                              phi=self.phi, jac=jac)

        components = ['x']
        if self.u:
            components.append('u')
        if lin:
            components.append('A')
            components.append('B')

        traj = trajectory(*components)
        for (tt, xx) in zip(t, x):
            dict = {'x': xx}
            if 'u' in components:
                dict['u'] = self.u(tt, xx)
            if 'A' in components:
                dict['A'] = dfdx(tt, xx)
            if 'B' in components:
                dict['B'] = dfdu(tt, xx)

            traj.addpoint(tt, **dict)

        return traj

if __name__ == "__main__":
    pass
