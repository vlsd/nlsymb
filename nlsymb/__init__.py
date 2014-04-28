import numpy as np
import sympy as sym

from functools import reduce
import time
import scipy
from scipy.integrate import ode
import scipy.interpolate
from copy import deepcopy
from timeout import TimeoutError
from termcolor import colored

# from matutils import matmult


def matmult(*x):
    """
    Shortcut for standard matrix multiplication.
    matmult(A,B,C) returns A*B*C.
    """
    return reduce(np.dot, x)


class Trajectory():
    # a class to reyysresent a trajectory, takes lists of points and
    # returns interpolation objects (callables)

    def __init__(self, *args):
        # takes as arguments the names of the fields it stores
        for name in args:
            setattr(self, '_' + name, [])
        self._t = []
        self.tmax = None
        self.tmin = None
        self.feasible = False

    # def __call__(self, t):
    # evaluates at t if there is only one series stored
    # TODO make sure this works; not really necessary now
    #    num = 0
    #    for k in self.__dict__.keys():
    #        if k[0] is not '_':
    #            num += 1
    #            key = k
    #    if num is 1:
    #        func = getattr(self, key)
    #        return func(t)

    def addpoint(self, t, **kwargs):
        # keyword arguments in the foysm x=val
        if self._t is []:
            self.tmax = t
            self.tmin = t
        else:
            if t > self.tmax:
                self.tmax = t
            if t < self.tmin:
                self.tmin = t
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
                ifunc = interxpolate(self._t, getattr(self, k),
                                     axis=0, kind='slinear')
                setattr(self, k[1:], ifunc)

    def __add__(self, other):
        if len(other._t) > len(self._t):
            return other + self

        names = (set(self.__dict__) & set(other.__dict__)) - {'t', '_t'}
        names = {k[1:] for k in names if k[0] is '_'}
        tj = Trajectory(*names)
        other.interpolate()
        for t in self._t:
            tj.addpoint(t, **{n: (getattr(self, n)(t) + getattr(other, n)(t))
                              for n in names})
        tj.interpolate()
        tj.feasible = False
        # use the most restrictive time limits
        tj.tlims = (max(self.tlims[0], other.tlims[0]),
                    min(self.tlims[1], other.tlims[1]))
        return tj

    def __neg__(self):
        return -1.0*self

    def __rmul__(self, scalar):
        # multiplies everything by the scalar
        names = {k[1:] for k in self.__dict__.keys()
                 if k[0] is '_' and k[1:] is not 't'}
        out = Trajectory(*names)
        for t in self._t:
            out.addpoint(t, **{n: (scalar * getattr(self, n)(t))
                               for n in names})

        out.interpolate()
        out.feasible = False
        out.tlims = self.tlims
        return out

    """ old version of add, see above for new version
    def __add__(self, other):
        out = deepcopy(self)
        for (t, x, u) in zip(out._t, out._x, out._u):
            x += direction.z(t)
            u += direction.v(t)
        out.interpolate()
        out.feasible = False
        return out
    """

    def xtoq(self, s):
        self._q = map(s.xtopq, self._x)
        self.interpolate()

    def xtonq(self, s):
        self._q = map(s.xtoq, self._x)
        self.interpolate()

    def __getstate__(self):
        temp = self.__dict__.copy()
        for k in temp.keys():
            if k[0] is '_':
                pass
            elif k not in ['tmin', 'tmax', 'feasible']:
                del temp[k]
        return temp


class LineSearch():

    def __init__(self, func, grad, alpha=1, beta=1e-8):
        # func takes a point
        # grad takes a point and a direction
        self.func = func
        self.grad = grad
        self.alpha = alpha
        self.beta = beta

    def search(self):
        x = self.x
        p = self.p
        # grad = self.grad(x, p)
        grad = 1
        func = self.func(x)
        gamma = self.alpha
        while True:
            try:
                if self.func(x + gamma * p) > \
                        func + self.beta * gamma * grad:
                    gamma = gamma / 2
                    print("decreasing gamma to %e" % gamma)
                    # this will not work with the -O flag
                    assert gamma > 1e-15, gamma
                else:
                    break
            except TimeoutError:
                gamma = gamma / 10
                print("Timed out, decreasing gamma to %e" % gamma)
            except OverflowError:
                gamma = gamma / 10
                print("Error in VODE, decreasing gamma to %e" % gamma)

        self.gamma = gamma


class Timer():

    def __init__(self, fmts=""):
        self.fmts = fmts + " took %fs to run"

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        delta = time.time() - self.start
        print(self.fmts % delta)


def sysIntegrate(func, init, control=None, phi=None, debug=False, 
                 tlims=(0, 10), jac=None, method='bdf', **kw):
    """
    func(t, x, u): returns xdot
    init: the value of x at tlims[0]
    'tlims': (ta, tb), ta < tb
    'control': a Controller() instance
    'jac': jac(t, x, u) the jacobian of func. used only if provided,
            not used if 'control' is provided
    'method': see the 'method' argument for the 'vode' integrator
    'debug': if True, prints debug statements
    'phi': phi(x) that returns the distance to the switching plane
    'jumps': [(tj,fj), ...] list of times and jump matrices
             fj is a matrix that multiplies x at the jump time
    'delfunc': delf(t, x, u) a callable that returns a jump matrix
    """

    ti, tf = tlims
    t, x = ([ti], [init])

    solver = ode(func, jac)
    solver.set_integrator('vode',
                          max_step=1e-2,
                          method=method)
    solver.set_initial_value(init, ti)

    if control is not None:
        solver.set_f_params(control)

    dim = len(init)

    jumps_out = []
    jumps_in = kw['jumps'] if 'jumps' in kw else []

    while solver.successful() and solver.t < tf + 1e-2:
        solver.integrate(tf, relax=True, step=True)
        
        xx = solver.y
        if jumps_in:
            for (tj, fj) in jumps_in:
                if t[-1] < tj and tj < solver.t:
                    xx = xx  + matmult(fj,xx)
                    solver.set_initial_value(xx, solver.t)

        x.append(xx)
        t.append(solver.t)
        
        if phi:
            dp, dn = map(phi, x[-2:])   # distance prev, distance next
            if dp * dn < 0:               # if a crossing occured
                # use interpolation (linear) to find the time
                # and config at the jump
                # TODO do a line search instead: scipy.optimize.brentq()
                alpha = dp / (dn - dp)
                tcross = t[-2] - alpha * (t[-1] - t[-2])
                xcross = x[-2] - alpha * (x[-1] - x[-2])

                # replace the wrong values
                #si = 1
                #xcross[si] = 0.0
                t[-1], x[-1] = (tcross, xcross)

                # obtain jump term
                if 'delfunc' in kw:
                    delf = kw['delfunc']
                    jmatrix = delf(tcross, xcross)
                    jumps_out.append((tcross, jmatrix))

                # reset integration
                solver.set_initial_value(xcross, tcross)
                if debug:
                    print("found intersection at t=%f" % tcross)
            #separation
            elif dp==0 and dn > 0:
                # right now the dynamics should take care of this
                pass

            

    # make the last point be exactly at tf
    # xf = x[-2] + (tf - t[-2])*(x[-1] - x[-2])/(t[-1] - t[-2])
    # x[-1] = xf
    # t[-1] = tf
    return (t[:-1], x[:-1], jumps_out)


# a wrapper around interp1d that also extrapolates
class interxpolate(scipy.interpolate.interp1d):
    def __call__(self, x):
        try:
            return super(interxpolate, self).__call__(x)
        except ValueError as e:
            # TODO make sure this is only triggered for the
            # proper exception. Maybe use error numbers?
            xs, ys = (self.x, self.y)
            if x < xs[0] - 2e-2 or x > xs[-1] + 2e-2:
                print "ERROR: Interpolation called out of bounds at time %f" % x
                raise

            # if it is within tolerance simply extrapolate
            if x < xs[0]:
                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            elif x > xs[-1]:
                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
                raise
