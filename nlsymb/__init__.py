import numpy as np
from functools import reduce
import time
import scipy
from scipy.integrate import ode
import scipy.interpolate
from copy import deepcopy

# from matutils import matmult
def matmult(*x):
    """
    Shortcut for standard matrix multiplication.
    matmult(A,B,C) returns A*B*C.
    """
    return reduce(np.dot, x)


class Trajectory():
    # a class to represent a trajectory, takes lists of points and
    # returns interpolation objects (callables)
    def __init__(self, *args):
        # takes as arguments the names of the fields it stores
        for name in args:
            setattr(self, '_' + name, [])
        self._t = []
        self.tmax = None
        self.tmin = None
        self.feasible = False
        
    #def __call__(self, t):
    #    # evaluates at t if there is only one series stored
    #    # TODO make sure this works; not really necessary now
    #    num = 0
    #    for k in self.__dict__.keys():
    #        if k[0] is not '_':
    #            num += 1
    #            key = k
    #    if num is 1:
    #        func = getattr(self, key)
    #        return func(t)

    def addpoint(self, t, **kwargs):
        # keyword arguments in the form x=val
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
                ifunc = interxpolate(self._t, getattr(self, k), axis=0)
                setattr(self, k[1:], ifunc)

    def __add__(self, direction):
        out = deepcopy(self)
        for (t, x, u) in zip(out._t, out._x, out._u):
            x += direction.z(t)
            u += direction.v(t)
        out.interpolate()
        out.feasible = False
        return out


class LineSearch():
    def __init__(self, func, grad, alpha=1, beta=1e-8):
        # func takes a point
        # grad takes a point and a direction
        self.func = func
        self.grad = grad
        self.alpha = alpha
        self.beta = beta

    def set_x(self, x):
        self.x = x

    def set_p(self, p):
        self.p = p

    def search(self):
        x = self.x
        p = self.p
        grad = self.grad(x, p)
        func = self.func(x)
        gamma = self.alpha
        while self.func(x + gamma * p) > func + self.beta * gamma * grad:
            gamma = gamma/2
            print("decreasing gamma to %f" % gamma)
        self.gamma = gamma


class Timer():
    def __init__(self, fmts=""):
        self.fmts = fmts + " took %fs to run"
    
    def __enter__(self): 
        self.start = time.time()

    def __exit__(self, *args):
        delta = time.time() - self.start
        print( self.fmts % delta )


def sysIntegrate(func, init,
                 control=None, phi=None, debug=False,
                 tlimits=(0, 10), jac=None, method='bdf'):
    # func(t, x, u) returns xdot
    # control is parameter that gets passed to func, representing
    # a controller
    # phi(x) returns the distance to the switching plane if any
    # init is the initial value of x at tlimits[0]

    ti, tf = tlimits
    t, x = ([ti], [init])

    solver = ode(func, jac)
    solver.set_integrator('vode',
                          max_step=1e-1,
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
                # TODO do a line search instead scipy.optimize.brentq()
                alpha = dp/(dn-dp)
                tcross = t[-2] - alpha*(t[-1] - t[-2])
                xcross = x[-2] - alpha*(x[-1] - x[-2])

                # replace the wrong values
                t[-1], x[-1] = (tcross, xcross)

                # reset integration
                solver.set_initial_value(xcross, tcross)
                if debug:
                    print("found intersection at t=%f" % tcross)

    # make the last point be exactly at tf
    #xf = x[-2] + (tf - t[-2])*(x[-1] - x[-2])/(t[-1] - t[-2])
    #x[-1] = xf
    #t[-1] = tf
    return (t[:-1], x[:-1])


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


