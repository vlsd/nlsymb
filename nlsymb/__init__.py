import numpy as np
from functools import reduce
import time
import scipy
from scipy.integrate import ode
import scipy.interpolate

# from matutils import matmult
def matmult(*x):
    """
    Shortcut for standard matrix multiplication.
    matmult(A,B,C) returns A*B*C.
    """
    return reduce(np.dot, x)


class LineSearch():
    def __init__(self, func, grad, alpha=1e-2, beta=1e-8):
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


