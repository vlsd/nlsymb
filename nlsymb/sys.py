import numpy as np
import sympy
import sympy.core
import sympy.core.symbol
from sympy.utilities.lambdify import lambdify
from sympy.utilities.decorator import xthreaded
import scipy
from compiler.ast import flatten
from scipy.interpolate import interp1d
import time
from copy import deepcopy

import tensor as tn


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

    def __call__(self, t):
        # evaluates at t if there is only one series stored
        # TODO make sure this works; not really necessary now
        num = 0
        for k in self.__dict__.keys():
            if k[0] is not '_':
                num += 1
                key = k
        if num is 1:
            func = getattr(self, key)
            return func(t)

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
        return out


class System():
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


class SymSys():
    # a representation of a hybrid/impulsive system
    # too much is hardcoded, but those functions had to go *somewhere*
    dim = 2

    t = S('t')
    z = map(S, ['z0', 'z1'])
    p = map(S, ['p0', 'p1'])
    q = map(S, ['q0', 'q1'])
    x = map(S, ['x0', 'x1', 'x2', 'x3'])
    u = map(S, ['u0', 'u1'])

    # create Ohm and dOhm
    _Ohm = np.array([z[0], z[1] + sym.sin(z[0])])

    _Psi = np.array([q[0], q[1] - sym.sin(q[0])])

    alltoz = [(q[i], _Ohm[i]) for i in range(dim)] + \
        [(x[i], z[i]) for i in range(dim)]
    alltoq = [(z[i], _Psi[i]) for i in range(dim)] + \
        [(x[i], _Psi[i]) for i in range(dim)]
    ztox = [(z[i], x[i]) for i in range(dim)]

    # dOhm/dz
    _dOhm = tdiff(_Ohm, z)
    #_dohm = tLambdify(z, _dOhm)

    # dPsi/dq
    _dPsi = tdiff(_Psi, q)
    #_dpsi = tLambdify(q, _dPsi)

    def Ohm(self, z):
        return tensorEval(self._Ohm, self.z, z)

    def dOhm(self, z):
        return tensorEval(self._dOhm, self.z, z)

    def Psi(self, q):
        return tensorEval(self._Psi, self.q, q)

    def dPsi(self, q):
        return tensorEval(self._dPsi, self.q, q)

    # this function takes and returns numerical values
    def xtopq(self, x):
        pz = self.P(x[:2])
        return self._ohm(*pz)

    def xtopz(self, x):
        return self.P(x[:2])

    def xtoq(self, x):
        q = x[:2]
        return self._ohm(*q)

    def xtoz(self, x):
        return x[:2]

    # M as a function of z
    def _buildMq(self):
        return np.eye(self.dim)*self.m

    # M inverse
    def _buildMqi(self):
        return np.eye(self.dim)/self.m

    # Mzar as a function of z
    def _buildMz(self):
        return matmult(self._dOhm.T, self.Mq, self._dOhm)

    # Mzar inverse
    def _buildMzi(self):
        # rule = [(self.q[i],self._Ohm[i]) for i in range(self.dim)]
        dpsi = np.empty_like(self._dPsi)
        for i in range(self.dim):
            for j in range(self.dim):
                dpsi[i, j] = self._dPsi[i, j].subs(self.alltoz,
                                                   simultaneous=True)
        return matmult(dpsi, self.Mqi, dpsi.T)

    # potentials
    def _build_Vq(self):
        return -self.m * self.g * self.q[1]

    def _build_Vz(self):
        return self._build_Vq().subs(self.alltoz, simultaneous=True)

    # \dot{x}=f(x)
    def _makefp(self):
        zdot = self.x[2:4]
        out = np.concatenate((zdot,
                              np.dot(self.Mzi,
                                     - object_einsum(
                                         'i,ijk,k', zdot, self.dMz, zdot)
                                     + object_einsum(
                                         'i,ikl,k', zdot, self.dMz, zdot)/2
                                     + self.dVz
                                     - np.dot(self._dOhm.T, self.u)
                                     )
                              ))

        return tSymExpr(tensorSubs(out, self.ztox))

    def _makefm(self):
        zdot = self.x[2:4]
        zz = self._P
        zzdot = np.dot(self._dP, zdot)

        out = -object_einsum('i,ijk,k', zzdot, self.dMzz, zzdot) \
            + object_einsum('i,ikj,k', zzdot, self.dMzz, zzdot)/2
        out = np.dot(self.Mzzi, out + self.dVzz - np.dot(self._dOhm.T, self.u))
        out = out - object_einsum('ijk,j,k',
                                  tdiff(self._dP, self.z), zdot, zdot)
        # note: dP is the same as dPinverse. woo!
        out = np.concatenate((zdot, np.dot(self._dPi, out)))

        return tSymExpr(tensorSubs(out, self.ztox))

    def _makeP(self, k):
        # builds the symbolic expression for the projection
        # does NOT check for feasible/infeasible stuff
        si = self.si
        z = self.z
        zs = z[si]
        out = deepcopy(z)

        out[si] = -zs
        for i in range(len(z)):
            if i != si:
                out[i] = z[i] - self.delta[i]*2*zs/(1+k*zs**2)

        return np.array(out)

    # returns a replacement rule u->mu+K(alpha-x) or something like that
    # takes in a trajectory object
    def feedback(self, t, K, traj):
        mu = traj.u(t)
        alpha = traj.x(t)
        u = mu + np.dot(K, alpha-self.x)
        return {self.u[i]: u[i] for i in range(u)}

    # can i conflate these three functions into one somehow?
    # probably, will have to think on it
    def f(self, t, xval, uval=[0, 0], ctrl=None):
        # choose between _fplus and _fmins
        # depending on the configuration
        # assume that ctrl is a rule for substituting u
        if xval[self.si] >= 0:
            func = self._fplus.func
        else:
            func = self._fmins.func

        vals = np.concatenate([[t], xval, uval])

        return func(*vals)

    def dfdx(self, t, xval, uval=[0, 0]):
        # choose between dfxm and dfxp
        if xval[self.si] > 0:
            func = self._dfxp.func
        else:
            func = self._dfxm.func
        vals = np.concatenate([[t], xval, uval])
        return func(*vals)

    def dfdu(self, t, xval, uval=[0, 0]):
        # choose between dfxm and dfxp
        if xval[self.si] > 0:
            func = self._dfup.func
        else:
            func = self._dfum.func
        vals = np.concatenate([[t], xval, uval])
        return func(*vals)

    def P(self, zval):
        # choose between identity and fancy projection
        if zval[self.si] > 0:
            return zval
        else:
            expr = self._P
            vals = zval
            params = self.z
            return tensorEval(expr, params, vals)

    def dP(self, zval):
        # for debug purposes
        return tensorEval(self._dP, self.z, zval)

    def phi(self, xval):
        return xval[1]

    def __init__(self, si=1, k=50.0, m=1.0, g=9.8):
        self.si = si
        # this is the special index: z[si] = phi(z)
        self.k = k  # constant for projection
        self.m = m  # mass
        self.g = g  # gravitational constant

        self.Mq = self._buildMq()
        self.Mz = self._buildMz()
        self.Mqi = self._buildMqi()
        self.Mzi = self._buildMzi()

        self.dMq = tdiff(self.Mq, self.q)
        self.dMz = tdiff(self.Mz, self.z)

        self.delta = self.Mzi[:, self.si] / self.Mzi[self.si, self.si]
        #self.ddelta = tdiff(self.delta, self.z)

        self.Vz = self._build_Vz()
        self.dVz = tdiff(self.Vz, self.z)

        self._P = self._makeP(self.k)
        self._dP = tdiff(self._P, self.z)
        self._dPi = np.array(sym.Matrix(self._dP).inv())

        self.ztozz = {self.z[i]: self._P[i] for i in range(self.dim)}
        self.Mzzi = tensorSubs(self.Mzi, self.ztozz)
        self.dMzz = tensorSubs(self.dMz, self.ztozz)
        self.dVzz = tensorSubs(self.dVz, self.ztozz)
        self.dPzz = tensorSubs(self._dP, self.ztozz)

        params = [self.t, self.x, self.u]

        self._fplus = self._makefp()
        self._fplus.callable(*params)
        self._fmins = self._makefm()
        self._fmins.callable(*params)

        self._dfxp = tSymExpr(self._fplus.diff(self.x))
        self._dfxp.callable(*params)
        self._dfxm = tSymExpr(self._fmins.diff(self.x))
        self._dfxm.callable(*params)
        self._dfup = tSymExpr(self._fplus.diff(self.u))
        self._dfup.callable(*params)
        self._dfum = tSymExpr(self._fmins.diff(self.u))
        self._dfum.callable(*params)

        self.controller = lambda t, x: [0, 0]
        self._ohm = tLambdify(self.z, self._Ohm)
        self._psi = tLambdify(self.q, self._Psi)


if __name__ == "__main__":
    pass
