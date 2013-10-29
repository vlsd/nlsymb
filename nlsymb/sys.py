import numpy as np
import sympy as sym
from sympy import Symbol as S
from copy import deepcopy
from scipy.integrate import trapz

import tensor as tn
from . import matmult, interxpolate, sysIntegrate, Trajectory
from lqr import LQR, Controller
from timeout import timeout


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
        self.ufun = kwargs['controller'] if 'controller' in keys else None
        self.dfdx = kwargs['dfdx'] if 'dfdx' in keys else None
        self.dfdu = kwargs['dfdu'] if 'dfdu' in keys else None
        self.phi = kwargs['phi'] if 'phi' in keys else None

        if self.ufun is None:
            self.dimu = 0
        else:
            self.dimu = len(self.dimu(tlims[0]))

    # to be called after a reference has been set
    def build_cost(self, **kwargs):
        self.cost = CostFunction(self.dim, self.dimu, self.ref, 
                                 projector=self.project, **kwargs)
        return self.cost

    def set_u(self, controller):
        if 'ufun' in self.__dict__.keys():
            self._uold = self.ufun
        self.ufun = controller

    def reset_u(self):
        if '_uold' in self.__dict__.keys():
            self.u = self._uold
        else:
            print("Nothing to reset to. Not doing anything.")

    # TODO make sure this works in all combinations of linearization
    # or not, and controlled or not; first, though, get it working with
    # everything on
    def integrate(self, use_jac=False, linearize=True,
                  interpolate=True,**kwargs):
        keys = kwargs.keys()
        xinit = kwargs['xinit'] if 'xinit' in keys else self.xinit
        lin = linearize
        interp = interpolate

        if self.ufun is not None:
            func = lambda t, x: self.f(t, x, self.ufun(t, x))
            dfdx = lambda t, x: self.dfdx(t, x, self.ufun(t, x))
            dfdu = lambda t, x: self.dfdu(t, x, self.ufun(t, x))
        else:
            func = self.f
            dfdx = self.dfdx
            dfdu = self.dfdu

        jac = dfdx if use_jac else None

        (t, x) = sysIntegrate(func, self.xinit, tlimits=self.tlims,
                              phi=self.phi, jac=jac)

        components = ['x']
        if 'ufun' in self.__dict__.keys():
            components.append('u')
        if lin:
            components.append('A')
            components.append('B')

        traj = Trajectory(*components)
        for (tt, xx) in zip(t, x):
            dict = {'x': xx}
            if 'u' in components:
                dict['u'] = self.ufun(tt, xx)
            if 'A' in components:
                dict['A'] = dfdx(tt, xx)
            if 'B' in components:
                dict['B'] = dfdu(tt, xx)

            traj.addpoint(tt, **dict)

        # interpolate, unless requested; 
        # saves a few manual calls
        if interp:
            traj.interpolate()
        else:
            pass

        if lin:
            print("linearizing...")
            self.lintraj = traj
            self.regulator = LQR(traj.A, traj.B, 
                                 tlims=self.tlims)

        traj.feasible = True
        traj.tlims = self.tlims
        return traj
    

    @timeout(30)
    def project(self, traj, tlims=None, lin=False):
        if traj.feasible:
            return traj

        if tlims is None:
            tlims = self.tlims

        self.xinit = traj.x(tlims[0])

        if 'regulator' in self.__dict__.keys():
            #print("regular projection")
            ltj = self.lintraj
            reg = self.regulator
            control = Controller(reference=traj, K=reg.K)

            self.set_u(control)
            #print(lin)
            return self.integrate(linearize=lin)
        else:
            print("integrating and linearizing for the first time")
            control = Controller(reference=traj)
            
            self.set_u(control)
            nutraj = self.integrate(linearize=True)

            return self.project(nutraj, tlims=tlims, 
                                lin=lin)

class CostFunction():
    def __init__(self, dimx, dimu, ref,
                 R=None, Q=None, PT=None, projector=None):
        self.dimx = dimx
        self.dimu = dimu
        self.ref = ref
        self.R = R
        self.Q = Q
        self.PT = PT
        self.projector = (lambda x: x) if projector is None else projector

    def __call__(self, traj):
        tj = self.projector(traj)
        T = tj.tlims[1]

        tlist = tj._t
        elist = []
        for (t, x, u) in zip(tlist, tj._x, tj._u):
            xd = self.ref.x(t)
            ud = self.ref.u(t)
            Q = self.Q(t)
            R = self.R(t)
            expr = matmult(x-xd, Q, x-xd) + matmult(u-ud, R, u-ud)
            elist.append(expr)

        # integrate the above
        out = 0.5 * trapz(elist, tlist)
        out += 0.5 * matmult(tj.x(T)-self.ref.x(T), self.PT,
                             tj.x(T)-self.ref.x(T))
        return out

    def grad(self, traj, dir):
        # this shouldn't be needed
        pass
    #    #tj = self.project(traj)
    #    
    #    T = dir.tmax

    #    tlist = dir._t
    #    elist = []
    #    for (t, z, v) in zip(tlist, dir._z, dir._v):
    #        a = dir.a(t)
    #        b = dir.b(t)
    #        elist.append(matmult(a.T, z) + matmult(b.T, v))

    #    out = trapz(elist, tlist)
    #    out += matmult(dir.r(T), dir.z(T))

    #    return out
    

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

    qtoz = zip(q, _Ohm)
    alltoz = qtoz + [(x[i], z[i]) for i in range(dim)]
    ztoq = zip(z, _Psi)
    alltoq = ztoq + [(x[i], _Psi[i]) for i in range(dim)]
    ztox = [(z[i], x[i]) for i in range(dim)]

    # dOhm/dz
    _dOhm = tn.diff(_Ohm, z)
    #_dohm = tn.lambdify(z, _dOhm)

    # dPsi/dq
    _dPsi = tn.diff(_Psi, q)
    #_dpsi = tn.lambdify(q, _dPsi)

    def Ohm(self, z):
        return tn.eval(self._Ohm, self.z, z)

    def dOhm(self, z):
        return tn.eval(self._dOhm, self.z, z)

    def Psi(self, q):
        return tn.eval(self._Psi, self.q, q)

    def dPsi(self, q):
        return tn.eval(self._dPsi, self.q, q)

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
                                     - tn.einsum(
                                         'i,ijk,k', zdot, self.dMz, zdot)
                                     + tn.einsum(
                                         'i,ikl,k', zdot, self.dMz, zdot)/2
                                     + self.dVz) \
                              + matmult(\
                                  tn.subs(self._dPsi, self.qtoz),
                                  self.Mqi, # here we might need subs
                                  self.u)
                              ))

        return tn.SymExpr(tn.subs(out, self.ztox))

    def _makefm(self):
        zdot = self.x[2:4]
        OhmP = tn.subs(self._Ohm, zip(self.z, self._P))
        OhmI = tn.subs(self._dPsi, zip(self.q, OhmP))

        zz = self._P
        zzdot = np.dot(self._dP, zdot)

        out = -tn.einsum('i,ijk,k', zzdot, self.dMzz, zzdot) \
            + tn.einsum('i,ikj,k', zzdot, self.dMzz, zzdot)/2
        out = np.dot(self.Mzzi, out + self.dVzz)
        out = out - tn.einsum('ijk,j,k',
                                  tn.diff(self._dP, self.z), zdot, zdot)
        out = out + matmult(OhmI,
                            self.Mqi, # in general, there should be a subs here
                            self.u)
        out = matmult(self._dPi, out)
        out = np.concatenate((zdot, out))

        return tn.SymExpr(tn.subs(out, self.ztox))

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
            return tn.eval(expr, params, vals)

    def dP(self, zval):
        # for debug purposes
        return tn.eval(self._dP, self.z, zval)

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

        self.dMq = tn.diff(self.Mq, self.q)
        self.dMz = tn.diff(self.Mz, self.z)

        self.delta = self.Mzi[:, self.si] / self.Mzi[self.si, self.si]
        #self.ddelta = tn.diff(self.delta, self.z)

        self.Vz = self._build_Vz()
        self.dVz = tn.diff(self.Vz, self.z)

        self._P = self._makeP(self.k)
        self._dP = tn.diff(self._P, self.z)
        self._dPi = np.array(sym.Matrix(self._dP).inv())

        self.ztozz = {self.z[i]: self._P[i] for i in range(self.dim)}
        self.Mzzi = tn.subs(self.Mzi, self.ztozz)
        self.dMzz = tn.subs(self.dMz, self.ztozz)
        self.dVzz = tn.subs(self.dVz, self.ztozz)
        self.dPzz = tn.subs(self._dP, self.ztozz)

        params = [self.t, self.x, self.u]

        self._fplus = self._makefp()
        self._fplus.callable(*params)
        self._fmins = self._makefm()
        self._fmins.callable(*params)

        self._dfxp = tn.SymExpr(self._fplus.diff(self.x))
        self._dfxp.callable(*params)
        self._dfxm = tn.SymExpr(self._fmins.diff(self.x))
        self._dfxm.callable(*params)
        self._dfup = tn.SymExpr(self._fplus.diff(self.u))
        self._dfup.callable(*params)
        self._dfum = tn.SymExpr(self._fmins.diff(self.u))
        self._dfum.callable(*params)

        self.controller = lambda t, x: [0, 0]
        self._ohm = tn.lambdify(self.z, self._Ohm)
        self._psi = tn.lambdify(self.q, self._Psi)


if __name__ == "__main__":
    pass
