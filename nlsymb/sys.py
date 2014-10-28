from nlsymb import deepcopy, np, sym, scipy, matmult,\
        interxpolate, sysIntegrate, Trajectory

import tensor as tn
from sympy import Symbol as S
from scipy.integrate import trapz
from sympy import S as symbol

#from nlsymb import matmult, interxpolate, sysIntegrate, Trajectory
from lqr import LQR, Controller
from timeout import timeout
from IPython.core.debugger import Tracer

class System(object):

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
        self.delf = kwargs['delf'] if 'delf' in kwargs else None

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
    # or not, and controlled or not;
    # Major cleanup needed.
    def integrate(self, use_jac=False, linearize=True,
                  interpolate=True, **kwargs):
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

        #Tracer()()
        if self.delf is not None:
            delfunc = lambda t, x: self.delf(t, x, self.ufun(t, x))
            (t, x, jumps) = sysIntegrate(func, self.xinit, tlims=self.tlims,
                                     phi=self.phi, jac=jac, delfunc=delfunc)
        else:
            (t, x, jumps) = sysIntegrate(func, self.xinit, tlims=self.tlims,
                                     phi=self.phi, jac=jac)


        #Tracer()()
        components = ['x']
        if 'ufun' in self.__dict__.keys():
            components.append('u')
        if lin:
            components.append('A')
            components.append('B')

        traj = Trajectory(*components)
        for (tt, xx) in zip(t, x):
            names = {'x': xx}
            if 'u' in components:
                names['u'] = self.ufun(tt, xx)
            if 'A' in components:
                names['A'] = dfdx(tt, xx)
            if 'B' in components:
                names['B'] = dfdu(tt, xx)

            traj.addpoint(tt, **names)

        # interpolate, unless requested;
        # saves a few manual calls
        if interp:
            traj.interpolate()

        if lin:
            print("linearizing...")
            self.lintraj = traj
            self.regulator = LQR(self.tlims, traj.A, traj.B)#, jumps=jumps)
            self.regulator.solve()

        traj.feasible = True
        traj.tlims = self.tlims
        traj.jumps = jumps
        return traj

    @timeout(100)
    def project(self, traj, tlims=None, lin=False):
        if traj.feasible:
            return traj

        if tlims is None:
            tlims = self.tlims

        self.xinit = traj.x(tlims[0])

        if 'regulator' in self.__dict__:
            # print("regular projection")
            ltj = self.lintraj
            reg = self.regulator
            control = Controller(reference=traj, K=reg.K)

            self.set_u(control)
            # print(lin)
            return self.integrate(linearize=lin)
        else:
            print("integrating and linearizing for the first time")
            control = Controller(reference=traj)
            
            self.set_u(control)
            nutraj = self.integrate(linearize=True)
            
            return self.project(nutraj, tlims=tlims, lin=lin)


class CostFunction(object):

    def __init__(self, dimx, dimu, ref,
                 R=None, Q=None, PT=None, projector=None):
        self.dimx = dimx
        self.dimu = dimu
        self.ref = ref
        self.R = R
        self.Q = Q
        self.PT = PT
        self.projector = (lambda x: x) if projector is None else projector

    def __call__(self, traj, tspace=False):
        tj = traj if traj.feasible or tspace else self.projector(traj) 
        ta, tb = tj.tlims
        T = tb

        tlist = np.linspace(ta, tb, (tb-ta)*1e3, endpoint=True)
        xlist = [tj.x(t) for t in tlist]
        ulist = [tj.u(t) for t in tlist]
        elist = []
        for (t, x, u) in zip(tlist, xlist, ulist):
            # only consider reference if not in tangent space
            xd = (0.0 if tspace else 1.0)*self.ref.x(t) 
            ud = (0.0 if tspace else 1.0)*self.ref.u(t)
            Q = self.Q(t)
            R = self.R(t)
            expr = matmult(x - xd, Q, x - xd) + matmult(u - ud, R, u - ud)
            elist.append(expr)

        # integrate the above
        out = 0.5 * trapz(elist, tlist)
        # don't add a terminal cost in tangent space
        if not tspace:
            out += 0.5 * matmult(tj.x(T) - self.ref.x(T), 
                                 self.PT, tj.x(T) - self.ref.x(T))
        return out

    def grad(self, traj, dir):
        # this shouldn't be needed
        pass

class SymSys(object):
    # a representation of a hybrid/impulsive system
    # this class is not to be called directly, but instead inherited
    # such that needed attributes, like q, x, M, etc. are implemented
    # before calling __init__()
    def __init__(self, si=0, **kwargs):
        # this is the special index: z[si] = phi(z)
        self.si = si
        self.t = S('t')
        n = self.dim

        self.qtoz = zip(self.q, self._Ohm)
        self.alltoz = self.qtoz + zip(self.x, self.z)
        self.ztoq = zip(self.z, self._Psi)
        self.alltoq = self.ztoq + zip(self.x, self._Psi)
        self.ztox = zip(self.z, self.x)

        # dOhm/dz, dPhi/dq, assuming the pieces are already defined
        self._dOhm = tn.diff(self._Ohm, self.z)
        self._dPsi = tn.diff(self._Psi, self.q)
        
        self.Mz = matmult(self._dOhm.T, self.Mq, self._dOhm)
        self.Mzi = self._Mzi() 

        self.dMq = tn.diff(self.Mq, self.q)
        self.dMz = tn.diff(self.Mz, self.z)

        self.delta = self.Mzi[:, self.si] / self.Mzi[self.si, self.si]
        # self.ddelta = tn.diff(self.delta, self.z)

        self.Vz = self.Vq.subs(self.alltoz, simultaneous=True)
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

        self._fplus = self._makefp(params)
        self._fmins = self._makefm(params)

        self._dfxp = tn.SymExpr(self._fplus.diff(self.x))
        self._dfxp.callable(*params)
        self._dfxm = tn.SymExpr(self._fmins.diff(self.x))
        self._dfxm.callable(*params)
        self._dfup = tn.SymExpr(self._fplus.diff(self.u))
        self._dfup.callable(*params)
        self._dfum = tn.SymExpr(self._fmins.diff(self.u))
        self._dfum.callable(*params)

        self._ohm = tn.lambdify(self.z, self._Ohm)
        self._psi = tn.lambdify(self.q, self._Psi)

        # make the jump term generator callable
        # and a bunch of other stuff as well
        self.delf = lambda t, x, u: self._delf(t, x, u)

        self.Ohm = lambda z: tn.eval(self._Ohm, self.z, z)
        self.dOhm = lambda z: tn.eval(self._dOhm, self.z, z)
        self.Psi = lambda q: tn.eval(self._Psi, self.q, q)
        self.dPsi = lambda q: tn.eval(self._dPsi, self.q, q)

    # this function takes and returns numerical values
    def xtopq(self, x):
        pz = self.P(x[:self.dim])
        return self._ohm(*pz)

    def xtopz(self, x):
        return self.P(x[:self.dim])

    def xtoq(self, x):
        q = x[:self.dim]
        return self._ohm(*q)

    def xtoz(self, x):
        return x[:self.dim]

    # Mz inverse
    def _Mzi(self):
        dpsi = np.empty_like(self._dPsi)
        for i in range(self.dim):
            for j in range(self.dim):
                dpsi[i, j] = self._dPsi[i, j].subs(self.alltoz,
                                                   simultaneous=True)
        return matmult(dpsi, self.Mqi, dpsi.T)

    # \dot{x}=f(x)
    def _makefp(self, params):
        zdot = self.x[self.dim:]
        out = np.concatenate((zdot,
                              np.dot(self.Mzi,
                                     - tn.einsum(
                                         'i,ijk,k', zdot, self.dMz, zdot)
                                     + tn.einsum(
                                         'i,ikl,k', zdot, self.dMz, zdot) / 2
                                     + self.dVz)
                              + matmult(
                                  tn.subs(self._dPsi, self.qtoz),
                                  self.Mqi,  # here we might need subs
                                  self.u)
                              ))

        out = tn.SymExpr(tn.subs(out, self.ztox))
        out.callable(*params)
        return out

    def _makefm(self, params):
        zdot = self.x[self.dim:]
        OhmP = tn.subs(self._Ohm, zip(self.z, self._P))
        OhmI = tn.subs(self._dPsi, zip(self.q, OhmP))

        zz = self._P
        zzdot = np.dot(self._dP, zdot)

        out = -tn.einsum('i,ijk,k', zzdot, self.dMzz, zzdot) \
            + tn.einsum('i,ikj,k', zzdot, self.dMzz, zzdot) / 2
        out = np.dot(self.Mzzi, out + self.dVzz)
        out = out - tn.einsum('ijk,j,k',
                              tn.diff(self._dP, self.z), zdot, zdot)
        out = out + matmult(OhmI, self.Mqi, self.u)
                            # in general, there should be a subs here
        out = matmult(self._dPi, out)
        out = np.concatenate((zdot, out))

        out = tn.SymExpr(tn.subs(out, self.ztox))
        out.callable(*params)

        return out

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
                out[i] = z[i] - self.delta[i] * 2 * zs / (1 + k * zs ** 2)

        return np.array(out)

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
        return xval[self.si]

    def dphi(self, xval):
        dphi = np.zeros(len(xval))
        dphi[self.si] = 1
        return dphi

    def _delf(self, t, xval, uval):
        # calculates the jump term assuming the field switches
        # between fplus and fminus at (t, x)
        params = np.concatenate(([t], xval, uval))
        dphi = self.dphi(xval)
        
        # determine if going from f- to f+
        # or vice-versa
        if matmult(dphi[:self.dim],xval[-self.dim:]) > 0:
            fp = self._fplus.func(*params)
            fm = self._fmins.func(*params)
        else:
            fp = self._fmins.func(*params)
            fm = self._fplus.func(*params)

        #out = 2*np.outer(fp-fm, dphi)/np.abs(matmult(fm+fp, dphi))
        #out = np.outer(fp-fm, dphi)/np.abs(matmult(fm, dphi))
        out = np.outer(fp, dphi)/matmult(dphi, fm) \
             #+ np.eye(len(fp)) - np.outer(dphi, dphi)/np.dot(dphi,dphi) 

        #Tracer()()
        #out = np.zeros((2*self.dim, 2*self.dim))
        #for i in range(self.dim):
        #    out[self.si, i] = -M[self.si, i]
        return out


class SinFloor2D(SymSys):
    # two dimensional point mass, sinusoidal floor
    def __init__(self, k=50.0, m=1.0, g=9.8, **kw):
        self.k = k
        self.m = m
        self.g = g
    
        self.dim = 2

        self.z = map(S, ['z0', 'z1'])
        self.p = map(S, ['p0', 'p1'])
        self.q = map(S, ['q0', 'q1'])
        self.x = map(S, ['x0', 'x1', 'x2', 'x3'])
        self.u = map(S, ['u0', 'u1'])

        self.Mq = np.eye(self.dim) * self.m
        self.Mqi = np.eye(self.dim) / self.m

        self.Vq = -self.m * self.g * self.q[1]
        
        # create Ohm and Psi
        self._Ohm = np.array([self.z[0], self.z[1] + sym.sin(self.z[0])])
        self._Psi = np.array([self.q[0], self.q[1] - sym.sin(self.q[0])])

        #self.controller = lambda t, x: [0, 0]
        
        super(SinFloor2D, self).__init__(si=1) 

class FlatFloor2D(SymSys):
    def __init__(self, k=50.0, m=1.0, g=9.8, **kw):
        self.k = k
        self.dim = 2

        self.z = map(S, ['z0', 'z1'])
        self.p = map(S, ['p0', 'p1'])
        self.q = map(S, ['q0', 'q1'])
        self.x = map(S, ['x0', 'x1', 'x2', 'x3'])
        self.u = map(S, ['u0', 'u1'])

        self.Mq = np.eye(self.dim) * m
        self.Mqi = np.eye(self.dim) / m

        self.Vq = -m * g * self.q[1]
        
        # create Ohm and Psi
        self._Ohm = self.z
        self._Psi = self.q
        
        super(FlatFloor2D, self).__init__(si=1) 



if __name__ == "__main__":
    pass
