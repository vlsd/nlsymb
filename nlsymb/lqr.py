import numpy as np
from scipy.linalg import schur
from numpy.linalg import inv
from scipy.integrate import ode
from scipy.integrate import trapz

from . import matmult, sysIntegrate, Trajectory


# check that the dimensions of A and B are correct and return them
def DimExtract(A, B):
    AA, BB = map(np.array, (A, B))

    if AA.shape[0] is not AA.shape[1]:
        raise Exception("A needs to be square")

    if AA.shape[0] is not BB.shape[0]:
        raise Exception("A and B need to share the first dimension")

    return BB.shape


class CARE(object):
# continuous algebraic ricatti equation and solver

    def __init__(self, A, B, **kwargs):
        self.dims = DimExtract(A, B)
        self.n, self.m = self.dims

        self.A = np.array(A)
        self.B = np.array(B)

        # get R and Q from qwargs
        keys = kwargs.keys()
        self.Q = np.array(kwargs['Q']) if 'Q' in keys \
            else np.eye(self.n)
        self.R = np.array(kwargs['R']) if 'R' in keys \
            else np.eye(self.n)

    def _solve(self):
        BRB = matmult(self.B, inv(self.R), self.B.T)
        M = np.vstack([
            np.hstack([self.A, -BRB]),
            np.hstack([-self.Q, -self.A.T])
        ])
        (L, Z, sdim) = schur(M, sort='lhp')
        U = Z.T

        self._P = matmult(inv(U[0:sdim, 0:sdim]),
                          U[0:sdim, sdim:]).conj().T

        return self._P

    @property
    def P(self):
        if hasattr(self, '_P'):
            return self._P
        else:
            return self.solve()


class CDRE(object):
# continuous differential ricatti equation and solver

    def __init__(self, tlims, A, B, **kwargs):
        self.tlims = tlims
        self.ta, self.tb = self.tlims

        self.dims = DimExtract(A(ta), B(ta))
        n, m = self.dims

        self.A, self.B = A, B

        # get R and Q and Pb from qwargs
        self.Q = kwargs['Q'] if 'Q' in kwargs.keys() \
            else lambda t: np.eye(n)
        self.R = kwargs['R'] if 'R' in kwargs.keys() \
            else lambda t: np.eye(m)

        # if Pb is not given get it from solving
        # a CARE at the final time
        if 'Pb' in kwargs.keys():
            self.Pb = kwargs['Pb']
        else:
            care = CARE(self.A(tb), self.B(tb),
                        R=self.R(tb), Q=self.Q(tb))
            self.Pb = care.P

        self.Pdot = lambda s, P: self._Pdot(s, P)

    def _Pdot(self, s, P):
        A, B = self.A(-s), self.B(-s)
        R, Q = self.R(-s), self.Q(-s)

        # rebuild the matrix from the array
        P = P.reshape((self.n, self.n))
        # do necessary matrix algebra
        Pd = matmult(P, B, inv(R), B.T, P) \
            - matmult(A.T, P) - matmult(P, A) - Q
        # ravel and multiply by -1 (for backwards integration)
        return -Pd.ravel()

    def _solve(self, **kwargs):
        sa, sb = -(self.ta, self.tb)
        solver = ode(self.Pdot)
        solver.set_integrator('vode', **kwargs)
        solver.set_initial_value(self.Pb.ravel(), sb)

        self._Pt = Trajectory('P')
        self._Pt.addpoint(-sb, P=Pb)

        while solver.successful() and solver.t < sa:
            solver.integrate(sa, step=True)
            self.Ptraj.addpoint(-solver.t, solver.y.reshape((n, n)))

        self._Pt.interpolate()
        return self._Pt

    def P(self, t):
        if hasattr(self, '_Pt'):
            return self._Pt.P(t)
        else:
            #                 ,here we add max_step option
            return self.solve().P(t)


class Controller():

    def __init__(self, **kwargs):
        self.ref = kwargs['reference']
        self.n = len(self.ref._x[0])
        self.m = len(self.ref._u[0])
        m, n = self.m, self.n
        self.C = kwargs['C'] if 'C' in kwargs.keys() else np.zeros(m)
        if 'K' in kwargs.keys():
            self.K = kwargs['K']
        else:
            self.K = np.zeros((m, n))

    def __call__(self, t, x):
        return self.ref.u(t) - \
            matmult(self.K(t), x - self.ref.x(t)) - self.C(t)


class LQR(CDRE):
# redefine LQR class to behave better (and more generally)

    """
    what we need:
     tlims = (ta, tb) : time interval
     A(t), B(t) : linear system matrices
     xa : initial condition
     dims : optional, dimensions of state and control
     Q(t), S(t), R(t), Qf=Pb : cost function matrices
                            if not provided, default is identity
     NOTE: S(t) not actually implemented
    """

    def __init__(self, tlims, A, B, xa, **kwargs):
        CDRE.__init__(tlims, A, B, **kwargs)

        self.xa = xa
        self.Qf = self.Pb
        if 'S' in kwargs.keys():
            self.S = kwargs['S']
        else:
            self.S = lambda t: np.zeros((n, m))

    def _solve(self):
        CDRE._solve()
        self._Kt = Trajectory('K')
        for (t, P) in zip(self._Pt._t, self._Pt._P):
            K = matmult(inv(self.R(t)), self.B(t).T, P)
            self._Kt.addpoint(t, K=K)

        self._Kt.interpolate()
        return self._Kt

    def K(self, t):
        if hasattr(self, '_Kt'):
            return self._Kt.K(t)
        else:
            return self._solve().K(t)


class LQ(LQR):
# class that implements an LQ problem and solver

    """
    what we need:
     dims : optional, dimensions of state and control
     tlims = (ta, tb) : time interval
     A(t), B(t) : linear system matrices
     q(t), r(t), qf : linear terms in cost function
     Q(t), S(t), R(t), Qf : quadratic model matrices
     xa : initial condition, not needed unless want to also
            return the optimal cost later on
     jumps : optional, list of pairs (t, f) at which dynamics
            switch and the jump term to be added to q(t) at that time
    """

    def __init__(self, tlims, A, B, **kwargs):
        LQR.__init__(tlims, A, B, **kwargs)
        # extra stuff

        self.q = kwargs['q']
        self.r = kwargs['r']
        self.qf = kwargs['qf']

        self.bdot = lambda s, b: self._bdot(s, b)

        self.jumps = kwargs['jumps'] if 'jump' in kwargs.keys() else []
        self.tjmp, self.fjmp = map(list, zip(*self.jumps))

    def _bdot(self, s, b):
        A, B = self.A(-s), self.B(-s)
        q, r = self.q(-s), self.r(-s)
        K = self.K(-s)

        # b is already a vector
        bd = matmult(K.T, r) - q - \
            matmult((A - matmult(B, K)).T, b)

        return -bd  # negative for reverse integration

    def _solve(self):
        LQR._solve()
        sa, sb = -(self.ta, self.tb)
        solver = ode(self.bdot)
        solver.set_integrator('vode', **kwargs)
        solver.set_initial_value(self.qf, sb)

        self._bt = Trajectory('b')
        self._bt.addpoint(-sb, b=self.qf)

        while solver.successful() and solver.t < sa:
            solver.integrate(sa, step=True)
            b = solver.y

            # find which jumps lie between this time step and the previous one
            # add the corresponding term to b = solver.y + jumpterm
            prevtime = np.min(self._bt._t)  # replace with call to _bt.tmin
            for (tj, fj) in self.jumps:
                if prevtime > tj and tj > -solver.t:
                    b = b + fj * (-solver.t - prevtime)

            self._bt.addpoint(-solver.t, b)

        self._bt.interpolate()
        self.b = self._bt.b

        self._Ct = Trajectory('C')
        for (t, b) in zip(self._bt._t, self._bt._b):
            C = matmult(inv(self.R(t)),
                        matmult(self.B(t).T, b) + self.r(t))
            self._Ct.addpoint(t, C=C)

        self._Ct.interpolate()
        return self.K, self._Ct.C

    def C(self, t):
        if hasattr(self, '_Ct'):
            return self._Ct.C(t)
        else:
            self.solve()
            return self._Ct.C(t)


class DescentDirection(object):
# implements a descent direction, given a quadratic model
# see section 6.3.2 of Elliot Johnson's thesis


    def _xdot(t, x):
            u = self._controller(t, x)
            A = self.A(t)
            B = self.B(t)
            return matmult(A, x) + matmult(B, u)
    
    def _solve(self):
        n, m = self.dims

        # make a zero trajectory
        zeroRef = Trajectory('x', 'u')
        zeroRef.addpoint(self.ta, x=np.zeros(n), u=np.zeros(m))
        zeroRef.addpoint(self.tb, x=np.zeros(n), u=np.zeris(m))
        zeroRef.interpolate()

        self._controler = Controller(reference=zeroRef, \
                                     K = self.lq.K, C = self.lq.C)


        (t, x) = sysIntegrate(self._xdot, self.dx0, tlims=self.tlims)
        tj = Trajectory('x', 'u')
        for (tt, xx) in zip(t, x):
            tj.addpoint(tt, x=xx, u=self._controller(tt, xx))
        tj.interpolate()
        self._direction = tj

        return tj

    @property
    def direction(self):
        if hasattr(self, '_direction'):
            return self._direction
        else:
            return self._solve()
    
    """
    @property
    def cost(self):
        elist = []
        tlist = self._t
        T = self.tmax
        for (t, z, v) in zip(tlist, self._z, self._v):
            expr = matmult(z, self._cost.Q(t), z) + \
                matmult(v, self._cost.R(t), v)
            elist.append(expr)

        out = trapz(elist, tlist) + matmult(
            self.z(T), self._cost.PT, self.z(T))

        return out
    """

class GradDirection(DescentDirection):
    def __init__(self, tlims, A, B, **kwargs):
        self.tlims = tlims
        self.ta, self.tb = self.tlims

        self.dims = DimExtract(A(ta), B(ta))
        n, m = self.dims

        self.A, self.B = A, B

        self.q = kwargs['q']
        self.r = kwargs['r']
        self.qf = wkargs['qf']

        # set initial condition to zero if nothing is passed
        self.dx0 = kwargs['dx0'] if 'dx0' in kwargs.keys() else np.zeros(n)

        self.lq = LQ(tlims, A, B, **kwargs)
        # don't need to set R, Q, Qf, S because set to I
        # and 0 by default



       

