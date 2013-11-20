import numpy as np
from scipy.linalg import schur
from numpy.linalg import inv
from scipy.integrate import ode

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

    def solve(self, **kwargs):
        BRB = matmult(self.B, inv(self.R), self.B.T)
        M = np.vstack([
            np.hstack([self.A, -BRB]),
            np.hstack([-self.Q, -self.A.T])
        ])
        (L, Z, sdim) = schur(M, sort='lhp')
        U = Z.T

        self.P = matmult(inv(U[0:sdim, 0:sdim]),
                         U[0:sdim, sdim:]).conj().T


class CDRE(object):
# continuous differential ricatti equation and solver

    def __init__(self, tlims, A, B, **kwargs):
        self.tlims = tlims
        self.ta, self.tb = self.tlims
        ta, tb = tlims

        self.dims = DimExtract(A(ta), B(ta))
        n, m = self.dims

        self.A, self.B = A, B

        # get R and Q and Pb from qwargs
        self.Q = kwargs['Q'] if 'Q' in kwargs \
            else lambda t: np.eye(n)
        self.R = kwargs['R'] if 'R' in kwargs \
            else lambda t: np.eye(m)

        # if Pb is not given get it from solving
        # a CARE at the final time
        if 'Pb' in kwargs:
            self.Pb = kwargs['Pb']
        else:
            # obtain P from an algebraic ricatti at tb
            #care = CARE(self.A(ta), self.B(ta), R=self.R(ta),
            #            Q=self.Q(ta))
            #care.solve()
            #self.Pb = care.P
            #self.Pb = np.eye(n)/100.0
            self.Pb = np.zeros((n,n))

        if 'jumps' in kwargs:
            self.jumps = kwargs['jumps']
        else:
            self.jumps = []

    def _Pdot(self, s, P):
        A, B = self.A(-s), self.B(-s)
        R, Q = self.R(-s), self.Q(-s)
        n, m = self.dims

        # rebuild the matrix from the array
        P = P.reshape((n, n))
        # do necessary matrix algebra
        Pd = matmult(P, B, inv(R), B.T, P) \
            - matmult(A.T, P) - matmult(P, A) - Q
        # ravel and multiply by -1 (for backwards integration)
        return -Pd.ravel()

    def solve(self, **kwargs):
        n, m = self.dims
        sa, sb = (-self.ta, -self.tb)

        Pdot = lambda s, P: self._Pdot(s, P)
        solver = ode(Pdot)

        solver.set_integrator('vode', max_step=1e-2, **kwargs)
        solver.set_initial_value(self.Pb.ravel(), sb)

        self._Ptj = Trajectory('P')
        results = [(-sb, self.Pb)]

        while solver.successful() and solver.t < sa + 1e-2:
            solver.integrate(sa, step=True)
            P = solver.y.reshape((n, n))
        
            # find which jumps lie between this time step and the previous one
            # add the corresponding term to b = solver.y + jumpterm
            if self.jumps:
                prevtime = results[-1][0]  # replace with call to _bt.tmin
                for (tj, fj) in self.jumps:
                    if prevtime > tj and tj > -solver.t:
                        #  positive sign because backwards integration
                        P = P + matmult(fj.T, P) + matmult(P, fj)
                        solver.set_initial_value(P.ravel(), solver.t) 
            
            results.append((-solver.t, P))

        for (t, P) in reversed(results):
            self._Ptj.addpoint(t, P=P)

        self._Ptj.interpolate()
        self.P = lambda t: self._Ptj.P(t)


class LQR(CDRE):
# redefine LQR class to behave better (and more generally)

    """
    what we need:
     tlims = (ta, tb) : time interval
     A(t), B(t) : linear system matrices
     xa : initial condition; not needed yet
     dims : optional, dimensions of state and control
     Q(t), S(t), R(t), Qf=Pb : cost function matrices
                            if not provided, default is identity
     NOTE: xa, S(t) not actually implemented
    """

    def __init__(self, tlims, A, B, **kwargs):
        super(LQR, self).__init__(tlims, A, B, **kwargs)

        if 'xa' in kwargs:
            self.xa = kwargs['xa']

        self.Qf = self.Pb
        if 'S' in kwargs:
            self.S = kwargs['S']
        else:
            self.S = lambda t: np.zeros((n, m))

    def solve(self, **kwargs):
        super(LQR, self).solve()
        self._Kt = Trajectory('K')
        for (t, P) in zip(self._Ptj._t, self._Ptj._P):
            K = matmult(inv(self.R(t)), self.B(t).T, P)
            self._Kt.addpoint(t, K=K)

        self._Kt.interpolate()
        self.K = lambda t: self._Kt.K(t)


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
        super(LQ, self).__init__(tlims, A, B, **kwargs)
        # extra stuff
        
        self.q = kwargs['q']
        self.r = kwargs['r']
        self.qf = kwargs['qf']

        self.bdot = lambda s, b: self._bdot(s, b)

        if 'jumps' in kwargs:
            self.jumps = kwargs['jumps']
        else:
            self.jumps = []

    def _bdot(self, s, b):
        A, B = self.A(-s), self.B(-s)
        q, r = self.q(-s), self.r(-s)
        K = self.K(-s)

        # b is already a vector
        bd = matmult(K.T, r) - q - \
            matmult((A - matmult(B, K)).T, b)

        return -bd  # negative for reverse integration

    def solve(self, **kwargs):
        super(LQ, self).solve()
        sa, sb = (-self.ta, -self.tb)
        solver = ode(self.bdot)
        solver.set_integrator('vode', max_step=1e-2, **kwargs)
        solver.set_initial_value(self.qf, sb)

        results = [(-sb, self.qf)]
        while solver.successful() and solver.t < sa + 1e-2:
            solver.integrate(sa, step=True)
            b = solver.y

            # find which jumps lie between this time step and the previous one
            # add the corresponding term to b = solver.y + jumpterm
            if self.jumps:
                prevtime = results[-1][0] # replace with call to _bt.tmin
                for (tj, fj) in self.jumps:
                    if prevtime > tj and tj > -solver.t:
                        # positive sign because backwards integration
                        b = b + matmult(fj.T, b)
                        solver.set_initial_value(b, solver.t) 

            results.append((-solver.t, b))

        self._bt = Trajectory('b')
        for (t, b) in reversed(results):
            self._bt.addpoint(t, b=b)

        self._bt.interpolate()
        self.b = self._bt.b

        self._Ct = Trajectory('C')
        for (t, b) in zip(self._bt._t, self._bt._b):
            C = matmult(inv(self.R(t)),
                        matmult(self.B(t).T, b) + self.r(t))
            self._Ct.addpoint(t, C=C)

        self._Ct.interpolate()
        self.C = lambda t: self._Ct.C(t)


class Controller(object):

    def __init__(self, **kwargs):
        self.ref = kwargs['reference']
        self.n = len(self.ref._x[0])
        self.m = len(self.ref._u[0])
        m, n = self.m, self.n
        self.C = kwargs['C'] if 'C' in kwargs else lambda t: np.zeros(m)
        if 'K' in kwargs:
            self.K = kwargs['K']
        else:
            self.K = lambda t: np.zeros((m, n))

    def __call__(self, t, x):
        return self.ref.u(t) - \
            matmult(self.K(t), x - self.ref.x(t)) - self.C(t)


class DescentDirection(object):
# implements a descent direction, given a quadratic model
# see section 6.3.2 of Elliot Johnson's thesis

    def _xdot(self, t, x):
        u = self._controller(t, x)
        A = self.A(t)
        B = self.B(t)
        return matmult(A, x) + matmult(B, u)

    def solve(self, **kwargs):
        n, m = self.dims

        # make a zero trajectory
        zeroRef = Trajectory('x', 'u')
        zeroRef.addpoint(self.ta, x=np.zeros(n), u=np.zeros(m))
        #zeroRef.addpoint((2*self.ta + self.tb)/3, x=np.zeros(n), u=np.zeros(m))
        #zeroRef.addpoint((self.ta+2*self.tb)/3, x=np.zeros(n), u=np.zeros(m))
        zeroRef.addpoint(self.tb, x=np.zeros(n), u=np.zeros(m))
        zeroRef.interpolate()

        self._controller = Controller(reference=zeroRef,
                                      K=self.lq.K, C=self.lq.C)

        xdot = lambda t, x: self._xdot(t, x)
        (t, x, jumps) = sysIntegrate(xdot, self.dx0, tlims=self.tlims,
                                    jumps=self.jumps)
        tj = Trajectory('x', 'u')
        for (tt, xx) in zip(t, x):
            tj.addpoint(tt, x=xx, u=self._controller(tt, xx))

        tj.interpolate()
        tj.tlims = self.tlims
        self.direction = tj


class GradDirection(DescentDirection):

    def __init__(self, tlims, A, B, **kwargs):
        self.tlims = tlims
        self.ta, self.tb = self.tlims
        ta, tb = tlims

        self.dims = DimExtract(A(ta), B(ta))
        n, m = self.dims

        self.A, self.B = A, B

        self.q = kwargs['q']
        self.r = kwargs['r']
        self.qf = kwargs['qf']

        self.jumps = kwargs['jumps'] if 'jumps' in kwargs else []

        # set initial condition to zero if nothing is passed
        self.dx0 = kwargs['dx0'] if 'dx0' in kwargs else np.zeros(n)

        self.lq = LQ(tlims, A, B, **kwargs)
        self.lq.solve()
        # don't need to set R, Q, Qf, S because set to I
        # and 0 by default
