import numpy as np
from scipy.linalg import schur
from numpy.linalg import inv
from scipy.integrate import ode
from scipy.integrate import trapz

from . import matmult, sysIntegrate, Trajectory


class LQR:
    # a class representing a LQR problem
    def __init__(self, A, B, Q=None, R=None, tlims=(0, 10),
                 Rscale=1, Qscale=1, PT=None):
        attribs = {'Q': Q, 'R': R, 'PT': PT, 'A': A, 
                   'B': B, 'tlims': tlims}
        for k, v in attribs.iteritems():
            setattr(self, k, v)

        self.t0 = tlims[0]
        self.tf = tlims[1]

        Adim = np.array(A(tlims[0])).shape
        if Adim[0] != Adim[1]:
            raise NameError('A must be square')
        else:
            self.n = Adim[0]

        Bdim = np.array(B(tlims[0])).shape
        if Bdim[0] != self.n:
            raise NameError("B's first dimension must match A")
        else:
            self.m = Bdim[1]

        if Q is None:
            # define it as identity
            self.Q = lambda t: Qscale*np.eye(self.n)

        if R is None:
            self.R = lambda t: Rscale*np.eye(self.m)

        if PT is None:
            self.PT = np.eye(self.n)
            #self.PT = self.care(self.tf)

        # this is a Trajectory object
        self.Ptraj = self.cdre()
        self.P = self.Ptraj.P

        self.Ktraj = Trajectory('K')
        for (t, P) in zip(self.Ptraj._t, self.Ptraj.P.y):
            K = matmult(inv(self.R(t)), self.B(t).T, P)
            self.Ktraj.addpoint(t, K=K)
        self.Ktraj.interpolate()
        self.K = self.Ktraj.K

    # returns Pbar
    def care(self, t):

        A = self.A(t)
        B = self.B(t)
        Q = self.Q(t)
        R = self.R(t)

        BRB = matmult(B, inv(R), B.T)
        M = np.vstack([
            np.hstack([A, -BRB]),
            np.hstack([-Q, -A.T])
        ])
        (L, Z, sdim) = schur(M, sort='lhp')
        U = Z.T

        return matmult(inv(U[0:sdim, 0:sdim]), U[0:sdim, sdim:]).conj().T

    # A, B, Q, R should be callables of t
    # Returns callable P(t)
    def cdre(self):
        A, B, Q, R = (self.A, self.B, self.Q, self.R)
        t0, tend = self.tlims

        # in order to do backward integration we need to define s=-t
        s0, send = (-tend, -t0)

        PT = self.PT
        n = self.n

        # this implements the riccati equation
        # by flattening the matrix P into a vector
        def Pdot(s, P):
            # rebuild the matrix from the array
            P = P.reshape((n, n))
            # do necessary matrix algebra
            Pd = matmult(P, B(-s), inv(R(-s)), B(-s).T, P) -\
                matmult(A(-s).T, P) - matmult(P, A(-s)) - Q(-s)
            # ravel and multiply by -1 (for backwards integration)
            return -Pd.ravel()

        # leaving this here until I change sysIntegrate to handle
        # backwards integration
        solver = ode(Pdot)
        solver.set_integrator('vode', max_step=1e-1)
        solver.set_initial_value(PT.ravel(), s0)
        t = [-s0]
        P = [PT]

        while solver.successful() and solver.t < send:
            solver.integrate(send, step=True)
            P.append(solver.y.reshape((n, n)))
            t.append(-solver.t)

        ptraj = Trajectory('P')
        for tt, pp in zip(t, P):
            # if this is slow, then import itertools and use izip
            ptraj.addpoint(tt, P=pp)
        ptraj.interpolate()

        return ptraj

    def bintegrate(self, func, shape, tlims):
        # TODO write this to do backwards, matrix integration
        pass


class Controller():
    def __init__(self, **kwargs):
        self.ref = kwargs['reference']
        if 'K' in kwargs.keys():
            self.K = kwargs['K']
            self.zero = False
        else:
            self.zero = True

    def __call__(self, t, x):
        if self.zero:
            return self.ref.u(t)
        else:
            return self.ref.u(t) - np.dot(self.K(t), x - self.ref.x(t))


class DescentDir(LQR):
    def __init__(self, traj, ref, cost=None, **kwargs):
        # not passing R and Q to LQR, getting set to
        # identity by default
        LQR.__init__(self, traj.A, traj.B, **kwargs)

        self._cost = cost
        self.traj = traj
        self.ref = ref
        tf = self.tf
        P1 = cost.PT
        self.rf = np.dot(P1, self.traj.x(tf) - self.ref.x(tf))
        self.a = lambda t: np.dot(cost.Q(t), self.traj.x(t) - self.ref.x(t))
        self.b = lambda t: np.dot(cost.R(t), self.traj.u(t) - self.ref.u(t))

        self.r = self.rsolve()

        if 'z0' in kwargs.keys():
            self.z0 = kwargs['z0']
        else:
            #self.z0 = -np.dot(inv(self.P(0)), self.r(0))
            self.z0 = np.zeros(self.n)

        self.direction = self.solve(self.z0)

        self.z = self.direction.z
        self.v = self.direction.v
        self._z = self.direction._z
        self._v = self.direction._v
        self._t = self.direction._t
        self.tmin = self.direction.tmin
        self.tmax = self.direction.tmax

    def rsolve(self):
        def rdot(s, r):
            t = -s
            A = self.A(t)
            B = self.B(t)
            Rinv = inv(self.R(t))
            P = self.P(t)
            b = self.b(t)
            a = self.a(t)
            out = np.dot((A - matmult(B, Rinv, B.T, P)).T, r)
            out += a - matmult(P, B, Rinv, b)
            # two minus signs make a plus below (one from equation
            # and one from backwards integration)
            return out

        # leaving this here until I change sysIntegrate to handle
        # backwards integration
        solver = ode(rdot)
        solver.set_integrator('vode', max_step=1e-1, min_step=1e-13)
        solver.set_initial_value(self.rf, -self.tf)
        t = [-self.tf]
        r = [self.rf]

        while solver.successful() and solver.t < self.t0:
            solver.integrate(self.t0, step=True)
            r.append(solver.y)
            t.append(-solver.t)

        rtraj = Trajectory('r')
        for tt, rr in zip(t, r):
            rtraj.addpoint(tt, r=rr)
        rtraj.interpolate()

        return rtraj.r

    def solve(self, z0):
        def v(t, z):
            Ri = inv(self.R(t))
            b = self.b(t)
            P = self.P(t)
            r = self.r(t)
            B = self.B(t)
            return -np.dot(Ri, b + matmult(B.T, P, z) + np.dot(B.T, r))

        def zdot(t, z):
            return np.dot(self.A(t), z) + np.dot(self.B(t), v(t, z))

        (t, z) = sysIntegrate(zdot, z0, tlimits=self.tlims)
        lintraj = Trajectory('z', 'v')
        for (tt, zz) in zip(t, z):
            lintraj.addpoint(tt, z=zz, v=v(tt, zz))
        lintraj.interpolate()

        return lintraj

    @property
    def cost(self):
        elist = []
        tlist = self._t
        T = self.tmax
        for (t, z, v) in zip(tlist, self._z, self._v):
            expr = matmult(z, self._cost.Q(t), z) + matmult(v, self._cost.R(t), v)
            elist.append(expr)

        out = trapz(elist, tlist) + matmult(self.z(T), self._cost.PT, self.z(T))

        return out

    def __rmul__(self, scalar):
        # should only be called for a descent direction
        # which only has z and v components
        out = Trajectory('z', 'v')
        for (t, z, v) in zip(self._t, self._z, self._v):
            out.addpoint(t, z=scalar*z, v=scalar*v)

        out.interpolate()
        return out
