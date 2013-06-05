import numpy as np
from aux import matmult, trajectory
from scipy.linalg import schur
from numpy.linalg import inv


class LQR:
    # a class representing a LQR problem
    def __init__(self, A, B, P=None, Q=None, R=None, tlims=(0, 10)):
        attribs = {'Q': Q, 'R': R, 'Pf': P, 'A': A, 'B': B, 'tlims': tlims}
        for k, v in attribs.iteritems():
            setattr(self, k, v)

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
            self.Q = lambda t: np.eye(self.n)

        if R is None:
            self.R = lambda t: 10*np.eye(self.m)

        if P is None:
            self.Pf = np.eye(self.n)

        # this is a trajectory object
        self.P = self.cdre()

        self.K = trajectory('K')
        for (t, P) in zip(self.P._t, self.P.P.y):
            K = matmult(inv(self.R(t)), self.B(t).T, P)
            self.K.addpoint(t, K=K)
        self.K.interpolate()

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

        P0 = self.care(tend)
        n = self.n

        from numpy.linalg import inv

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
        # backwards differentiation
        from scipy.integrate import ode
        solver = ode(Pdot)
        solver.set_integrator('vode', max_step=1e-1)
        solver.set_initial_value(P0.ravel(), s0)
        t = [-s0]
        P = [P0]

        while solver.successful() and solver.t < send:
            solver.integrate(send, step=True)
            P.append(solver.y.reshape((n, n)))
            t.append(-solver.t)

        ptraj = trajectory('P')
        for tt, pp in zip(t, P):
            # if this is slow, then import itertools and use izip
            ptraj.addpoint(tt, P=pp)
        ptraj.interpolate()

        return ptraj
