# this is written for python2.7
# will not work with python3.3
# TODO figure out why!?

import numpy as np
import sympy as sym
from sympy import Symbol as S
from copy import deepcopy

import nlsymb
nlsymb = reload(nlsymb)

from nlsymb import Timer
from nlsymb.sys import *
from nlsymb.lqr import *


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    tlims = (0, 2)

    """
    t = np.linspace(0, 10, 100)
    x = map(ref.x, t)
    u = map(ref.u, t)
    """

    with Timer():
        with Timer():
            s = SymSys(k=10)
            qinit = np.array([0, 10])
            qdoti = np.array([1, -2])
            xinit = np.concatenate((s.Psi(qinit),
                                    np.dot(s.dPsi(qinit), qdoti)))

        ref = Trajectory('x', 'u')
        ref.addpoint(0, x=xinit, u=[0, 0])
        ref.addpoint(2, x=[-6, -7, 0, 0], u=[0, 0])
        ref.interpolate()

        nlsys = System(s.f, tlims=tlims, xinit=xinit,
                       dfdx=s.dfdx, dfdu=s.dfdu)
        nlsys.set_phi(s.phi)

        zerocontrol = Controller(reference=ref)
        nlsys.set_u(zerocontrol)

        trajectories = []
        with Timer():
            lintraj = nlsys.integrate()
            lintraj.interpolate()
            trajectories.append(deepcopy(lintraj))

        with Timer():
            lqrtest = LQR(lintraj.A, lintraj.B, tlims=tlims, Rscale=10)
            # use reference trajectory as initial guess
            # does not have to be the case
            nucontrol = Controller(reference=ref, K=lqrtest.K)

        with Timer():
            nlsys.set_u(nucontrol)
            nutraj = nlsys.integrate()
            nutraj.interpolate()
            trajectories.append(deepcopy(nutraj))

        with Timer():
            print("calculating descent direction")
            descdir = DescentDir(nutraj, ref, tlims=tlims, Rscale=100)
            print("cost of trajectory before descent: %f" %
                  descdir.cost())

        with Timer():
            print("running Armijo")
            ls = LineSearch(descdir.cost, descdir.grad)
            ls.set_x(nutraj)
            ls.set_p(descdir)
            ls.search()

        with Timer():
            print("applying descent direction")
            # this is where the line search goes
            nutraj += ls.gamma * descdir
            print("cost of trajectory after descent: %f" %
                descdir.cost(traj=nutraj))

        with Timer():
            # projecting
            nulqr = LQR(nutraj.A, nutraj.B, tlims=tlims, Rscale=10)
            nucontrol = Controller(reference=nutraj, K=nulqr.K)
            nlsys.set_u(nucontrol)
            nutraj = nlsys.integrate()
            nutraj.interpolate()
            trajectories.append(deepcopy(nutraj))

    qref = [s.xtopq(ref.x(t)) for t in lintraj._t]
    q = map(s.xtopq, lintraj.x.y)
    qnu = map(s.xtopq, nutraj.x.y)

    plt.plot([qq[0] for qq in q],
             [np.sin(qq[0]) for qq in q])
    plt.plot([qq[0] for qq in qref], [qq[1] for qq in qref])
    plt.plot([qq[0] for qq in q], [qq[1] for qq in q])
    plt.plot([qq[0] for qq in qnu], [qq[1] for qq in qnu])
    plt.axis('equal')
    plt.show()
