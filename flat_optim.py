# this is written for python2.7
# will not work with python3.3
# TODO figure out why!?

import numpy as np
import sympy as sym
from sympy import Symbol as S

import nlsymb
#nlsymb = reload(nlsymb)

from nlsymb import Timer, LineSearch
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
        with Timer("creating symbolic system"):
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
        nlsys.set_ref(ref)

        #zerocontrol = Controller(reference=ref)
        #nlsys.set_u(zerocontrol)

        trajectories = []
        with Timer("initial projection"):
            #tj = nlsys.integrate()
            #trajectories.append(tj)

        #with Timer("first projection"):
            tj = nlsys.project(ref, lin=True)
            trajectories.append(tj)

        for index in range(10):
            with Timer("descent direction and line search "):
                descdir = DescentDir(tj, ref, tlims=tlims, Rscale=1)
                print("cost of trajectory before descent: %f" %
                      nlsys.cost(tj))

                ls = LineSearch(nlsys.cost, nlsys.grad)
                ls.set_x(tj)
                ls.set_p(descdir)
                ls.search()
                
                tj += ls.gamma * descdir
                print("cost of trajectory after descent: %f" %
                    nlsys.cost(tj))

            with Timer("second projection"):
                tj = nlsys.project(tj, tlims=tlims, lin=True)
                trajectories.append(tj)

    tjt = tj

    qref = [s.xtopq(ref.x(t)) for t in tjt._t]
    q0 = map(s.xtopq, trajectories[0]._x)
    qnu = map(s.xtopq, tjt._x)

    plt.plot([qq[0] for qq in q0],
             [np.sin(qq[0]) for qq in q0])
    plt.plot([qq[0] for qq in qref], [qq[1] for qq in qref])
    plt.plot([qq[0] for qq in q0], [qq[1] for qq in q0])
    plt.plot([qq[0] for qq in qnu], [qq[1] for qq in qnu])
    plt.axis('equal')
    plt.show()
