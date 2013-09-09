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

# plots a trajectory on the given canvas
def TPlot(tj, fig=None, xlims=(-7,7), clear=False):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure()
        #rect = 0.15, 0.1, 0.7, 0.3
        ax = fig.gca(aspect='equal')
        xlist = np.linspace(*xlims)
        bound, = ax.plot(xlist, np.sin(xlist), color='red', lw=2)

    ax = fig.gca()
    q = np.array(tj._q).T
    ax.plot(q[0], q[1])
    fig.show()
    return fig

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import pickle

    tlims = (0, 2)

    """
    t = np.linspace(0, 10, 100)
    x = map(ref.x, t)
    u = map(ref.u, t)
    """

    with Timer("whole program"):
        with Timer("creating symbolic system"):
            s = SymSys(k=10)

        qinit = np.array([0, 10])
        qdoti = np.array([2, 0])
        xinit = np.concatenate((s.Psi(qinit),
                                np.dot(s.dPsi(qinit), qdoti)))

        
        ref_file = open('openlooptj.pkl', 'rb')
        ref = pickle.load(ref_file)
        ref_file.close()
        ref.interpolate()
        
        """
        # make a feasible reference trajectory
        qinit = np.array([0, 10])
        qdoti = np.array([2, 0])
        xinit = np.concatenate((s.Psi(qinit),
                                np.dot(s.dPsi(qinit), qdoti)))
        """

        nlsys = System(s.f, tlims=tlims, xinit=xinit,
                       dfdx=s.dfdx, dfdu=s.dfdu)
        nlsys.phi = s.phi
        nlsys.ref = ref

        with Timer("building cost function"):
            Rcost = lambda t: np.diag([1e-2, 1])
            Qcost = lambda t: np.diag([10, 10, 1e-3, 1e-3])
            PTcost = Qcost(2)
            cost = nlsys.build_cost(R=Rcost, Q=Qcost, PT=PTcost)
        
        #zerocontrol = Controller(reference=ref)
        #nlsys.set_u(zerocontrol)

        trajectories = []
        with Timer("initial projection and descent direction"):
            tj = nlsys.project(ref,lin=True)
            trajectories.append(tj)
        
            descdir = DescentDir(tj, ref, tlims=tlims, cost=cost)
            print("cost of trajectory before descent: %f" %
                  cost(tj))
            ddircost = descdir.cost
            print("cost of descent direction: %f" % 
                  ddircost)

        index = 0
        while ddircost > 1e-5 :
            index = index+1

            with Timer("descent direction and line search "):
                if index is not 1:
                    descdir = DescentDir(tj, ref, tlims=tlims, cost=cost)
                    print("cost of trajectory before descent: %f" %
                          cost(tj))
                    ddircost = descdir.cost
                    print("cost of descent direction: %f" % 
                          ddircost)

                ls = LineSearch(cost, cost.grad, alpha=1e-2)
                ls.x = tj
                ls.p = descdir
                ls.search()
                
                tj += ls.gamma * descdir
                print("cost of trajectory after descent: %f" %
                    cost(tj))

            with Timer("second projection"):
                tj = nlsys.project(tj, tlims=tlims, lin=True)
                trajectories.append(tj)


    #tjt = tj

    #qref = [s.xtopq(ref.x(t)) for t in tjt._t]
    #q0 = map(s.xtopq, trajectories[0]._x)
    #qnu = map(s.xtopq, tjt._x)

    #plt.plot([qq[0] for qq in q0],
    #         [np.sin(qq[0]) for qq in q0])
    #plt.plot([qq[0] for qq in qref], [qq[1] for qq in qref])
    #plt.plot([qq[0] for qq in q0], [qq[1] for qq in q0])
    #plt.plot([qq[0] for qq in qnu], [qq[1] for qq in qnu])
    #plt.axis('equal')
    #plt.show()



