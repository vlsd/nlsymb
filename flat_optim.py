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
        xlist = np.linspace(*xlims, num=200)
        bound, = ax.plot(xlist, np.sin(xlist), color='red', lw=2)

    ax = fig.gca()
    q = np.array(tj._q).T
    ax.plot(q[0], q[1])
    fig.show()
    ax.redraw_in_frame()
    return fig

def quickPlot():
    fig = TPlot(ref)
    #TPlot(itj, fig=fig)
    for tj in trajectories[-3:]:
        tj.xtoq(s)
        TPlot(tj, fig=fig)

    return fig

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time
    import pickle
    
    # the following lines are in order to be able to reload nlsymb in ipython
    # dreload(nlsymb, excludes)
    from IPython.lib.deepreload import reload as dreload
    excludes = ['time', 'pickle', 'matplotlib.pyplot', 'sys', '__builtin__', '__main__', 'numpy', 'scipy', 'matplotlib', 'os.path', 'sympy', 'scipy.integrate', 'scipy.interpolate', 'nlsymb.sympy', 'nlsymb.numpy', 'nlsymb.scipy', 'nlsymb.copy', 'copy', 'nlsymb.time']
    
    tlims = (1.7, 2.7)

    """
    t = np.linspace(0, 10, 100)
    x = map(ref.x, t)
    u = map(ref.u, t)
    """

    with Timer("whole program"):
        with Timer("creating symbolic system"):
            s = SymSys(k=5)

        # load the reference (target) trajectory
        ref_file = open('openlooptj.pkl', 'rb')
        ref = pickle.load(ref_file)
        ref_file.close()
        ref.xtoq(s)
        ref.interpolate()
        
        # make an initial guess trajectory
        qinit = np.array([0.0, 10.0])
        qdoti = np.array([-1, 0.0])


        xinit = np.concatenate((s.Psi(qinit),
                                np.dot(s.dPsi(qinit), qdoti)))
        
        itj = Trajectory('x','u')
        #tmid = (tlims[0] + tlims[1])/2
        itj.addpoint(tlims[0], x=ref.x(tlims[0]), u=ref.u(tlims[0]))
        #itj.addpoint(tlims[0], x=ref.x(tlims[0])*1.1, u=ref.u(tlims[0]))
        #itj.addpoint(1.5, x=ref.x(1.5), u=ref.u(1.5))
        itj.addpoint(tlims[1], x=ref.x(tlims[1]), u=np.array([1.0,0.0]))
        itj.xtoq(s)
        itj.interpolate()
        
    
        nlsys = System(s.f, tlims=tlims, xinit=itj.x(tlims[0]),
                       dfdx=s.dfdx, dfdu=s.dfdu)
        nlsys.phi = s.phi
        nlsys.ref = ref

        Rcost = lambda t: np.diag([1, 1])
        Qcost = lambda t: np.diag([10, 10, 1, 1])

        PTcost = Qcost(2)
        
        #zerocontrol = Controller(reference=ref)
        #nlsys.set_u(zerocontrol)

        trajectories = []
        with Timer("initial projection and descent direction"):
            tj = nlsys.project(itj,lin=True)
            trajectories.append(tj)
       
            cost = nlsys.build_cost(R=Rcost, Q=Qcost, PT=PTcost)
            descdir = DescentDir(tj, ref, tlims=tlims, cost=cost)
            print("cost of trajectory before descent: %f" %
                  cost(tj))
            ddircost = descdir.cost
            print("cost of descent direction: %f" % 
                  ddircost)

        index = 0
        ls = None
        while ddircost > 1e-2 :
            index = index+1

            with Timer("descent direction and line search "):
                if index is not 1:
                    cost = nlsys.build_cost(R=Rcost, Q=Qcost, PT=PTcost)
                    descdir = DescentDir(tj, ref, tlims=tlims, cost=cost)
                    print("cost of trajectory before descent: %f" %
                          cost(tj))
                    ddircost = descdir.cost
                    print("cost of descent direction: %f" % 
                          ddircost)

                if ls is None:
                    alpha = 100/ddircost
                else:
                    alpha = ls.gamma*2
                ls = LineSearch(cost, cost.grad, alpha=alpha)
                ls.x = tj
                ls.p = descdir
                ls.search()
                
                tj = tj + ls.gamma * descdir
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


