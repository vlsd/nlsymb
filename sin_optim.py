# this is written for python2.7
# will not work with python3.3
# TODO figure out why!?

import sys
import numpy as np
import sympy as sym
from sympy import Symbol as S

import nlsymb
# nlsymb = reload(nlsymb)

from nlsymb import Timer, LineSearch, np, colored
from nlsymb.sys import *
from nlsymb.lqr import *


# coming soon to a theatre near you
# DoublePlot


def DPlot(tj, s, fig=None, clear=False,
          xlims=(-2.6, 0.2), ylims=(-1.6, 1.1), label="",
          **kwargs):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure()
        # rect = 0.15, 0.1, 0.7, 0.3
        axl = fig.add_subplot(121, aspect='equal', xlim=xlims, ylim=ylims,
                              xlabel="$x(m)$", ylabel="$y(m)$",
                              title='(a)')
        axr = fig.add_subplot(122, aspect='equal', xlim=xlims, ylim=ylims,
                              xlabel=r"$\bar{x}$", ylabel=r"$\bar{y}$",
                              title='(b)')
        xlist = np.linspace(*xlims, num=200)
        bound = axl.fill_between(xlist, ylims[0], np.sin(xlist),
                                 facecolor='grey', alpha=0.5)
        bound = axr.fill_between(xlims, ylims[0], 0.0,
                                 facecolor='grey', alpha=0.5)
        philbl = axl.text(-6, -4, "$\phi(q)<0$")
        psilbl = axr.text(-6, -4, r"$\bar{\phi}(\bar{q})<0$")

    [axl, axr] = fig.get_axes()

    tj.xtoq(s)
    q = np.array(tj._q).T
    qb = np.array(map(s.Psi, tj._q)).T

    tj.xtonq(s)
    z = np.array(tj._q).T
    zb = np.array(map(s.Psi, tj._q)).T

    axl.plot(q[0], q[1], 'b-', label='q' + label, **kwargs)
    axl.plot(z[0], z[1], 'r--', label='z' + label, **kwargs)

    axr.plot(qb[0], qb[1], 'b-', label='qb' + label, **kwargs)
    axr.plot(zb[0], zb[1], 'r--', label='zb' + label, **kwargs)
    fig.show()
    # ax.redraw_in_frame()
    return fig

# plots a trajectory on the given canvas


def TPlot(tj, s, fig=None, ax=None, init=False,
          #xlims=(-3.1, 0.2), ylims=(-1.6, 1.1), label="",
          xlims=(-3.1, 0.2), ylims=(-1.6, 1.1), label="",
          **kwargs):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.figure(figsize=(4, 4))
        # rect = 0.15, 0.1, 0.7, 0.3
        ax = fig.gca(aspect='equal', xlim=xlims, ylim=ylims,
                     xlabel="$x(m)$", ylabel="$y(m)$")
        xlist = np.linspace(*xlims, num=200)
        bound = ax.fill_between(xlist, ylims[0], np.sin(xlist),
                                facecolor='grey', alpha=0.5)
        philbl = ax.text(-1, -1, "$\phi(q)<0$")

    if ax is None:
        ax = fig.gca()

    if init is not False:
        ax.set(aspect='equal', xlim=xlims, ylim=ylims,
               xlabel="$x(m)$", ylabel="$y(m)$")
        xlist = np.linspace(*xlims, num=200)
        ax.fill_between(xlist, ylims[0], np.sin(xlist),
                        facecolor='grey', alpha=0.5)
        ax.text(-1, -1, "$\phi(q)<0$")

    tj.xtoq(s)
    q = np.array(tj._q).T
    tj.xtonq(s)
    z = np.array(tj._q).T
    ax.plot(z[0], z[1], '--', label='z' + label, **kwargs)
    ax.plot(q[0], q[1], '-', label='q' + label, lw=1.5, **kwargs)
    fig.show()
    plt.draw()
    return fig


def quickPlot():
    fig = TPlot(ref)
    # TPlot(itj, fig=fig)
    for tj in trajectories:
        tj.xtonq(s)
        TPlot(tj, fig=fig)

    return fig

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time
    import pickle

    # the following lines are in order to be able to reload nlsymb
    # in ipython
    # dreload(nlsymb, excludes)
    from IPython.lib.deepreload import reload as dreload
    excludes = ['time', 'pickle', 'matplotlib.pyplot', 'sys',
                '__builtin__', '__main__', 'numpy', 'scipy',
                'matplotlib', 'os.path', 'sympy', 'scipy.integrate',
                'scipy.interpolate', 'nlsymb.sympy', 'nlsymb.numpy',
                'nlsymb.scipy', 'nlsymb.copy', 'copy', 'nlsymb.time',
                'scipy.linalg', 'numpy.linalg']

    # load the reference (target) trajectory
    ref_fn = sys.argv[1]
    ref_file = open(ref_fn, 'rb')
    ref = pickle.load(ref_file)
    ref.feasible = False # let's not assume feasibility
    ref_file.close()


    # ref.tlims might be all jacked, lemme fix it first
    #ref.tlims = (min(ref._t), max(ref._t))
    #tlims = ref.tlims
    tlims = (0, 1.9)
    ta, tb = tlims

    """
    t = np.linspace(0, 10, 100)
    x = map(ref.x, t)
    u = map(ref.u, t)
    """

    with Timer("whole program"):
        with Timer("creating symbolic system"):
            #s = FlatFloor2D(k=3)
            s = SinFloor2D(k=3)

        # ref.xtonq(s)
        ref.interpolate()
        ref.tlims = tlims

        if len(sys.argv)>2:
            # initial trajectory was passed to us, use it
            init_file = open(sys.argv[2], 'rb')
            itj = pickle.load(init_file)
            init_file.close()
            itj.feasible = False # let's not assume feasibility
            if not hasattr(itj, 'jumps'):
                itj.jumps=[]
        else:
            # make an initial guess trajectory
            qinit = np.array([0.0, 1.0])
            qdoti = np.array([0.0, 0.0])

            xinit = np.concatenate((s.Psi(qinit),
                                    np.dot(s.dPsi(qinit), qdoti)))

            itj = Trajectory('x', 'u')
            #tmid1 = (2*tlims[0] + tlims[1])/3
            #tmid2 = (tlims[0] + 2*tlims[1])/3
            itj.addpoint(tlims[0], x=xinit, u=np.array([0.0, 0.0]))
            #itj.addpoint(tmid1,    x=ref.x(tmid1),    u=np.array([0.0, 0.0]))
            #itj.addpoint(tmid2,    x=ref.x(tmid2),    u=np.array([0.0, 0.0]))
            # itj.addpoint(tlims[0], x=ref.x(tlims[0])*1.1, u=ref.u(tlims[0]))
            # itj.addpoint(1.5, x=ref.x(1.5), u=ref.u(1.5))
            itj.addpoint(tlims[1], x=xinit, u=np.array([0.0, 0.0]))
            itj.jumps=[]

        itj.xtoq(s)
        itj.interpolate()

        nlsys = System(s.f, tlims=tlims, xinit=itj.x(tlims[0]),
                       dfdx=s.dfdx, dfdu=s.dfdu)
        nlsys.phi = s.phi
        nlsys.ref = ref
        nlsys.delf = s.delf

        Rcost = lambda t: np.diag([5, 5])
        Qcost = lambda t: t*np.diag([100, 200, 1, 1])/tb
        #Qcost = lambda t: t*np.diag([10, 10, 1, 1])

        PTcost = np.diag([0,0,0,0])
        #PTcost = Qcost(tb)

        # zerocontrol = Controller(reference=ref)
        # nlsys.set_u(zerocontrol)

        trajectories = []
        costs = []
        gradcosts = []
        with Timer("initial projection and descent direction"):
            tj = nlsys.project(itj, lin=True)
            trajectories.append(tj)

            cost = nlsys.build_cost(R=Rcost, Q=Qcost, PT=PTcost)
            q = lambda t: matmult(tj.x(t) - ref.x(t), Qcost(t))
            r = lambda t: matmult(tj.u(t) - ref.u(t), Rcost(t))
            qf = matmult(tj.x(tb) - ref.x(tb), PTcost)

            descdir = GradDirection(tlims, tj.A, tj.B, jumps=tj.jumps,
                                    q=q, r=r, qf=qf)
            descdir.solve()

            costs.append(cost(tj))
            print("[initial cost]\t\t" +
                  colored("%f" % costs[-1], 'red'))

            ddir = descdir.direction
            ddircost = cost(ddir, tspace=True)
            gradcosts.append(ddircost)
            print("[descent direction]\t" + colored("%f" % ddircost, 'yellow'))

        index = 0
        ls = None
        while ddircost > 1e-3:
            index = index + 1

            with Timer("line search "):
                if index is not 1:
                    costs.append(cost(tj))
                    print("[cost]\t\t" + colored("%f" % costs[-1], 'blue'))

                    ddir = descdir.direction
                    ddircost = cost(ddir, tspace=True)
                    gradcosts.append(ddircost)
                    print("[descent direction]\t" +\
                          colored("%f" % ddircost, 'yellow'))

                if ls is None:
                    alpha = max(1 / ddircost, 1e-3)
                else:
                    alpha = ls.gamma * 10
                ls = LineSearch(cost, cost.grad, alpha=alpha, beta=1e-8)
                ls.x = tj
                ls.p = descdir.direction
                ls.search()

                tj = tj + ls.gamma * descdir.direction
                # print("cost of trajectory after descent: %f" % cost(tj))

            with Timer("second projection"):
                tj = nlsys.project(tj, tlims=tlims, lin=True)
                trajectories.append(tj)
            with Timer("saving trajectory to file"):
                ofile = open('pkl/sin_plastic_opt_tj.p','wb')
                pickle.dump(tj, ofile)
                ofile.close()
                                            
            cost = nlsys.build_cost(R=Rcost, Q=Qcost, PT=PTcost)
            q = lambda t: matmult(tj.x(t) - ref.x(t), Qcost(t))
            r = lambda t: matmult(tj.u(t) - ref.u(t), Rcost(t))
            qf = matmult(tj.x(tb) - ref.x(tb), PTcost)

            with Timer("descent direction"):
                descdir = GradDirection(tlims, tj.A, tj.B, jumps=tj.jumps,
                                    q=q, r=r, qf=qf)
                descdir.solve()


    # tjt = tj

    # qref = [s.xtopq(ref.x(t)) for t in tjt._t]
    # q0 = map(s.xtopq, trajectories[0]._x)
    # qnu = map(s.xtopq, tjt._x)

    # plt.plot([qq[0] for qq in q0],
    #         [np.sin(qq[0]) for qq in q0])
    # plt.plot([qq[0] for qq in qref], [qq[1] for qq in qref])
    # plt.plot([qq[0] for qq in q0], [qq[1] for qq in q0])
    # plt.plot([qq[0] for qq in qnu], [qq[1] for qq in qnu])
    # plt.axis('equal')
    # plt.show()
