# this is written for python2.7
# will not work with python3.3
# TODO figure out why!?

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
          xlims=(-1, 3), ylims=(-2, 2), label="",
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

    tlims = (0, 3)
    ta, tb = tlims

    """
    t = np.linspace(0, 10, 100)
    x = map(ref.x, t)
    u = map(ref.u, t)
    """

    with Timer("whole program"):
        with Timer("creating symbolic system"):
            #s = FlatFloor2D(k=3)
            s = SinFloor2D(k=30, g=0.0)

        # pick an initial point and velocity
        qinit = np.array([0.5, 1.5])
        qdoti = np.array([0.0, -1.0])

        xinit = np.concatenate((s.Psi(qinit),
                                np.dot(s.dPsi(qinit), qdoti)))

        nlsys = System(s.f, tlims=tlims, xinit=xinit,
                       dfdx=s.dfdx, dfdu=s.dfdu, algebra=s.P)
        nlsys.phi = s.phi
        nlsys.delf = s.delf

        zerocontrol = lambda t,x: np.array([0,0])
        nlsys.set_u(zerocontrol)

        # integrate, but don't linearize
        tj = nlsys.integrate(linearize=False)


