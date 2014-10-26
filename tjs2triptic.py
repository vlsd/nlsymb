#!/usr/bin/env python2

import nlsymb
from nlsymb import sys as nlsys
import matplotlib.pyplot as plt
import pickle
import sys
from numpy import array, linspace, sin

def load_files(name):

    # load the reference (target) trajectory
    ref_fn = "pkl/ref_%s.p" % name
    ref_file = open(ref_fn, 'rb')
    ref = pickle.load(ref_file)
    ref.feasible = False # let's not assume feasibility
    ref_file.close()

    # load the list of generated trajectories
    tjs_fn = "pkl/tjs_%s.p" % name
    tjs_file = open(tjs_fn, 'rb')
    tjs = pickle.load(tjs_file)
    tjs_file.close()

    return ref, tjs

if __name__ == "__main__":

    # we need this to know what files to load and what
    # files to write out to
    name = sys.argv[1]

    xlims=(-15, 5)
    ylims=(-2, 6) 

    try:
        ref
        tjs
        print "Data already present, not loading from file"
    except NameError:
        print "Loading data from files..."
        ref, tjs = load_files(name)

    # only plot the figures at these indices
    indices = [0, 3, 10]

    try:
        s
        print "Symbolic system exists, not creating it again"
    except NameError:
        print "Creating symbolic system..."
        s = nlsys.SinFloor2D(k=3)
        ref.xtoq(s)

    fig = plt.figure()
    ax1 = fig.add_subplot(
        311, 
        aspect='equal',
        xlim=xlims, ylim=ylims,
        ylabel="$y(m)$",
    )
    ax2 = fig.add_subplot(
        312,
        aspect='equal',
        xlim=xlims, ylim=ylims,
        ylabel=r"$y(m)$",
    )
    ax3 = fig.add_subplot(
        313,
        aspect='equal',
        xlim=xlims, ylim=ylims,
        xlabel="$x(m)$", ylabel="$y(m)$",
    )

    xlist = linspace(*xlims, num=200)

    for index, tj, ax in zip(
        indices,
        [tjs[i] for i in indices],
        fig.get_axes()
    ):
        ax.locator_params(axis='y', nbins=4)
        ax.fill_between(
            xlist,
            ylims[0], sin(xlist),
            facecolor='grey', alpha=0.5
        )

        ax.text(-12.5, -1.5, "$\phi(q)<0$")
        ax.text(1, 3, r"$n=%d$" % index)

        tj.xtoq(s)
        q = array(tj._q).T
        tj.xtonq(s)
        z = array(tj._q).T
        ax.plot(ref._q[0][0], ref._q[0][1], 'x')
        ax.plot(z[0], z[1], '--', label='z' + name)
        ax.plot(q[0], q[1], '-', label='q' + name, lw=1.5)

    fig.show()
    plt.draw()

