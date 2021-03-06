import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import shape
import numpy as np
import pickle
import sys


def init():
    qmass.set_data([], [])
    qtraj.set_data([], [])
    zmass.set_data([], [])
    ztraj.set_data([], [])

    qqmass.set_data([], [])
    qqtraj.set_data([], [])
    zzmass.set_data([], [])
    zztraj.set_data([], [])

    #qxforce = axl.arrow(0, 0, 0, 0)

    return qmass, qtraj, zmass, ztraj 


def animate(i):
    t = float(i) / rate
    newPq = s.xtopq(atj.x(t))
    newqq = s.xtoq(atj.x(t))
    uval = atj.u(t)

    qmass.set_data(newPq)
    qtraj.set_data(qtraj.get_xdata() + [newPq[0]],
                   qtraj.get_ydata() + [newPq[1]])

    qqmass.set_data(newqq)
    qqtraj.set_data(qqtraj.get_xdata() + [newqq[0]],
                    qqtraj.get_ydata() + [newqq[1]])

    for artist in axl.artists:
        artist.remove()
    if uval[0] != 0  and uval[1] != 0:
        qxforce = axl.arrow(newPq[0], newPq[1], 0.5*uval[0], 0.5*uval[1],
                            length_includes_head=True, width=0.01)
    else:
        qxforce = None

    newPz = s.xtopz(atj.x(t))
    newzb = s.xtoz(atj.x(t))

    zmass.set_data(newPz)
    ztraj.set_data(ztraj.get_xdata() + [newPz[0]],
                   ztraj.get_ydata() + [newPz[1]])

    zzmass.set_data(newzb)
    zztraj.set_data(zztraj.get_xdata() + [newzb[0]],
                    zztraj.get_ydata() + [newzb[1]])

    return qmass, qtraj, zmass, ztraj, qqmass, qqtraj, zzmass, zztraj, qxforce


if __name__ == "__main__":
    # parse command line arguments
    # first is the reference trajectory
    infile = open(sys.argv[1], 'rb')
    rtj = pickle.load(infile)
    infile.close()
    rtj.interpolate()

    # now load the animation trajectory
    if len(sys.argv) > 2:
        infile = open(sys.argv[2], 'rb')
        atj = pickle.load(infile)
        infile.close()
    atj.interpolate()

    tmin = atj._t[0]
    tmax = atj._t[-1]
    rate = 30.0  # in frames per second
    
    fig = plt.figure(figsize=(8, 4.5))  # 16:9 ratio
    xmin = -10# -3 * np.pi
    xmax = 2 #np.pi
    ymin = -6
    ymax = 6
    axl = fig.add_subplot(121, xlim=(xmin, xmax), ylim=(ymin, ymax),
                          aspect='equal', xlabel="$x(m)$",
                          ylabel="$y(m)$",
                          title='Euclidean Space')
    # axl.tick_params(pad=-20)

    axr = fig.add_subplot(122, xlim=(xmin, xmax), ylim=(ymin, ymax),
                          aspect='equal', xlabel=r'$\bar{x}$',
                          ylabel=r'$\bar{y}$',
                          title='Modified Space')
    # axr.set_yticklabels([])
    # axr.set_xticklabels([])

    xarray = np.linspace(xmin, xmax, 100)

    qmass, = axl.plot([], [], 'bo', ms=6)
    qqmass, = axl.plot([], [], 'ro', ms=6)
    qqtraj, = axl.plot([], [], 'r--', lw=1)
    qtraj, = axl.plot([], [], 'b-', lw=1)
    qxforce = axl.arrow(0, 0, 1, 1)

    sinfloor = axl.fill_between(xarray, ymin, np.sin(xarray),
                                facecolor='grey', alpha=0.5)
    sinlabel = axl.text(-6, -4, r"$\phi(q)<0$")
    rtoq = s.xtoq(rtj.x(tmin))
    axl.plot(rtoq[0], rtoq[1], 'bx', lw=6)


    flatfloor = axr.fill_between(xarray, ymin, 0 * xarray,
                                 facecolor='grey', alpha=0.5)
    flatlabel = axr.text(-6, -4, r'$\bar{\phi}(\bar{q})<0$')
    rtoz = s.xtoz(rtj.x(tmin))
    axr.plot(rtoz[0], rtoz[1], 'bx', lw=6)


    zmass, = axr.plot([], [], 'bo', ms=6)
    zzmass, = axr.plot([], [], 'ro', ms=6)
    zztraj, = axr.plot([], [], 'r--', lw=1)
    ztraj, = axr.plot([], [], 'b-', lw=1)


    ani = animation.FuncAnimation(fig, animate,
                                  frames=int(rate * (tmax - tmin)),
                                  interval=1000 / (rate), blit=True,
                                  init_func=init, repeat=True)

    # dpi = 720/height in inches for 720p output
    ani.save('sim_real_time.mp4', fps=30, bitrate=7500, dpi=160)

    plt.close()
