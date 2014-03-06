import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import shape
import numpy as np


def init():
    qmass.set_data([], [])
    qtraj.set_data([], [])
    zmass.set_data([], [])
    ztraj.set_data([], [])

    qqmass.set_data([], [])
    qqtraj.set_data([], [])
    zzmass.set_data([], [])
    zztraj.set_data([], [])

    return qmass, qtraj, zmass, ztraj


def animate(i):
    t = float(i) / rate
    newPq = s.xtopq(tj.x(t))
    newqq = s.xtoq(tj.x(t))

    qmass.set_data(newPq)
    qtraj.set_data(qtraj.get_xdata() + [newPq[0]],
                   qtraj.get_ydata() + [newPq[1]])

    qqmass.set_data(newqq)
    qqtraj.set_data(qqtraj.get_xdata() + [newqq[0]],
                    qqtraj.get_ydata() + [newqq[1]])

    newPz = s.xtopz(tj.x(t))
    newzb = s.xtoz(tj.x(t))

    zmass.set_data(newPz)
    ztraj.set_data(ztraj.get_xdata() + [newPz[0]],
                   ztraj.get_ydata() + [newPz[1]])

    zzmass.set_data(newzb)
    zztraj.set_data(zztraj.get_xdata() + [newzb[0]],
                    zztraj.get_ydata() + [newzb[1]])

    return qmass, qtraj, zmass, ztraj, qqmass, qqtraj, zzmass, zztraj


if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 4.5))  # 16:9 ratio
    xmin = -3 * np.pi
    xmax = np.pi
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
    sinfloor = axl.fill_between(xarray, ymin, np.sin(xarray),
                                facecolor='grey', alpha=0.5)
    sinlabel = axl.text(-6, -4, r"$\phi(q)<0$")

    flatfloor = axr.fill_between(xarray, ymin, 0 * xarray,
                                 facecolor='grey', alpha=0.5)
    flatlabel = axr.text(-6, -4, r'$\bar{\phi}(\bar{q})<0$')

    zmass, = axr.plot([], [], 'bo', ms=6)
    zzmass, = axr.plot([], [], 'ro', ms=6)
    zztraj, = axr.plot([], [], 'r--', lw=1)
    ztraj, = axr.plot([], [], 'b-', lw=1)

    tmin = tj._t[0]
    tmax = tj._t[-1]
    rate = 90.0  # in frames per second

    ani = animation.FuncAnimation(fig, animate,
                                  frames=int(rate * (tmax - tmin)),
                                  interval=1000 / (rate), blit=True,
                                  init_func=init, repeat=True)

    # dpi = 720/height in inches for 720p output
    ani.save('sim_super_slow.mp4', fps=30, bitrate=7500, dpi=160)

    plt.close()
