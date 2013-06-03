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
    t = float(i)/rate
    newPq = s.xtopq(lintraj.x(t))
    newqq = s.xtoq(lintraj.x(t))

    qmass.set_data(newPq)
    qtraj.set_data(qtraj.get_xdata() + [newPq[0]],
                   qtraj.get_ydata() + [newPq[1]])

    qqmass.set_data(newqq)
    qqtraj.set_data(qqtraj.get_xdata() + [newqq[0]],
                    qqtraj.get_ydata() + [newqq[1]])

    newPz = s.xtopz(lintraj.x(t))
    newzb = s.xtoz(lintraj.x(t))

    zmass.set_data(newPz)
    ztraj.set_data(ztraj.get_xdata() + [newPz[0]],
                   ztraj.get_ydata() + [newPz[1]])

    zzmass.set_data(newzb)
    zztraj.set_data(zztraj.get_xdata() + [newzb[0]],
                    zztraj.get_ydata() + [newzb[1]])

    return qmass, qtraj, zmass, ztraj, qqmass, qqtraj, zzmass, zztraj


if __name__ == "__main__":
    fig = plt.figure(figsize=(10, 4))
    xmin = -25
    xmax = 5
    ymin = -12
    ymax = 12
    axl = fig.add_subplot(121, xlim=(xmin, xmax), ylim=(ymin, ymax),
                          aspect='equal', xlabel="$x(m)$", ylabel="$y(m)$",
                          title='Euclidean Space')
    axr = fig.add_subplot(122, xlim=(xmin, xmax), ylim=(ymin, ymax),
                          aspect='equal', xlabel='$x_z$', ylabel='$y_z$',
                          title='Modified Space')
    axr.set_yticklabels([])
    axr.set_xticklabels([])

    xarray = np.linspace(xmin, xmax, 100)

    qmass, = axl.plot([], [], 'bo', ms=6)
    qqmass, = axl.plot([], [], 'ro', ms=6)
    qtraj, = axl.plot([], [], 'b--', lw=1)
    qqtraj, = axl.plot([], [], 'r--', lw=1)
    sinfloor = axl.fill_between(xarray, ymin, np.sin(xarray),
                                facecolor='grey', alpha=0.5)
    sinlabel = axl.text((xmax+xmin)/2, -3, "$\phi(q)<0$")

    flatfloor = axr.fill_between(xarray, ymin, 0*xarray,
                                 facecolor='grey', alpha=0.5)
    flatlabel = axr.text(-13, -3.5, "$\psi(q_z)<0$")

    zmass, = axr.plot([], [], 'bo', ms=6)
    zzmass, = axr.plot([], [], 'ro', ms=6)
    ztraj, = axr.plot([], [], 'b--', lw=1)
    zztraj, = axr.plot([], [], 'r--', lw=1)

    tmin = lintraj._t[0]
    tmax = lintraj._t[-1]
    rate = 30.0  # in frames per second

    ani = animation.FuncAnimation(fig, animate, frames=int(rate*(tmax-tmin)),
                                  interval=1000/rate, blit=True,
                                  init_func=init, repeat=True)

    ani.save('sin_floor_2d_sim.mp4',
             fps=30, extra_args=['-vcodec', 'libx264'])

    #plt.show()
