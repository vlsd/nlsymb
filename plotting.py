# takes a list of costs and a list of gradient norms
def MakeConvFig(cost, grad, **kwargs):
    # setting up
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(4, 4)) 
    axl = fig.gca()
    axr = axl.twinx()

    cformat = 'bo-'
    gformat = 'rs--'
    kwargs.update({'lw': 1.5})

    axl.plot(cost, cformat, label="$J(x_i,u_i)$", **kwargs)
    axl.set(ylabel="$J(x_i,u_i)$")
    axl.yaxis.label.set_color('blue')
    axl.tick_params(axis='y', colors='blue')

    axr.plot(grad, gformat, label="$||\\nabla J(x_i,u_i)||^2$", **kwargs)
    axr.set(ylabel="$||\\nabla J(x_i,u_i)||^2$")
    axr.yaxis.label.set_color('red')
    axr.tick_params(axis='y', colors='red')

    axl.set_xlabel('iteration #')

    # ask matplotlib for the plotted objects and their labels
    linesr, labelsr = axr.get_legend_handles_labels()
    linesl, labelsl = axl.get_legend_handles_labels()
    axr.legend(linesr + linesl, labelsr + labelsl, loc=0)

    plt.draw()

    return fig

# plots one trajectory on an axis
def tjPlot(tj, s, ax,
          xlims=(-3.1, 0.2), ylims=(-1.6, 1.1), label="",
          **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    ax.set(aspect='equal', xlim=xlims, ylim=ylims, xlabel="$x(m)$")
    xlist = np.linspace(*xlims, num=200)
    bound = ax.fill_between(xlist, ylims[0], np.sin(xlist),
                            facecolor='grey', alpha=0.5)
    philbl = ax.text(-1, -1, "$\phi(q)<0$")

    tj.xtoq(s)
    q = np.array(tj._q).T
    tj.xtonq(s)
    z = np.array(tj._q).T
    ax.plot(z[0], z[1], '--', label='z' + label, **kwargs)
    ax.plot(q[0], q[1], '-', label='q' + label, lw=1.5, **kwargs)
    #fig.show()
    #plt.draw()
    #return fig



# makes figure 2 in the acc 2014 paper
# ref is the reference trajectory
# trajectories is a list of trajectories
# iterations is an associated list of the iteration number
def makeTriptic(ref, s, trajectories, iterations, **kwargs):
    n = len(trajectories)
    
    #setting up
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=n, sharey=True, figsize=(5*n,4))
    
    ax[0].set(ylabel="$y(m)$")
    for i in range(n):
        tjPlot(ref, s, ax=ax[i])
        tjPlot(trajectories[i], s, ax[i])
        ax[i].text(-2.7, -1.3, "$i = %d$" % iterations[i])
        ax[i].text(-0.1, 0.7, "$t_a$")
        ax[i].text(-2.2, 0.0, "$t_b$")
        ax[i].set(title='(%c)' % chr(i+ord('a')))


    plt.draw()
    return fig, ax
