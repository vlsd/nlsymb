# takes a list of costs and a list of gradient norms
def MakeConvFig(cost, grad, **kwargs):
    # setting up
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(4,4))
    axl = fig.gca()
    axr = axl.twinx()

    cformat = 'bo-'
    gformat = 'rs--'
    kwargs.update({'lw': 1.5})
    
    
    axl.plot(cost, cformat, label="cost", **kwargs)
    axl.set(ylabel="$J(x_i,u_i)$")
    
    axr.plot(grad, gformat, label="grad", **kwargs)
    axr.set(ylabel="$||\\nabla J(x_i,u_i)||^2$")

    axl.set_xlabel('iteration #')
    plt.draw()

    return fig
