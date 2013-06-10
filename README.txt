======
nlsymb
======

nlsymb is a small collection of classes and functions that use and 
extend the tools of `numpy` and `sympy` for use with nonlinear 
dynamic systems that require somewhat intricate tensor algebra 
between symbolic object. An example use is::

    #!/usr/bin/env python

    from nlsymb import aux
    from nlsymb import object_einsum
    from nlsymb import lqr

    with nlsymb.Timer():
        ref = aux.trajectory('x', 'u')
        ref.addpoint(0, x=xinit, u=[1, 0])
        ref.addpoint(2, x=[-6, -7, 0, 0], u=[0, 1])
        ref.interpolate()

    print "reference x at time %f is %f " % (1.34, ref.x(1.34)) 


Installation
============

Not implemented yet. For now just clone the repo and manually copy
things where they need to be.


TODO
====
More than I can think of. Too too much. 
