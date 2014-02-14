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
things where they need to be or develop directly in this folder.


TODO
====
Make sure that flat_optim.py works as advertised (fixed some typos
but no time to check in depth). Also add a few benchmark trajectories
for it, like constant reference and control variations.

Write run_optim.py file that takes type of floor, initial point,
reference trajectory and cost functions as parameters/reads them from
a file.


