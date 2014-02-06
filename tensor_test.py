from nlsymb import np, sym, matmult, tensor
import nlsymb.tensor as tn

from sympy import Symbol as S

if __name__ == "__main__":
    x1 = S('x1')
    x2 = S('x2')
    x = np.array([x1, x2])
    f = np.array([x[0], x[1]**2, 3*x[0]+x[1]]) 
    print f
    print tn.subs(f, {x1:2.0, x2:3.0})
    print tn.eval(f, x, [2.0, 3.0])

    A = tn.diff(f, x)
    print A
    print A[0, 1]
    Anum = tn.eval(A, x, [2,3])
    print Anum[0,1]

    g = np.dot(A, x)
    print g

    g = tn.einsum('ij,j', A, x)
    print g
