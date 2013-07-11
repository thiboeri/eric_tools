import numpy
from eric_tools.mlp_utils import *
from scipy.linalg import lstsq

def mDA(X, p, Z = []):
    # pseudocode is in dim x batch shape
    # add a row of ones : constant feature
    # transposed to fit the original pseudocode: X is (d x n)
    X = X.T


    X = numpy.vstack((X, numpy.ones(X.shape[1])))
     
    d, n = X.shape

    #print X

    # W = P*Q^-1

    q = numpy.ones(d) - p
    q[-1] = 1
    q = cast32(q)
    #q = q.reshape(len(q), 1)

    S = numpy.dot(X, X.T)
    #print S
   
    Q = numpy.zeros((d,d))

    for alpha in range(d):
        for beta in range(d):
            if alpha != beta:
                Q[alpha, beta] = S[alpha, beta] * q[alpha] * q[beta]
            else:
                Q[alpha,beta] = S[alpha, beta] *q[alpha]
    #print Q

    P = numpy.zeros((d,d))

    for alpha in range(d):
        for beta in range(d):
            P[alpha, beta] = S[alpha, beta] * q[beta]

    #print P

    # W is transposed for computing hiddens through dot(X,W)
    W = lstsq(Q.T, P.T)[0].T[:-1]
    h = numpy.tanh(numpy.dot(W, X))
    #print W
    Zh = []
    if Z:
        for z in Z:
            z_in = z.T
            z_in = numpy.vstack((z_in, numpy.ones(z_in.shape[1])))
            Zh.append(numpy.tanh(numpy.dot(W, z_in)).T)

    return W.T, h.T, Zh 

    
    #n, d = X.shape
    #q = numpy.ones(1, d) * (1-p)

    

#X = numpy.array([[1,3],[2,4]])

#W1 = mDA(X, 0.1)

#print W1

