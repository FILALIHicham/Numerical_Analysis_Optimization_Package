#=================================================\\Documentation//==================================================

"""
This module contains different types of optimization methods that search for the min of a multi-dimensional function, 
such as Newton method, Gradient descent, Conjugate Gradient...

"""

#================================================\\Libraries needed//================================================

import numpy as np
from scipy import optimize as op
import numpy.linalg as npla
from numdifftools import Gradient,Hessian
from scipy.misc import derivative as derive
from math import sqrt
from numpy import asarray
from numpy.random import rand

#=================================================\\Gradient methods//===============================================

# 1.Gradient Descent:

def GradientDescent( f , x0 , delta=1e-5 ):
    """
    GradientDescent( f , x0 , delta=1e-5 ).
    This method finds global minimum using gradient descent.

    Parameters:
    f (function): the function to minimize
    x0 (vector): initial value for gradient descent
    delta (float): tolerance

    Returns:
    x: the argument that minimizes the function

    """
    x = x0
    d = Gradient(f)(x)
    while npla.norm(d) > delta:
        phi = lambda alpha: f(x - alpha * d)
        alpha = op.newton(phi,0)
        x = x - d * alpha
        d = Gradient(f)(x)
    return x


# 2.Conjugate Gradient:

def ConjugateGradient( f , x , Q , b ):
    """
    ConjugateGradient( f , x , Q , b ).
    This method finds global minimum using conjugate gradient.

    Parameters:
    f (function): the function to minimize
    x (vector): initial value for conjugategradient
    Q (array): positive definite nxn symmetric matrix
    b (vector)

    Returns:
    x: the argument that minimizes the function
    
    """
    n = x.shape[0]
    p = -(np.dot(Q,x)-b)
    for i in range(n):
        alpha = -np.dot((np.dot(Q,x)-b),p)/np.dot(np.dot(p,Q),p)
        x = x + alpha*p
        grad = Gradient(f)(x)
        beta = np.dot(np.dot(grad,Q),p)/np.dot(np.dot(p,Q),p)
        p = -grad + beta*p
    return x


# 3. AdaGrad:

def adagrad( f , df , bounds , n_iter = 100 , step_size = .1 ):
    """
    adagrad( f , df , bounds , n_iter = 100 , step_size = .1 ).
    This method finds global minimum using the stochastic gradient descent variant: AdaGrad.

    Parameters:
    f (function): the function to minimize
    df (function): the first derivative of the function to minimize
    bounds (vector): uncertainty interval
    n_iter (int): number of iteration
    step_size (vector): step size

    Returns:
    solution: small interval that encloses the minimum value
        
    """
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])# generate an initial point
    # list of the sum square gradients for each variable
    sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    for it in range(n_iter):
        # calculate gradient
        gradient = df(solution[0], solution[1])
        # update the sum of the squared partial dfs
        for i in range(gradient.shape[0]):
            sq_grad_sums[i] += gradient[i]**2.0
        # build a solution one variable at a time
        new_solution = list()
        for i in range(solution.shape[0]):
            # calculate the step size for this variable
            alpha = step_size / (1e-8 + sqrt(sq_grad_sums[i]))
            # calculate the new position in this variable
            value = solution[i] - alpha * gradient[i]
            # store this variable
            new_solution.append(value)
        # evaluate candidate point
        solution = asarray(new_solution)
    return solution


#==================================================\\Newton methods//================================================

# 1.Newton Method:

def Newton( f, x0 , delta = 1e-5 ):
    """
    Newton( f, x0 , delta = 1e-5 ).
    This method finds global minimum using Newton method.

    Parameters:
    f (function): the function to minimize
    x0 (vector): initial value for Newton method
    delta (float): tolerance

    Returns:
    x: the argument that minimizes the function

    """
    x,n = x0, len(x0)
    I = np.identity(n)
    d = -np.dot(npla.inv(Hessian(f)(x)), Gradient(f)(x))
    while(npla.norm(d) > delta):
        x = x + d
        if(np.all(np.linalg.eigvals(Hessian(f)(x)) > 0)):
            d = -np.dot(npla.inv(Hessian(f)(x)),Gradient(f)(x))
        else:
            d = -np.dot(npla.inv(delta*I+Hessian(f)(x)),Gradient(f)(x))
    return x


# 2.Armijo Method:

def test( n , phi , eps , alpha ):
    __phi = phi(alpha)
    dphi = Gradient(phi)(0)
    _phi = phi(0)
    return __phi <= _phi + eps * dphi if n == 1 else __phi > _phi + eps * dphi


def armijo( phi , alpha0 , e = .01 , beta = 2 ):
    """
    armijo( phi , alpha0 , eps = .01 , beta = 2 ).
    This method calculate the optimal step size for descent methods.

    Parameters:
    phi (function): the function to minimize
    alpha0 (vector): initial step size of our optimization algorithm
    e (float): tolerance (a small value)
    beta (float): scaling factor

    Returns:
    alpha: improved step size

    """
    if alpha0 == 0:
        alpha = .1
    else:
        alpha = alpha0
    if test(1,phi, e, alpha):
        while not(test(2,phi,e, beta*alpha)):
            alpha = beta * alpha
    else:
        while not(test(1,phi, e, alpha)):
            alpha = alpha / beta
    return alpha


# 3.DFP:

def DFP(H,alphak,d,grd0, grd1):
    y = grd1 - grd0 
    dT = d.T 
    A = (alphak * d @ dT) / (dT @ y) 
    B = (-H @ y @ (H @ y).T) / (y.T @ H @ y) 
    return H + A + B


# 4.Quasi-Newton with DFP and armijo:

def quasiNewton(f, X0, delta = .01):
    """
    def quasiNewton(f, X0, delta = .01)
    This method minimize a multi-dimensional function
    
    Parameters:
    f (function): multi-dimensional function to minimize
    x0 (vector): intial point's coordinates
    delta (float): small quantity to check convergence

    Returns:
    X0: small interval that encloses the minimum value.

    """
    X0 = np.array([X0]).T
    global X
    X = X0
    grd1 = np.array([Gradient(f)(X0)]).T
    H = np.eye(len(X))
    while(np.linalg.norm(grd1) > delta):
        d = -H @ grd1
        phi = lambda alpha : f(X - alpha*grd1)
        alphak = armijo(phi,1)
        X = X0 + alphak * d
        grd0 = grd1
        grd1 = np.array([Gradient(f)(X)]).T
        H = DFP(H,alphak,d,grd0, grd1)
        X0 = X
    return X0


#====================================================================================================================