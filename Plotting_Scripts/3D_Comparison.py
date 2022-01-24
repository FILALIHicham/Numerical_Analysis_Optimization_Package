#=================================================\\Documentation//==================================================

"""
This program generates and saves a gif file containing a 3D comparison plot of the multi_dimensional optimization methods
that we have imported in the package. In this program, I have defined the "iter" function which allows to collect 
the progress data of each optimisation method, so I redefined the optimisation methods to implement the "iter" function in them. 

"""
#================================================\\Libraries needed//================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.linalg as npla
from numdifftools import Gradient,Hessian
from matplotlib import cm
from math import sqrt
from numpy import asarray
from numpy.random import rand
from scipy.misc import derivative as deriv

#===================================================\\The Program//==================================================

def iter(f,x,XL,YL):
    XL.append(x[0])
    YL.append(x[1])

# 1.Gradient Descent:
XL=[]
YL=[]
def armijo(f,alpha0 = 1,beta = 2,delta = 0.05):
    """
    armijo(function,alpha = 1,delta = 0.05).
  
    Calculates optimal step size for descent methods.
  
    Parameters:
    f (function): the function to minimize
    alpha (float): initial step size of our optimization algorithm
    beta(float): scaling factor
    delta (float): tolerance (a small value)
  
    Returns:
    alpha: improved step size after one iteration
  
    """
    alpha = alpha0
    while f(alpha) <= f(0) + delta * alpha * deriv(f, 0, dx = 1e-5):
        alpha *= beta
    while f(alpha) > f(0) + delta * alpha * deriv(f, 0, dx = 1e-5):
        alpha /= beta
    return alpha
    
def GradientDescent(f, x0, delta=1e-5):
    """
    gradient_descent(function, x0, delta=1e-5).
  
    Finds global minimum using gradient descent.
  
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
        iter(f,x,XL,YL)
        g = lambda alpha: f(x - alpha * d)
        alpha = armijo(g)
        x = x - d * alpha
        d = Gradient(f)(x)
    iter(f,x,XL,YL)
    iter(f,x,XL,YL)
    return x

# 2.Conjugate Gradient:
XL2=[]
YL2=[]
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
        iter(f,x,XL2,YL2)
        alpha = -np.dot((np.dot(Q,x)-b),p)/np.dot(np.dot(p,Q),p)
        x = x + alpha*p
        grad = Gradient(f)(x)
        beta = np.dot(np.dot(grad,Q),p)/np.dot(np.dot(p,Q),p)
        p = -grad + beta*p
    iter(f,x,XL2,YL2)
    iter(f,x,XL2,YL2)
    return x

# 3. AdaGrad:
XL3=[]
YL3=[]
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
        iter(f,solution,XL3,YL3)
    iter(f,solution,XL3,YL3)
    return solution

# 4.Newton Method:
XL4=[]
YL4=[]

def Newton( f, x0 , delta = 1e-5 ):
    '''
    Newton( f, x0 , delta = 1e-5 ).
    This method finds global minimum using Newton method.

    Parameters:
    f (function): the function to minimize
    x0 (vector): initial value for Newton method
    delta (float): tolerance

    Returns:
    x: the argument that minimizes the function

    '''
    x,n = x0, len(x0)
    I = np.identity(n)
    d = -np.dot(npla.inv(Hessian(f)(x)), Gradient(f)(x))
    while(npla.norm(d) > delta):
        iter(f,x,XL4,YL4)
        x = x + d
        if(np.all(np.linalg.eigvals(Hessian(f)(x)) > 0)):
            d = -np.dot(npla.inv(Hessian(f)(x)),Gradient(f)(x))
        else:
            d = -np.dot(npla.inv(delta*I+Hessian(f)(x)),Gradient(f)(x))
    iter(f,x,XL4,YL4)
    return x


# defining the function
f=lambda x: -2*np.sin(x[0])+x[0]**2/3+x[1]**2/3-4
df = lambda x,y: asarray([(2*x)/3-2*np.cos(x), (2*y)/3])
t= lambda x,y: -2*np.sin(x)+x**2/3+y**2/3-4
t=np.vectorize(t)
x0=np.array([10,-10])
Q=np.array([[2,5],[5,2]])
b=np.array([2/5,5])
bounds = asarray([[-15, 15], [-15, 15]])
x1=np.array([2.6,10])

# making the plot figure
fig = plt.figure()
ax = plt.axes(projection='3d')
X = np.arange(-10, 10, 0.05)
X, Y = np.meshgrid(X, X)
Z =t(X, Y)

# get the optimize progress.
S = GradientDescent(f,x0,1e-5)
S1=ConjugateGradient(f,x0,Q,b)
S2=adagrad(f,df,bounds,step_size=2)
S3=Newton(f,x1,1e-7)
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True,cmap=cm.coolwarm,alpha=0.6)
#ax.zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf, shrink=0.5, aspect=10)

# setting the datasets of each method.
ZL=t(XL,YL)
data = np.array([XL,YL,ZL])
ZL2=t(XL2,YL2)
data2 = np.array([XL2,YL2,ZL2])
ZL3=t(XL3,YL3)
data3 = np.array([XL3,YL3,ZL3])
ZL4=t(XL4,YL4)
data4 = np.array([XL4,YL4,ZL4])

line, = ax.plot([10], [-10], f(x0), 'b', label='Gradient Descent', lw=2)
point, = ax.plot([10], [-10], f(x0), 'bo')
line2, = ax.plot([10], [-10], f(x0), 'r', label='Conjugate Gradient', lw=2)
point2, = ax.plot([10], [-10], f(x0), 'ro')
line3, = ax.plot([10], [-10], f(x0), 'g', label='Adagrad', lw=2)
point3, = ax.plot([10], [-10], f(x0), 'go')
line4, = ax.plot([10], [-10], f(x0), 'm', label='Newton', lw=2)
point4, = ax.plot([10], [-10], f(x0), 'mo')

ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')

ax.legend([line,line2,line3,line4], [line.get_label(), line2.get_label(), line3.get_label(), line4.get_label()], loc=0,prop={"size":6})

# animation function.  This is called sequentially
def animate(i):
    if i<data.shape[1]:
        line.set_data(data[:2, :i])
        line.set_3d_properties(data[2, :i])
        point.set_data(data[:2, i-1:i])
        point.set_3d_properties(data[2, i-1:i])
    if i<data2.shape[1]:
        line2.set_data(data2[:2, :i])
        line2.set_3d_properties(data2[2, :i])
        point2.set_data(data2[:2, i-1:i])
        point2.set_3d_properties(data2[2, i-1:i])
    if i<data3.shape[1]:
        line3.set_data(data3[:2, :i])
        line3.set_3d_properties(data3[2, :i])
        point3.set_data(data3[:2, i-1:i])
        point3.set_3d_properties(data3[2, i-1:i])
    if i<data4.shape[1]:
        line4.set_data(data4[:2, :i])
        line4.set_3d_properties(data4[2, :i])
        point4.set_data(data4[:2, i-1:i])
        point4.set_3d_properties(data4[2, i-1:i])
    return line, point, line2, point2, line3, point3, line4, point4

anim = animation.FuncAnimation(fig, animate,frames=max(data.shape[1],data2.shape[1],data3.shape[1],data4.shape[1]), interval=100, repeat_delay=500, blit=True)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=3, metadata=dict(artist='FILALI Hicham'), bitrate=-1)
ax.view_init(15, 25)
plt.title("Multi-Variable Function Optimization Comparison.")
plt.show()
#anim.save('3D_Comparison.gif',writer=writer)

#====================================================================================================================
