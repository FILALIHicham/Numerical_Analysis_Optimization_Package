#=================================================\\Documentation//==================================================

"""
This program generates and saves a gif file containing a comparison plot of the optimisation Interpolation methods
that we have imported in the package. In this program, I have defined the "iter" function which allows to collect 
the progress data of each optimisation method, so I redefined the optimisation methods to implement the "iter" function in them. 

"""
#================================================\\Libraries needed//================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

#===================================================\\The Program//==================================================

def iter(f,x,points,images):
    points.append(x)
    images.append(f(x))


def f(x):
    return -((x-5)**2*x**2)/5
def fp(x):
    return -(2*(x-5)*x*(2*x-5))/5
def fpp(x):
    return -(12*x**2-60*x+50)/5


def f(x):
    return 0.65-(0.75/(1+x**2))-(0.65*x*math.atan(1/x))
def fp(x):
    return (1.5*x)/(1+x**2)**2+(0.65*x)/(1+x**2)-0.65*math.atan(1/x)
def fpp(x):
    return (2.8-3.2*x**2)/(1+x**2)**3

# 1.NEWTON-RAPSON METHOD:
pts=[]
img=[]

def newton_raphson( fp , fpp , xi , epsilon , max ):
    """
    newton_raphson( fp , fpp , xi , epsilon , max ). 
    This is a root-finding method that finds the minimum of a function (f) using newton-rapshon method.

    Parameters:
    fp (function): the first derivative of the function to minimize
    fpp (function): the second derivative of the function to minimize
    xi (float): the starting point for the iterative process
    epsilon (float): very small number
    max (int): max number of iterations

    Returns:
    x: optimum value
        
    """
    k=0
    x=xi
    fpx=fp(x)
    fppx=fpp(x)
    iter(f,x,pts,img)
    while abs(fpx)>epsilon and k<max:
        x=x-fpx/fppx
        fpx=fp(x)
        fppx=(fp(x+epsilon)-fpx)/epsilon
        k+=1
        iter(f,x,pts,img)
    if k==max:
        print("Error")
    else:
        iter(f,x,pts,img)
        return x


# 2.QUASI-NEWTON METHOD:
pts2=[]
img2=[]

def quasi_newton( f , xi , step , epsilon , max ):
    """
    quasi_newton( f , xi , step , epsilon , max ). 
    This is a root-finding method that finds the minimum of a function (f) using quasi-newton method.

    Parameters:
    f (function): the function to minimize
    xi (float): the starting point for the iterative process
    step (float): small step size to calculate an approached value to the derivatives
    epsilon (float): very small number
    max (int): max number of iterations

    Returns:
    x: optimum value
        
    """
    k=0
    x=xi
    fx=f(x)
    fplus=f(x+step)
    fmoins=f(x-step)
    x=x-((step*(fplus-fmoins))/(2*(fplus-2*fx+fmoins)))
    fplus=f(x+step)
    fmoins=f(x-step)
    dfx=(fplus-fmoins)/(2*step)
    iter(f,x,pts2,img2)
    while abs(dfx)>epsilon and k<max:
        fplus=f(x+step)
        fmoins=f(x-step)
        fx=f(x)
        dfx=(fplus-fmoins)/(2*step)
        x=x-((step*(fplus-fmoins))/(2*(fplus-2*fx+fmoins)))
        k+=1
        iter(f,x,pts2,img2)
    if k==max:
        print("Error")
    else:
        return x


# 3.SECANT METHOD:
pts3=[]
img3=[]

def secant( fp , a , b , epsilon ):
    """
    secant( fp , a , b , epsilon ). 
    This is a root-finding method that finds the minimum of a function (f) using secant method.

    Parameters:
    fp (function): the first derivative of the function to minimize
    a (float): inferior born of the initial uncertainty interval
    b (float): superior born of the initial uncertainty interval
    epsilon (float): very small number

    Returns:
    c: optimum value
        
    """
    c = a-fp(a)*(b-a)/(fp(b)-fp(a))
    prev = a
    iter(f,c,pts3,img3)
    while abs(c-prev) > epsilon:
        if fp(a)*fp(c)> 0:
            a = c
        else:
            b = c
            prev = c
            c = a-fp(a)*(b-a)/(fp(b)-fp(a))
        iter(f,c,pts3,img3)
    return c


f=np.vectorize(f)

fig = plt.figure()

ax = plt.axes(xlim=(0.1, 1))
fct,=ax.plot(np.linspace(0.1, 1,100), f(np.linspace(0.1, 1,100)),'black', label='$ f(x)= 0.65-(0.75/(1+x^2))-(0.65x/arctan(1/x)) $')


line, = ax.plot([], [], 'b-', label="Newton-Raphson")
point, = ax.plot([], [], 'bo')
line2, = ax.plot([], [], 'r-', label="Quasi-Newton")
point2, = ax.plot([], [], 'ro')
line3, = ax.plot([], [], 'g-', label="Secant")
point3, = ax.plot([], [], 'go')

ax.legend([fct,line,line2,line3], [fct.get_label(),line.get_label(), line2.get_label(), line3.get_label()],loc=0)

# get the optimize progress.

newton_raphson(fp,fpp,0.01,0.0001,100)
quasi_newton(f,0.1,0.01,0.0001,100)
secant(fp,0.01,1.1,0.001)

# setting the datasets of each method.
res_y = f(pts)
x = pts
y = img
data = np.array([x,y])

res_y2 = f(pts2)
x = pts2
y = img2
data2 = np.array([x,y])

res_y3 = f(pts3)
x = pts3
y = img3
data3 = np.array([x,y])

# initializing the plots.
def init():
    point.set_data([], [])
    line.set_data([],[])
    point2.set_data([], [])
    line2.set_data([],[])
    point3.set_data([], [])
    line3.set_data([],[])
    return point,line,point2,line2,point3,line3

# animation function.  This is called sequentially
def animate(i):
    if i<len(pts):
        point.set_data(data[::,i:i+1])
        line.set_data(data[::,:i+1])
    if i<len(pts2):
        line2.set_data(data2[...,:i+1])
        point2.set_data(pts2[i], res_y2[i])
    if i<len(pts3):
        line3.set_data(data3[...,:i+1])
        point3.set_data(pts3[i], res_y3[i])
    return point,line,point2,line2,point3,line3

Writer = animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='FILALI Hicham'), bitrate=-1)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=max(len(pts2),len(pts),len(pts3)), interval=500, blit=True)
plt.title("Function Optimization Comparison. (Interpolation Methods)")
plt.show()

#anim.save('Interpolation_Methods_Comparison.gif', writer=writer)

#====================================================================================================================
