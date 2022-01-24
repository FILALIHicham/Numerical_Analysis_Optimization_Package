#=================================================\\Documentation//==================================================

"""
This program generates and saves a gif file containing a comparison plot of the optimisation elimination methods
that we have imported in the package. In this program, I have defined the "iter" function which allows to collect 
the progress data of each optimisation method, so I redefined the optimisation methods to implement the "iter" function in them. 

"""
#================================================\\Libraries needed//================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#===================================================\\The Program//==================================================

def iter(f,x,points,images):
    points.append(x)
    images.append(f(x))

def f(x):
    return (x**3-x**2-9.)**2
def fp(x):
    return 2*x*(3*x-2)*(x**3-x**2-9)
def fpp(x):
    return 2*x*(3*x-2)*(x**3-x**2-9)

pts=[]
img=[]
def fixed_step_size( f , guess , step ):
    """
    fixed_step_size( f , guess , step ). 
    This method finds the minimum of a function (f) using the fixed step size search.

    Parameters:
    f (function): the function to minimize
    guess (float): initial value to start the algorithm
    step (float): step size 

    Returns:
    (guess,guess+k*step): a tuple containing an interval that encloses the minimum value

    """
    k=(f(guess)<f(guess+step))*(-2)+1
    x1=f(guess)
    x2=f(guess+k*step)
    iter(f,guess,pts,img)
    while x1>=x2:
        x1=x2
        guess=guess+k*step
        x2=f(guess+k*step)
        iter(f,guess,pts,img)
    iter(f,(guess+guess+k*step)/2,pts,img)
    iter(f,(guess+guess+k*step)/2,pts,img)
    iter(f,(guess+guess+k*step)/2,pts,img)
    return (guess,guess+k*step)


# 2.EXHAUSTIVE SEARCH:
pts2=[]
img2=[]

def exhaustive_search( f , a , b , n ):
    """
    exhaustive_search( f , a , b , n ). 
    This method finds the minimum of a function (f) using exhaustive search.

    Parameters:
    f (function): the function to minimize
    a (float): inferior born of the initial uncertainty interval
    b (float): superior born of the initial uncertainty interval
    n (int): number of equidistant points to test

    Returns:
    (x-(1/n),x+(1/n)): a tuple containing an interval that encloses the minimum value

    """
    min=f(a)
    x=a
    iter(f,x,pts2,img2)
    for i in np.linspace(a,b,n):
        if f(i)<=min:
            min=f(i)
            x=i
            iter(f,x,pts2,img2)
    iter(f,x,pts2,img2) 
    iter(f,x,pts2,img2)     
    return (x-(1/n),x+(1/n))


# 3.DICHOTOMOUS SEARCH:
pts3=[]
img3=[]
def dichotomous_search( f, a , b , delta ):
    """
    dichotomous_search( f, a , b , delta ). 
    This method finds the minimum of a function (f) using dichotomous search.

    Parameters:
    f (function): the function to minimize
    a (float): inferior born of the initial uncertainty interval
    b (float): superior born of the initial uncertainty interval
    delta (float): precision wanted

    Returns:
    (a,b): a tuple containing an interval that encloses the minimum value
    
    """    
    iter(f,(a+b)/2,pts3,img3)
    while round(abs(b-a), 3) > abs(delta):
        x = (a + b - delta)/2
        y = (a + b + delta)/2
        if f(x) < f(y):
            b = y
        else:
            a = x
        iter(f,(a+b)/2,pts3,img3)
    return (a,b)


# 4.INTERVAL HALVING METHOD:
pts4=[]
img4=[]

def interval_halving( f , a , b , e ):
    """
    interval_halving( f , a , b , e ). 
    This method finds the minimum of a function (f) using interval halving.

    Parameters:
    f (function): the function to minimize
    a (float): inferior born of the initial uncertainty interval
    b (float): superior born of the initial uncertainty interval
    e (float): precision wanted

    Returns:
    (a,b): a tuple containing an interval that encloses the minimum value
        
    """
    iter(f,(a+b)/2,pts4,img4)
    while b-a>e:
        s=(b-a)/4
        x1=a+s
        x2=b-s
        x0=a+2*s
        f0=f(x0)
        f1=f(x1)
        f2=f(x2)
        if f1<f0<f2:
            x0=x1
            b=x0
        elif f2<f0<f1:
            x0=x2
            a=x0
        elif f0<f1 and f0<f2:
            a=x1
            b=x2
        iter(f,(a+b)/2,pts4,img4)
    return (a,b)


# 5.FIBONACCI METHOD:

def fib( n ):
    """
    fib( n ). 
    This method calculate the nth Fibonacci number.

    Parameters:
    n(int): the nth Fibonacci number in the Fibonacci sequence

    Returns:
    f: nth Fibonacci number.
        
    """
    if n < 2:
        return 1
    else:
        f = fib(n-1) + fib(n-2)
    return f

pts5=[]
img5=[]

def fibonacci_method( f , a , b , n ):
    """
    fibonacci_method( f , a , b , n ).
    This method finds the minimum of a function (f) using Fibonacci method.

    Parameters:
    f (function): the function to minimize
    a (float): inferior born of the initial uncertainty interval
    b (float): superior born of the initial uncertainty interval
    delta (float): number of tests

    Returns:
    (a,b): a tuple containing an interval that encloses the minimum value
        
    """
    c=a+(fib(n-2)/fib(n))*(b-a)
    d=a+(fib(n-1)/fib(n))*(b-a)
    fc=f(c)
    fd=f(d)
    iter(f,(a+b)/2,pts5,img5)
    while(n>2):
        n-=1
        if fc<fd:
            b=d
            d=c
            fd=fc
            c=a+(fib(n-2)/fib(n))*(b-a)
            fc=f(c)
        else:
            a=c
            c=d
            fc=fd
            d=a+(fib(n-1)/fib(n))*(b-a)
            fd=f(d)
        iter(f,(a+b)/2,pts5,img5)
    return(a,b)


# 6.GOLDEN SECTION METHOD:
pts6=[]
img6=[]

def golden_section( f , a , b , e ):
    """
    golden_section( f , a , b , e ). 
    This method finds the minimum of a function (f) using golden section method.

    Parameters:
    f (function): the function to minimize
    a (float): inferior born of the initial uncertainty interval
    b (float): superior born of the initial uncertainty interval
    e (float): precision wanted

    Returns:
    (a,b): a tuple containing an interval that encloses the minimum value
        
    """
    L=b-a
    iter(f,(a+b)/2,pts6,img6)
    while b-a>=e:
        L=L*0.618
        x1=b-L
        x2=a+L
        if f(x1)<=f(x2):
            b=x2
        else:
            a=x1
        iter(f,(a+b)/2,pts6,img6)
    return (a,b)



f=np.vectorize(f)

fig = plt.figure()

ax = plt.axes(xlim=(-0.5, 3.5), ylim=(-10, 100))
fct,=ax.plot(np.linspace(-0.5, 3.5,100), f(np.linspace(-0.5, 3.5,100)),'black', label='$ f(x)= (x^3-x^2-9)^2 $')


line, = ax.plot([], [], 'b-', label="Fixed step size")
point, = ax.plot([], [], 'bo')
line2, = ax.plot([], [], 'r-', label="Exhaustive search")
point2, = ax.plot([], [], 'ro')
line3, = ax.plot([], [], 'g-', label="Dichotomous search")
point3, = ax.plot([], [], 'go')
line4, = ax.plot([], [], 'c-', label="Interval halving")
point4, = ax.plot([], [], 'co')
line5, = ax.plot([], [], 'y-', label="Fibonacci")
point5, = ax.plot([], [], 'yo')
line6, = ax.plot([], [], 'm-', label="Golden-section")
point6, = ax.plot([], [], 'mo')


ax.legend([fct,line,line2,line3,line4,line5,line6], [fct.get_label(),line.get_label(), line2.get_label(), line3.get_label(), line4.get_label(), line5.get_label(), line6.get_label()], loc=0)



# get the optimize progress.

fixed_step_size(f,0.7,0.5)
exhaustive_search(f,0.7,3.5,12)
dichotomous_search(f,0,3.5,0.01)
interval_halving(f,0.7,3.5,0.01)
fibonacci_method(f,0.7,3.5,12)
golden_section(f,0.7,3.5,0.01)

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

res_y4 = f(pts4)
x = pts4
y = img4
data4 = np.array([x,y])

res_y5 = f(pts5)
x = pts5
y = img5
data5 = np.array([x,y])

res_y6 = f(pts6)
x = pts6
y = img6
data6 = np.array([x,y])

# initializing the plots.
def init():
    point.set_data([], [])
    line.set_data([],[])
    point2.set_data([], [])
    line2.set_data([],[])
    point3.set_data([], [])
    line3.set_data([],[])
    point4.set_data([], [])
    line4.set_data([],[])
    point5.set_data([], [])
    line5.set_data([],[])
    point6.set_data([], [])
    line6.set_data([],[])
    return point,line,point2,line2,point3,line3,point4,line4,point5,line5,point6,line6


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
    if i<len(pts4):
        line4.set_data(data4[...,:i+1])
        point4.set_data(pts4[i], res_y4[i])
    if i<len(pts5):
        line5.set_data(data5[...,:i+1])
        point5.set_data(pts5[i], res_y5[i])
    if i<len(pts6):
        line6.set_data(data6[...,:i+1])
        point6.set_data(pts6[i], res_y6[i])
    return point, line,point2,line2,point3,line3,point4,line4,point5,line5,point6,line6


Writer = animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='FILALI Hicham'), bitrate=-1)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=max(len(pts2),len(pts),len(pts3),len(pts4),len(pts5),len(pts6)), interval=500, blit=True)
plt.title("Function Optimization Comparison. (Elimination Methods)")
plt.show()
#anim.save('Elimination_Methods_Comparison.gif', writer=writer)

#====================================================================================================================