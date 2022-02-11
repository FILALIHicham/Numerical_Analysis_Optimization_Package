#=================================================\\Documentation//==================================================

"""
This module contains different types of optimization methods that search for the min of a one dimensional unimodal function, 
using both the search with elimination and search with interpolation methods.

"""

#================================================\\Libraries needed//================================================

import numpy as np

#=======================================\\Searching with elimination methods//=======================================

# 1.UNRESTRICTED SEARCH:

# Search with fixed step size:

def fixed_step_size( f , guess , step=.01):
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
    while x1>=x2:
        x1=x2
        guess=guess+k*step
        x2=f(guess+k*step)
    return (guess,guess+k*step)


# Search with accelerated step size:

def accelerated_step_size( f , guess , step=.01 ):
    """
    accelerated_step_size( f , guess , step ). 
    This method finds the minimum of a function (f) using the accelerated step size search.

    Parameters:
    f (function): the function to minimize
    guess (float): initial value to start the algorithm
    step (float): the initial step size 

    Returns:
    (guess,guess+k*step): a tuple containing an interval that encloses the minimum value

    """
    s=step
    k=(f(guess)<f(guess+step))*(-2)+1
    x1=f(guess)
    x2=f(guess+k*step)
    while abs(x1-x2)>10**-2:
        step=s
        x1=f(guess)
        x2=f(guess+k*step)
        while x1>=x2:
            x1=x2
            step=step*2
            x2=f(guess+k*step)
            guess=guess+k*(step/2)
    return (guess,guess+k*step)


# 2.EXHAUSTIVE SEARCH:

def exhaustive_search( f , a , b , n=20 ):
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
    for i in np.linspace(a,b,n):
        if f(i)<=min:
            min=f(i)
            x=i
    return (x-(1/n),x+(1/n))


# 3.DICHOTOMOUS SEARCH:

def dichotomous_search( f, a , b , delta=0.01 ):
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
    while round(abs(b-a), 3) > abs(delta):
        x = (a + b - delta)/2
        y = (a + b + delta)/2
        if f(x) < f(y):
            b = y
        else:
            a = x
    return (a,b)


# 4.INTERVAL HALVING METHOD:

def interval_halving( f , a , b , e=0.01 ):
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


def fibonacci_method( f , a , b , n=10 ):
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
    return(a,b)


# 6.GOLDEN SECTION METHOD:


def golden_section( f , a , b , e=0.01 ):
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
    while b-a>=e:
        L=L*0.618
        x1=b-L
        x2=a+L
        if f(x1)<=f(x2):
            b=x2
        else:
            a=x1
    return (a,b)


#======================================\\Searching with interpolation methods//======================================

# 1.NEWTON-RAPHSON METHOD:

def newton_raphson( fp , fpp , xi , epsilon=1e-5 , max=100 ):
    """
    newton_raphson( fp , fpp , xi , epsilon , max ). 
    This is a root-finding method that finds the minimum of a function (f) using newton-raphson method.

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
    while abs(fpx)>epsilon and k<max:
        x=x-fpx/fppx
        fpx=fp(x)
        fppx=(fp(x+epsilon)-fpx)/epsilon
        k+=1
    if k==max:
        print("Error")
    else:
        return x


# 2.QUASI-NEWTON METHOD:

def quasi_newton( f , xi , step , epsilon=1e-5 , max=100 ):
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
    while abs(dfx)>epsilon and k<max:
        fplus=f(x+step)
        fmoins=f(x-step)
        fx=f(x)
        dfx=(fplus-fmoins)/(2*step)
        x=x-((step*(fplus-fmoins))/(2*(fplus-2*fx+fmoins)))
        k+=1
    if k==max:
        print("Error")
    else:
        return x


# 3.SECANT METHOD:

def secant( fp , a , b , epsilon=1e-5 ):
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
    while abs(c-prev) > epsilon:
        if fp(a)*fp(c)> 0:
            a = c
        else:
            b = c
            prev = c
            c = a-fp(a)*(b-a)/(fp(b)-fp(a))
    return c


#====================================================================================================================