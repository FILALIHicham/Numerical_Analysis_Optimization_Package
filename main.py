#==================================================\\Description//===================================================

"""
This file contain a main function where I tested some of the functions that the package contains.
"""

#================================================\\Libraries needed//================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import timeit
from NumAn_Op.one_dim_min import *
from NumAn_Op.multi_dim_min import *
from NumAn_Op.sys_eq import *
import math

#===================================================\\The Program//==================================================


def g(x):
    return 0.65-(0.75/(1+x**2))-(0.65*x*math.atan(1/x))
def gp(x):
    return (1.5*x)/(1+x**2)**2+(0.65*x)/(1+x**2)-0.65*math.atan(1/x)
def gpp(x):
    return (2.8-3.2*x**2)/(1+x**2)**3

def f(x):
    return (x**3-x**2-9.)**2
def fp(x):
    return 2*x*(3*x-2)*(x**3-x**2-9)
def fpp(x):
    return 2*x*(3*x-2)*(x**3-x**2-9)

D=lambda x: -2*np.sin(x[0])+x[0]**2/3+x[1]**2/3-4
dD = lambda x,y: asarray([(2*x)/3-2*np.cos(x), (2*y)/3])
t= lambda x,y: -2*np.sin(x)+x**2/3+y**2/3-4

D =lambda x : 0.5 * ( x[0]**2 + x[1]**2 )
dD = lambda x,y: asarray([x,y])
t= lambda x,y: 0.5*(x**2+y**2) 

t=np.vectorize(t)
x0=np.array([-0.5,1])
Q=np.array([[8,-4],[-4,8]])
b=np.array([0,12])
bounds = asarray([[2,1], [2,1]])
x1=np.array([2,1])

M=np.array([[2,4,-2],[4,9,-3],[-2,-3,7]])
N=M=np.array([[2,4,-2,2],[4,9,-3,8],[-2,-3,7,10]])
MM=np.array([[4,-2,2],[-2,2,-4],[2,-4,11]])


answer=True
while answer:
    print ("""
    1. One dimensional minimization
    2. Multi-dimensional minization
    3. Equation systems
    4. Exit/Quit
    """)
    answer=input("Which module do you want to try ? : ") 
    if answer=="1": 
        print ("""
        1.  Search with fixed step size
        2.  Exhaustive Search
        3.  Dichotomous Search
        4.  Interval Halving
        5.  Fibonacci Method
        6.  Golden Section
        7.  Newton Raphson
        8.  Quasi Newton
        9.  Secant
        """)
        answer2=input("\n Which function do you want to try? : ")
        if answer2=="1":
            print("Fixed Step Size:")
            t1=timeit.default_timer()
            print("The optimum point is: "+str((fixed_step_size(f,0.7,0.5)[0]+fixed_step_size(f,0.7,0.5)[1])/2))
            t2=timeit.default_timer()
            d1=t2-t1
            print("The execution time in seconds is: "+str(d1))
            input("Press Enter to continue")
        elif answer2=="2":
            print("Exhaustive Search:")
            t3=timeit.default_timer()
            print("The optimum point is: "+str((exhaustive_search(f,0.7,3.5,12)[0]+exhaustive_search(f,0.7,3.5,12)[1])/2))
            t4=timeit.default_timer()
            d2=t4-t3
            print("The execution time in seconds is: "+str(d2))
            input("Press Enter to continue")
        elif answer2=="3":
            print("Dichotomous Search:")
            t5=timeit.default_timer()
            print("The optimum point is: "+str((dichotomous_search(f,0,3.5,0.01)[0]+dichotomous_search(f,0,3.5,0.01)[1])/2))
            t6=timeit.default_timer()
            d3=t6-t5
            print("The execution time in seconds is: "+str(d3))
            input("Press Enter to continue")
        elif answer2=="4":
            print("Interval Halving:")
            t7=timeit.default_timer()
            print("The optimum point is: "+str((interval_halving(f,0.7,3.5,0.01)[0]+interval_halving(f,0.7,3.5,0.01)[1])/2))
            t8=timeit.default_timer()
            d4=t8-t7
            print("The execution time in seconds is: "+str(d4))
            input("Press Enter to continue")
        elif answer2=="5":
            print("Fibonacci Method:")
            t9=timeit.default_timer()
            print("The optimum point is: "+str((fibonacci_method(f,0.7,3.5,12)[0]+fibonacci_method(f,0.7,3.5,12)[1])/2))
            t10=timeit.default_timer()
            d5=t10-t9
            print("The execution time in seconds is: "+str(d5))
            input("Press Enter to continue")
        elif answer2=="6":
            print("Golden Section:")
            t11=timeit.default_timer()
            print("The optimum point is: "+str((fixed_step_size(f,0.7,0.5)[0]+fixed_step_size(f,0.7,0.5)[1])/2))
            t12=timeit.default_timer()
            d6=t12-t11
            print("The execution time in seconds is: "+str(d6))
            input("Press Enter to continue")
        elif answer2=="7":
            print("Newton-Rapson Method:")
            t13=timeit.default_timer()
            print("The optimum point is: "+str(newton_raphson(gp,gpp,0.01,0.0001,100)))
            t14=timeit.default_timer()
            d7=t14-t13
            print("The execution time in seconds is: "+str(d7))
            input("Press Enter to continue")
        elif answer2=="8":
            print("Quasi-Newton Method:")
            t15=timeit.default_timer()
            print("The optimum point is: "+str(quasi_newton(g,0.1,0.01,0.0001,100)))
            t16=timeit.default_timer()
            d8=t16-t15
            print("The execution time in seconds is: "+str(d8))
            input("Press Enter to continue")
        elif answer2=="9":
            print("Secant Method:")
            t17=timeit.default_timer()
            print("The optimum point is: "+str(secant(gp,0.01,1.1,0.01)))
            t18=timeit.default_timer()
            d9=t18-t17
            print("The execution time in seconds is: "+str(d9))
            input("Press Enter to continue")
        else:
            print("\n Not Valid Choice Try again") 
    elif answer=="2":
        print ("""
        1.  Gradient Descent
        2.  Conjugate Gradient
        3.  AdaGrad
        4.  Newton
        """)
        answer2=input("\n Which function do you want to try? : ") 
        if answer2=="1":
            print("Gradient Descent:")
            t1=timeit.default_timer()
            print("The optimum point is: "+str(GradientDescent(D,x0,1e-5)))
            t2=timeit.default_timer()
            d1=t2-t1
            print("The execution time in seconds is: "+str(d1))
            input("Press Enter to continue")
        elif answer2=="2":
            print("Conjugate Gradient:")
            t3=timeit.default_timer()
            print("The optimum point is: "+str(ConjugateGradient(D,x0,Q,b)))
            t4=timeit.default_timer()
            d2=t4-t3
            print("The execution time in seconds is: "+str(d2))
            input("Press Enter to continue")
        elif answer2=="3":
            print("AdaGrad:")
            t5=timeit.default_timer()
            print("The optimum point is: "+str(adagrad(D,dD,bounds,step_size=2)))
            t6=timeit.default_timer()
            d3=t6-t5
            print("The execution time in seconds is: "+str(d3))
            input("Press Enter to continue")
        elif answer2=="4":
            print("Newton:")
            t7=timeit.default_timer()
            print("The optimum point is: "+str(Newton(D,x1,1e-7)))
            t8=timeit.default_timer()
            d4=t8-t7
            print("The execution time in seconds is: "+str(d4))
            input("Press Enter to continue")
    elif answer=="3":
        print ("""
        1.  Inverse
        2.  Gauss Jordan
        3.  LU Decomposition
        4.  Cholesky Decomposition
        """)
        answer3=input("\n Which function do you want to try? : ")
        if answer3=="1":
            print("Matrix Inverse:")
            t1=timeit.default_timer()
            print("The inverse is: ")
            print(inverse(M))
            t2=timeit.default_timer()
            d1=t2-t1
            print("The execution time in seconds is: "+str(d1))
            input("Press Enter to continue")
        elif answer3=="2":
            print("Gauss Jordan :")
            t3=timeit.default_timer()
            print("The solution is: ")
            print(Gauss_jordan(N))
            t4=timeit.default_timer()
            d2=t4-t3
            print("The execution time in seconds is: "+str(d2))
            input("Press Enter to continue")
        elif answer3=="3":
            print("LU Decomposition:")
            t5=timeit.default_timer()
            L,U=LU_decomposition(M)
            print("L =\n",L)
            print("U =\n",U)
            t6=timeit.default_timer()
            d3=t6-t5
            print("The execution time in seconds is: "+str(d3))
            input("Press Enter to continue")
        elif answer3=="4":
            print("Cholesky Decomposition:")
            t7=timeit.default_timer()
            print(Cholesky_Decomposition(MM))
            t8=timeit.default_timer()
            d4=t8-t7
            print("The execution time in seconds is: "+str(d4))
            input("Press Enter to continue")
    elif answer=="4":
        print("\n Goodbye")
        answer=False
    elif answer !="":
        print("\n Not Valid Choice Try again") 