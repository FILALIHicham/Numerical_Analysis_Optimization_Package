#=================================================\\Documentation//==================================================

"""
This module contains some types of numerical analysis methods for solving systems of equations & matrix decompositions, 
such as Gauss-Jordan, LU decomposition and Cholesky decomposition.

"""

#================================================\\Libraries needed//================================================

import numpy as np

#=========================================\\The Elimination Of Gauss-Jordan//========================================

# 1.The Inverse of a Matrix:

def inverse( M ):
    """
    inverse( M ). 
    This method calculate the inverse of a matrix (M).

    Parameters:
    M(array): the matrix that you want to inverse

    Returns:
    I: the inverted matrix.

    """
    n=len(M)
    I=np.eye(n)
    for indice in range(0,n-1):
        p=M[indice][indice]
        for i in range(indice+1,n):
            q =M[i][indice]/p
            for k in range(n):
                M[i][k]=M[i][k]-q*M[indice][k]
                I[i][k]=I[i][k]-q*I[indice][k]

    for indice in range(n-1,0,-1):
        p=M[indice][indice]
        for i in range(indice-1,-1,-1):
            q =M[i][indice]/p
            for k in range(n):
                M[i][k]=M[i][k]-q*M[indice][k]
                I[i][k]=I[i][k]-q*I[indice][k]

    for k in range(0,n):
        q=M[k][k]
        for j in range(0,n):
            I[k][j]=I[k][j]/q
            M[k][j]=M[k][j]/q
    return I


# 2.Gauss_Jordan method to solve system of linear equations:

def Gauss_jordan( M ):
    """
    Gauss_jordan( M ). 
    This method solves a system of linear equations and returns the solutions in a list.

    Parameters:
    M(array): array which each line of it is a linear equation of the system

    Returns:
    list(M[:,n:n+1]): a list of the system solutions.

    """
    n=len(M)
    for i in range(0,n-1):
        p=M[i][i]
        for j in range(i+1,n):
            q =M[j][i]/p
            for k in range(n+1):
                M[j][k]=M[j][k]-q*M[i][k]

    M[n-1][n] = M[n-1][n]/M[n-1][n-1]
    M[n-1][n-1]= 1
    for i in range(n-1,0,-1):
        p=M[i][i]
        for j in range(i-1,-1,-1):
            q =M[j][i]/p
            for k in range(n+1):
                M[j][k]=M[j][k]-q*M[i][k]

    M[0][n] = M[0][n]/M[0][0]
    M[0][0]= 1
    return list(M[:,n:n+1])


#=============================================\\LU Decomposition Method//============================================

# 1.LU Decomposition:

def LU_decomposition( M ):
    """
    LU_decomposition( M ). 
    This method applies the LU decomposition to a square matrix (M).

    Parameters:
    M(array): a square matrix

    Returns:
    (I,M): tuple containing lower and upper triangular matrix
    
    """
    n=len(M)
    I=np.eye(n)
    for i in range(0,n-1):
        p=M[i][i]
        for j in range(i+1,n):
            q =M[j][i]/p
            for k in range(n):
                M[j][k]=M[j][k]-q*M[i][k]

            I[j][i]=q

    return (I,M)


# 2.LU Decomposition method to solve system of equations:

def LU_decomposition_solve( M ):
    """
    LU_decomposition_solve( M ). 
    This method applies the LU decomposition to solve a system of equations.

    Parameters:
    M(array): array which each line of it is a linear equation of the system

    Returns:
    x: a list of the system solutions.
        
    """
    n=len(M)
    L,U=LU_decomposition(M[:,0:n])
    y=[]
    y.append(M[0,n]/L[0,0])
    for i in range(1,n):
        sum=0
        for j in range(i):
            sum+=L[i,j]*y[j]
        y.append((M[i,n]-sum)/L[i,i])
    x=[]
    x.append(y[n-1]/U[n-1,n-1])
    for i in range(n-2,-1,-1):
        sum=0
        for j in range(i+1,n):
            sum+=U[i,j]*x[n-j-1]
        x.append((y[i]-sum)/U[i,i])
    x.reverse()
    return x


#===========================================\\Cholesky Decomposition Method//=========================================

# 1.Cholesky Decomposition:

def Cholesky_Decomposition( M ):
    """
    Cholesky_Decomposition( M ). 
    This method applies the Cholesky decomposition to a positive definite matrix (M).

    Parameters:
    M(array): positive definite matrix

    Returns:
    A: lower triangular matrix.
       
    """
    n=len(M)
    A=np.zeros((n,n))
    A[0][0]=M[0][0]**(1/2)
    for i in range(1,n):
        A[i][0]=M[i][0]/A[0][0]
    for j in range(1,n):
        sum=0
        for k in range(0,j):
            sum+=A[j][k]**2
        A[j][j]=(M[j][j]-sum)**(1/2)
        for i in range(j+1,n):
            sum=0
            for k in range(j):
                sum+=A[i][k]*A[j][k]
        A[i][j]=(M[i][j]-sum)/A[j][j]
    return A

# 2.Cholesky Decomposition method to solve system of equations:

def Cholesky_Decomposition_solve( M ):
    """
    Cholesky_Decomposition_solve( M ). 
    This method applies the Cholesky decomposition to solve a system of equations.

    Parameters:
    M(array): array which each line of it is a linear equation of the system

    Returns:
    x: a list of the system solutions.
           
    """
    n=len(M)
    L=Cholesky_Decomposition(M[:,0:n])
    y=[]
    y.append(M[0,n]/L[0,0])
    for i in range(1,n):
        sum=0
        for k in range(i):
            sum+=L[i,k]*y[k]
        y.append((M[i,n]-sum)/L[i,i])
    U=L.T
    x=[]
    x.append(y[n-1]/U[n-1,n-1])
    for i in range(n-2,-1,-1):
        sum=0
        for k in range(i+1,n):
            sum+=U[i,k]*x[n-k-1]
        x.append(round((y[i]-sum)/U[i,i]))
    x.reverse()
    return x


#====================================================================================================================