# Numerical_Analysis_Optimization_Package
Python package that contains some numerical analysis & optimization algorithms.

Instructions:
1. Install:
   Run this command in your terminal:
   
   ```
   
   pip install NumAn-Op
   
   ```
 
2. Modules:
   There are 3 modules in this package: 
   
      - [one_dim_min.py](https://github.com/FILALIHicham/Numerical_Analysis_Optimization_Package/blob/main/NumAn_Op/one_dim_min.py) 
      
      - [sys_eq.py](https://github.com/FILALIHicham/Numerical_Analysis_Optimization_Package/blob/main/NumAn_Op/sys_eq.py)
      
      - [multi_dim_min.py](https://github.com/FILALIHicham/Numerical_Analysis_Optimization_Package/blob/main/NumAn_Op/multi_dim_min.py)
      
   To use them you can import them as following:
   
   ```
   
   from NumAn_Op import one_dim_min
   
   from NumAn_Op import sys_eq
   
   from NumAn_op import multi_dim_min
   
   ```
   
   After importing the modules, you can use the help() function to get information about the modules and the functions that it contains.

Following are the algorithms present in this package:

I. One dimensional function minimization algorithms:
   - Searching with elimination methods
     - Unrestricted search
     - Exhaustive search
     - Dichotomous search
     - Interval halving method
     - Fibonacci method
     - Golden section method
   - Searching with interpolation methods
     - Newton-Rapson method
     - Quasi-Newton method
     - Secant method  

II. System of Equations & Decompositions:
  - The Elimination Of Gauss-Jordan
  - LU Decomposition Method
  - Cholesky Decomposition Method
  
III. Multi-dimensional function minimization algorithms:
  - Gradient methods
    - Gradient Descent method
    - Conjugate Gradient method
    - AdaGrad
  - Newton methods
    - Newton method
    - Quasi-Newton with DFP and armijo

# Visualization of some the progress of some algorithms
Following are some plots visualizing the progress of some algorithms that the package contains. You can find all the scripts to make them in the [Plotting_Scripts](https://github.com/FILALIHicham/Numerical_Analysis_Optimization_Package/tree/main/Plotting_Scripts) folder in this repository.

1. One dimensional function minimization comparison:
   - Elimination methods comparison
   

      ![Function Optimization Comparison. (Elimination Methods)](https://github.com/FILALIHicham/Numerical_Analysis_Optimization_Package/blob/main/Elimination_Methods_Comparison.gif)
      
      
      
   - Interpolation methods comparison
   

      ![Function Optimization Comparison. (Interpolation Methods)](https://github.com/FILALIHicham/Numerical_Analysis_Optimization_Package/blob/main/Interpolation_Methods_Comparison.gif)
      
      
2. Mutli-Variable function minimization comparison:


      ![Multi-dimensional function minimization algorithms comparison](https://github.com/FILALIHicham/Numerical_Analysis_Optimization_Package/blob/main/3D_Comparison.gif)
      
Note: You won't get the same progess path for the Adagrad method if you try to use the scripts in the [Plotting_Scripts](https://github.com/FILALIHicham/Numerical_Analysis_Optimization_Package/tree/main/Plotting_Scripts) folder, and this is due to the fact that Adagrad is a variant of the stochastic gradient descent method, this means that it takes a random starting point each time.


