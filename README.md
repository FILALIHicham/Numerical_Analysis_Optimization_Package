# Numerical_Analysis_Optimization_Package
Python package that contains some numerical analysis & optimization algorithms.

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
Following are some plots visualizing the progress of some algorithms that the package contains. You can find all the scripts to make them in the [Plotting_Scripts](/Plotting_Scripts/) folder in this repository.

1. One dimensional function minimization comparison:
   - Elimination methods comparison
   

      ![Function Optimization Comparison. (Elimination Methods)](/Elimination_Methods_Comparison.gif)
      
      
      
   - Interpolation methods comparison
   

      ![Function Optimization Comparison. (Interpolation Methods)](/Interpolation_Methods_Comparison.gif)
      
      
2. Mutli-Variable function minimization comparison:


      ![Multi-dimensional function minimization algorithms comparison](/3D_Comparison.gif)
      
Note: You won't get the same progess path for the Adagrad method if you try to use the scripts in the [Plotting_Scripts](/Plotting_Scripts/) folder, and this is due to the fact that Adagrad is a variant of the stochastic gradient descent method, this means that it takes a random starting point each time.


