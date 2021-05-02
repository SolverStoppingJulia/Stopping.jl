## LinearAlgebraStopping: A Stopping for linear algebra

The Stopping-structure can be adapted to any problem solved by iterative methods. We discuss here `LAStopping` a specialization of an `AbstractStopping` for linear systems:
$$
Ax=b \text{ or } \min_x \frac{1}{2}\|Ax - b\|^2
$$
. We highlight here the specifities of such instance:
- The problem is either an `LLSModel` or `Stopping.LinearSystem`.
- These two types of problems have some access on `A`, `b` and counters of evaluations. The matrix `A` can be either given as a sparse/dense matrix or a linear operator.
- Default optimality functions are checking either the system directly or the normal equation.

```julia
#Problem definition:
m, n = 200, 100 #size of A: m x n
A    = 100 * rand(m, n) #It's a dense matrix :)
xref = 100 * rand(n)
b    = A * xref
#Our initial guess
x0 = zeros(n)

#Two definitions of LAStopping: 1) for dense matrix:
la_stop = LAStopping(A, b, GenericState(x0), 
                     max_iter = 150000, 
                     rtol = 1e-6, 
                     max_cntrs = init_max_counters_NLS(residual = 150000))
#2) for a linear operator:
op_stop = LAStopping(LinearSystem(LinearOperator(A), b), 
                     GenericState(x0), 
                     max_iter = 150000, 
                     rtol = 1e-6, 
                     max_cntrs = init_max_counters_linear_operators(nprod = 150000))
```
