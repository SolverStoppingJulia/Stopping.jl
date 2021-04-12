## Stopping for Linear Algebra

The Stopping structure eases the implementation of algorithms and the
stopping criterion.

The following examples illustrate solver for linear algebra:
```math
Ax = b 
```
where A is an m x n matrix and b a vector of size m.

This tutorial illustrates the different step in preparing the resolution of a
new problem.
 - we create a `LinearAlgebraProblem` (that stores A, b)
 - we use the `GenericState` storing x and the `current_time`
 - we create a `LinearAlgebraStopping`
 - the optimality function `linear_system_check!`

The Julia file corresponding to this tutorial can be found [here](https://github.com/Goysa2/Stopping.jl/tree/master/test/examples/linear-algebra.jl).

```
using LinearAlgebra, Stopping, Test
```

```
m, n = 400, 200 #size of A: m x n
A    = 100 * rand(m, n)
xref = 100 * rand(n)
b    = A * xref

#Our initial guess
x0 = zeros(n)
```

```
mutable struct LinearAlgebraProblem
    A :: Any #matrix type
    b :: Vector
end
```

```
la_pb = LinearAlgebraProblem(A, b)
la_state = GenericState(xref)

@test norm(la_pb.A * xref - la_pb.b) <= 1e-6
```

```
mutable struct LinearAlgebraStopping <: AbstractStopping

        # problem
        pb :: LinearAlgebraProblem

        # stopping criterion
        optimality_check :: Function

        # Common parameters
        meta :: AbstractStoppingMeta

        # current state of the problem
        current_state :: AbstractState

        # Stopping of the main problem, or nothing
        main_stp :: Union{AbstractStopping, Nothing}

        function LinearAlgebraStopping(pb               :: LinearAlgebraProblem,
                                       optimality_check :: Function,
                                       current_state    :: AbstractState; kwargs...)
         return new(pb, linear_system_check!, StoppingMeta(; kwargs...), la_state, nothing)
        end
end

import Stopping._optimality_check

function _optimality_check(stp  :: LinearAlgebraStopping; kwargs...)

 optimality = stp.optimality_check(stp.pb, stp.current_state; kwargs...)

 return optimality
end

```

```
function linear_system_check!(pb    :: LinearAlgebraProblem,
                              state :: AbstractState; kwargs...)
 return norm(pb.A * state.x - pb.b)
end

@test linear_system_check!(la_pb, la_state) == 0.0
update!(la_state, x = x0)
@test linear_system_check!(la_pb, la_state) != 0.0

la_stop = LinearAlgebraStopping(la_pb, linear_system_check!, la_state,
                                max_iter = 150000, rtol = 1e-6)
```

Randomized block Kaczmarz
```
function RandomizedBlockKaczmarz(stp :: AbstractStopping; kwargs...)

    A,b = stp.pb.A, stp.pb.b
    x0  = stp.current_state.x

    m,n = size(A)
    xk  = x0

    OK = start!(stp)

    while !OK

     i  = Int(floor(rand() * m)+1) #rand a number between 1 and m
     Ai = A[i,:]
     xk  = Ai == 0 ? x0 : x0 - (dot(Ai,x0)-b[i])/dot(Ai,Ai) * Ai

     OK = update_and_stop!(stp, x = xk)
     x0  = xk

    end

 return stp
end
```

```
RandomizedBlockKaczmarz(la_stop)
@test status(la_stop) == :Optimal
```
