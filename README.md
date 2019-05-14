# Stopping

## Purpose

Tools to ease the uniformization of stopping criteria in iterative solvers.

When a solver is called on an optimization model, four outcome may happen:

1. the approximate solution is obtained, the problem is considered solved
2. the problem is declared unbounded
3. the maximum available ressources is not sufficient to compute the solution
4. some algorithm dependent failure happens

This tool eases the first 3 items above. It defines a type

    mutable struct GenericStopping <: AbstractStopping
        problem       :: Any          # an arbitrary instance of a problem
        meta          :: StoppingMeta # contains the used parameters
        current_state :: State        # the current state

The [StoppingMeta](https://github.com/Goysa2/Stopping.jl/blob/master/src/StoppingMetamod.jl) provides default tolerances, maximum ressources, ...  as well as (boolean) information on the result.

We provide some specialization of the GenericStopping for instance :
  * [NLPStopping](https://github.com/Goysa2/Stopping.jl/blob/master/src/NLPStoppingmod.jl): for non-linear programming;
  * [LS_Stopping](https://github.com/Goysa2/Stopping.jl/blob/master/src/LineSearchStoppingmod.jl): for 1d optimization;
  * more to come...
  
In these examples, the function `optimality_residual` computes the residual of the optimality conditions is an additional attribute of the type.

## Functions

The tool provides two functions:
* `start!(stp :: AbstractStopping)` initializes the time and the tolerance at the starting point and check wether the initial guess is optimal.
* `stop!(stp :: AbstractStopping)` checks optimality of the current guess as well as failure of the system (unboundedness for instance) and maximum ressources (number of evaluations of functions, elapsed time ...)

The stopping uses the informations furnished by the State to evaluate its functions. Communication between the two can be done through the following functions:
* `update_and_start!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call start!.
* `update_and_stop!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call stop!.
* `fill_in!(stp :: AbstractStopping, x :: Iterate)` a function that fill in all the State with all the informations required to correctly evaluate the stopping functions. This can reveal useful, for instance, if the user do not trust the informations furnished by the algorithm in the State.

## How to install

The stopping package can be installed and tested through the Julia package manager:

```julia
(v0.7) pkg> add https://github.com/Goysa2/Stopping.jl
(v0.7) pkg> test Stopping
```
Note that the package [State.jl](https://github.com/Goysa2/State.jl) is required and can be installed also through the package manager:
```julia
(v0.7) pkg> add https://github.com/Goysa2/State.jl
```
## Example

As an example, a naïve version of the Newton method is provided. First we import the packages:
```
using NLPModels, State, Stopping
```

We create an uncontrained quadratic optimization problem using [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl):
```
A = rand(5, 5); Q = A' * A;

f(x) = 0.5 * x' * Q * x
nlp = ADNLPModel(f,  ones(5))
```

We use our NLPStopping structure by creating our State and Stopping:

```
nlp_at_x = NLPAtX(ones(5))
stop_nlp = NLPStopping(nlp, (x,y) -> Stopping.unconstrained(x,y), nlp_at_x)
```

Now a basic version of Newton to illustrate how to use State and Stopping.

```
function newton(stp :: NLPStopping)
    state = stp.current_state; xt = state.x;
    update!(state, x = xt, gx = grad(stp.pb, xt), Hx = hess(stp.pb, xt))
    OK = start!(stp)

    while !OK
        d = -inv(state.Hx) * state.gx

        xt = xt + d

        update!(state, x = xt, gx = grad(stp.pb, xt), Hx = hess(stp.pb, xt))

        OK = stop!(stp)
    end

    return stp
end

stop_nlp = newton(stop_nlp)
```

We can look at the meta to know what happened
```
@show stop_nlp.meta.tired #ans: false
@show stop_nlp.meta.unbounded #ans: false
@show stop_nlp.meta.optimal #ans: true
```

We reached optimality!

## Long-Term Goals

Future work will address constrained problems. Then, fine grained information will consists in the number of constraint, their gradient etc. evaluation. The optimality conditions will be based on KKT equations. Separate tolerances for optimality and feasibility will be developed.

Future work will adress more sophisticated problems such as mixed integer optimization problems, optimization with uncertainty. The list of suggester optimality functions will be enriched with state of the art conditions.
