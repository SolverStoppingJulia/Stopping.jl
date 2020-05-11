# Stopping.jl

Documentation for Stopping.jl

## Purpose

Stopping.jl is a set of tools to ease the uniformization of stopping criteria in iterative solvers.

When a solver is called on an optimization model, four outcomes may happen:

1. the approximate solution is obtained, the problem is considered solved;
2. the problem is declared unsolvable (unboundedness, infeasibility, etc.);
3. the maximum available resources are not sufficient to compute the solution;
4. some algorithm dependent failure happens.

This tool eases the first three items above. It defines a type [GenericStopping](https://github.com/Goysa2/Stopping.jl/blob/master/src/Stopping/GenericStoppingmod.jl):

    mutable struct GenericStopping <: AbstractStopping
        problem       :: Any                              # an arbitrary instance of a problem
        meta          :: AbstractStoppingMeta             # contains the used parameters
        current_state :: AbstractState                    # the current state
        main_stp      :: Union{AbstractStopping, Nothing} # Stopping of the main problem, or nothing

The [StoppingMeta](https://github.com/Goysa2/Stopping.jl/blob/master/src/Stopping/StoppingMetamod.jl) provides default tolerances, maximum resources, ...  as well as (boolean) information on the result.

### Your Stopping your way

The GenericStopping (with GenericState) provides a complete structure to handle stopping criteria.
Then, depending on the problem structure, you can specialize a new Stopping by
redefining a State and some functions specific to your problem.

We provide some specialization of the GenericStopping for optimization:
  * [NLPStopping](https://github.com/Goysa2/Stopping.jl/blob/master/src/Stopping/NLPStoppingmod.jl) with [NLPAtX](https://github.com/Goysa2/Stopping.jl/blob/master/src/State/NLPAtXmod.jl) as a specialized State for non-linear programming (based on [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl));
  * [LS_Stopping](https://github.com/Goysa2/Stopping.jl/blob/master/src/Stopping/LineSearchStoppingmod.jl) with [LSAtT](https://github.com/Goysa2/Stopping.jl/blob/master/src/State/LSAtTmod.jl) as a specialized State for 1d optimization;
  * more to come.

In these examples, the function `optimality_residual` computing the residual of the optimality conditions is an additional attribute of the types.

## Functions

An AbstractStopping has two main functions:
* `start!(stp :: AbstractStopping)` initializes the time and the tolerance at the starting point and check whether the initial guess is optimal;
* `stop!(stp :: AbstractStopping)` checks optimality of the current guess as well as a failure of the system (unboundedness for instance) and maximum resources (number of evaluations of functions, elapsed time ...).

Stopping uses the information furnished by the State to assess the success. Communication between the two can be done through the following functions:
* `update_and_start!(stp :: AbstractStopping; kwargs...)` updates the State with information furnished as kwargs and then call start!.
* `update_and_stop!(stp :: AbstractStopping; kwargs...)` updates the State with information furnished as kwargs and then call stop!.
* `fill_in!(stp :: AbstractStopping, x :: Iterate)` a function that fill in all the State with all the information required to evaluate the Stopping functions correctly. This can reveal useful, for instance, if the user do not trust the information furnished by the algorithm in the State.
* `reinit!(stp :: AbstractStopping)` reinitialize the entries of
the Stopping to reuse for another call.

Consult the [HowTo tutorial](https://github.com/Goysa2/Stopping.jl/blob/master/test/examples/runhowto.jl) to learn more about the possibilities offered by Stopping.

You can also access other examples of algorithms in the [test/examples](https://github.com/Goysa2/Stopping.jl/blob/master/test/examples/) folder, which for instance illustrate the strength of Stopping with subproblems:
* Consult the [OptimSolver tutorial](https://github.com/Goysa2/Stopping.jl/blob/master/test/examples/run-optimsolver.jl) for more on how to use Stopping with nested algorithms.
* Check the [Benchmark tutorial](https://github.com/Goysa2/Stopping.jl/blob/master/test/examples/benchmark.jl) to see how Stopping can combined with [SolverBenchmark.jl](https://juliasmoothoptimizers.github.io/SolverBenchmark.jl/).
* Stopping can be adapted to closed solvers via a buffer function as in [Buffer tutorial](https://github.com/Goysa2/Stopping.jl/blob/master/test/examples/buffer.jl) for an instance with [Ipopt](https://github.com/JuliaOpt/Ipopt.jl) via [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl).

## How to install
Install and test the Stopping package with the Julia package manager:
```julia
pkg> add Stopping
pkg> test Stopping
```
You can also access the most up-to-date version of the Stopping package using:
```julia
pkg> add https://github.com/Goysa2/Stopping.jl
pkg> test Stopping
```
## Example

As an example, a naive version of the Newton method is provided [here](https://github.com/Goysa2/Stopping.jl/blob/master/test/examples/newton.jl). First we import the packages:
```
using LinearAlgebra, NLPModels, Stopping
```
We consider a quadratic test function, and create an uncontrained quadratic optimization problem using [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl):
```
A = rand(5, 5); Q = A' * A;
f(x) = 0.5 * x' * Q * x
nlp = ADNLPModel(f,  ones(5))
```

We now initialize the NLPStopping. First create a State.
```
nlp_at_x = NLPAtX(ones(5))
```
We use [unconstrained_check](https://github.com/Goysa2/Stopping.jl/blob/master/src/Stopping/nlp_admissible_functions.jl) as an optimality function
```
stop_nlp = NLPStopping(nlp, unconstrained_check, nlp_at_x)
```
Note that, since we used a default State, an alternative would have been:
```
stop_nlp = NLPStopping(nlp)
```

Now a basic version of Newton to illustrate how to use Stopping.
```
function newton(stp :: NLPStopping)

    #Notations
    pb = stp.pb; state = stp.current_state;
    #Initialization
    xt = state.x

    #First, call start! to check optimality and set an initial configuration
    #(start the time counter, set relative error ...)
    OK = update_and_start!(stp, x = xt, gx = grad(pb, xt), Hx = hess(pb, xt))

    while !OK
        #Compute the Newton direction (state.Hx only has the lower triangular)
        d = (state.Hx + state.Hx' - diagm(0 => diag(state.Hx))) \ (- state.gx)
        #Update the iterate
        xt = xt + d
        #Update the State and call the Stopping with stop!
        OK = update_and_stop!(stp, x = xt, gx = grad(pb, xt), Hx = hess(pb, xt))
    end

    return stp
end
```
Finally, we can call the algorithm with our Stopping:
```
stop_nlp = newton(stop_nlp)
```
and consult the Stopping to know what happened
```
#We can then ask stop_nlp the final status
@test :Optimal in status(stop_nlp, list = true)
#Explore the final values in stop_nlp.current_state
printstyled("Final solution is $(stop_nlp.current_state.x)", color = :green)
```
We reached optimality, and thanks to the Stopping structure this simple looking
algorithm verified at each step of the algorithm:
- time limit has been respected;
- evaluations of the problem are not excessive;
- the problem is not unbounded (w.r.t. x and f(x));
- there is no NaN in x, f(x), g(x), H(x);
- the maximum number of iteration (call to stop!) is limited.

## Long-Term Goals

Stopping is aimed as a tool for improving the reusability and robustness in the implementation of iterative algorithms. We warmly welcome any feedback or comment leading to potential improvements.

Future work will address more sophisticated problems such as mixed-integer optimization problems, optimization with uncertainty. The list of suggested optimality functions will be enriched with state of the art conditions.
