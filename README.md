# Stopping

![CI](https://github.com/SolverStoppingJulia/Stopping.jl/workflows/CI/badge.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/SolverStoppingJulia/Stopping.jl/badge.svg?branch=master)](https://coveralls.io/github/SolverStoppingJulia/Stopping.jl?branch=master)
[![codecov](https://codecov.io/gh/SolverStoppingJulia/Stopping.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SolverStoppingJulia/Stopping.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SolverStoppingJulia.github.io/Stopping.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SolverStoppingJulia.github.io/Stopping.jl/dev/)

## Purpose

Tools to ease the uniformization of stopping criteria in iterative solvers.

When a solver is called on an optimization model, four outcomes may happen:

1. the approximate solution is obtained, the problem is considered solved
2. the problem is declared unsolvable (unboundedness, infeasibility ...)
3. the maximum available resources are not sufficient to compute the solution
4. some algorithm dependent failure happens

This tool eases the first three items above. It defines a type

    mutable struct GenericStopping <: AbstractStopping
        problem              :: Any                  # an arbitrary instance of a problem
        meta                 :: AbstractStoppingMeta # contains the used parameters and stopping status
        current_state        :: AbstractState        # Current information on the problem
        main_stp             :: Union{AbstractStopping, Nothing} # Stopping of the main problem, or nothing
        listofstates         :: Union{ListStates, Nothing} # History of states
        user_specific_struct :: Dict                  # User-specific structure

The [StoppingMeta](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/src/Stopping/StoppingMetamod.jl) provides default tolerances, maximum resources, ...  as well as (boolean) information on the result.

### Your Stopping your way

The GenericStopping (with GenericState) provides a complete structure to handle stopping criteria.
Then, depending on the problem structure, you can specialize a new Stopping by
redefining a State and some functions specific to your problem.

We provide some specialization of the GenericStopping for optimization:
  * [NLPStopping](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/src/Stopping/NLPStoppingmod.jl) with [NLPAtX](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/src/State/NLPAtXmod.jl) as a specialized State for non-linear programming (based on [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)), or with [OneDAtX](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/src/State/OneDAtXmod.jl) as a specialized State for 1d optimization;
  * [LAStopping](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/src/Stopping/LinearAlgebraStopping.jl) with [GenericState](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/src/State/GenericStatemod.jl): for linear algebra problems.

## Functions

The tool provides two main functions:
* `start!(stp :: AbstractStopping)` initializes the time and the tolerance at the starting point and check wether the initial guess is optimal.
* `stop!(stp :: AbstractStopping)` checks optimality of the current guess as well as failure of the system (unboundedness for instance) and maximum resources (number of evaluations of functions, elapsed time ...)

Stopping uses the informations furnished by the State to evaluate its functions. Communication between the two can be done through the following functions:
* `update_and_start!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call start!.
* `update_and_stop!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call stop!.
* `fill_in!(stp :: AbstractStopping, x :: Iterate)` a function that fill in all the State with all the informations required to correctly evaluate the stopping functions. This can reveal useful, for instance, if the user do not trust the informations furnished by the algorithm in the State.
* `reinit!(stp :: AbstractStopping)` reinitialize the entries of
the Stopping to reuse for another call.

Consult the [HowTo tutorial](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/test/examples/runhowto.jl) to learn more about the possibilities offered by Stopping.

You can also access other examples of algorithms in the [test/examples](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/test/examples/) folder, which for instance illustrate the strenght of Stopping with subproblems:
* Consult the [OptimSolver tutorial](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/test/examples/run-optimsolver.jl) for more on how to use Stopping with nested algorithms.
* Check the [Benchmark tutorial](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/test/examples/benchmark.jl) to see how Stopping can combined with [SolverBenchmark.jl](https://juliasmoothoptimizers.github.io/SolverBenchmark.jl/).
* Stopping can be adapted to closed solvers via a buffer function as in [Buffer tutorial](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/test/examples/buffer.jl) for an instance with [Ipopt](https://github.com/JuliaOpt/Ipopt.jl) via [NLPModelsIpopt](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl).
* Consult the [WarmStart](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/test/examples/gradient-lbfgs.jl) to use Stopping in a warm-start context using internal user-defined structure and the list of states.

## How to install
Install and test the Stopping package with the Julia package manager:
```julia
pkg> add Stopping
pkg> test Stopping
```
You can access the most up-to-date version of the Stopping package using:
```julia
pkg> add https://github.com/SolverStoppingJulia/Stopping.jl
pkg> test Stopping
pkg> status Stopping
```
## Example

```
using NLPModels, Stopping #import the packages
nlp = ADNLPModel(x -> sum(x),  ones(2)) #initialize an NLPModel with automatic differentiation
```
We now initialize the `NLPStopping`. 
```
nlp_at_x = NLPAtX(ones(5)) #First create a State.
```
We use [unconstrained_check](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/src/Stopping/nlp_admissible_functions.jl) as an optimality function
```
stop_nlp = NLPStopping(nlp, nlp_at_x, optimality_check = unconstrained_check)
```
The following algorithm shows the most basic features of Stopping. It does many checks for you. In this innocent-looking algorithm, the call to `update_and_start!` and `update_and_stop!` will verifies unboundedness of `x`, the time spent in the algorithm, the number of iterations (= number of call to `stop!`), and the domain of `x` (in case some of its components become `NaN` for instance).
```julia
function rand_solver(stp :: AbstractStopping, x0 :: AbstractVector)

    x = x0
    #First, call start! to check optimality and set an initial configuration
    OK = update_and_start!(stp, x = x)

    while !OK
        #Run some computations and update the iterate
        d = rand(length(x))
        x += d

        #Update the State and call the Stopping with stop!
        OK = update_and_stop!(stp, x = x, d = d)
    end

    return stp
end
```
Finally, we can call the algorithm with our Stopping:
```
stop_nlp = rand_solver(stop_nlp, )
```
and consult it to know what happened
```
status(stop_nlp, list = true)
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
