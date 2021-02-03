# Stopping.jl

This package provides general tools for the uniformization of stopping criteria for iterative solvers.
When a solver is called on an optimization model, four outcomes may happen:

1. the approximate solution is obtained, the problem is considered solved
2. the problem is declared unsolvable (unboundedness, infeasibility ...)
3. the maximum available resources are not sufficient to compute the solution
4. some algorithm dependent failure happens

There are many advantages in using Stopping:
- make your code more readible by outsourcing some tests to Stopping.
- Le the user a hand on the stopping criteria.
- 

## How to install
Install and test the Stopping package with the Julia package manager:
```julia
pkg> add Stopping
pkg> test Stopping
```
You can access the most up-to-date version of the Stopping package using:
```julia
pkg> add https://github.com/vepiteski/Stopping.jl
pkg> test Stopping
```

## Stopping

The stopping are represented by an instance (a subtype) of an `AbstractStopping`. Such instances are composed of
-  `problem :: Any` # an arbitrary instance of a problem
-  `meta :: AbstractStoppingMeta` # contains the used parameters and stopping status
-  `current_state :: AbstractState` # Current information on the problem

While the `problem` is up to the user, the `meta` and the `current_state` are specific features of Stopping.jl.
The `meta` contains all the parameters relative to the stopping criteria (tolerances, limits ...). We implemented
`StoppingMeta()` which should already offer a set of default parameters.
See [StoppingMeta](https://github.com/vepiteski/Stopping.jl/blob/master/src/Stopping/StoppingMetamod.jl) for more
detailed information.

The `current_state` contains all the information relative to a problem. We implemented a `GenericState` as an
illustration of the behavior of such object that typically contains:
- `x` the current iterate
- `d` the current direction
- `res` the current residual
- `current_time` the current time
- `current_score` the current optimality score
- ... other information relative to the problems

When running the loop, the `State` is updated and the `Stopping` declares optimality or not based on this information.

## Main Methods

The tool provides two main functions:
* `start!(stp :: AbstractStopping)` initializes the time and the tolerance at the starting point and check wether the initial guess is optimal.
* `stop!(stp :: AbstractStopping)` checks optimality of the current guess as well as failure of the system (unboundedness for instance) and maximum resources (number of evaluations of functions, elapsed time ...)

Stopping uses the informations furnished by the State to evaluate its functions. Communication between the two can be done through the following functions:
* `update_and_start!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call start!.
* `update_and_stop!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call stop!.
* `fill_in!(stp :: AbstractStopping, x :: Iterate)` a function that fill in all the State with all the informations required to correctly evaluate the stopping functions. This can reveal useful, for instance, if the user do not trust the informations furnished by the algorithm in the State.
* `reinit!(stp :: AbstractStopping)` reinitialize the entries of
the Stopping to reuse for another call.

## Example I: A Stopping-algorithm

Now a basic version of Newton to illustrate how to use Stopping.
```
function rand(stp :: GenericStopping, x_0 :: AbstractVector)

    #First, call start! to check optimality and set an initial configuration
    OK = update_and_start!(stp, x = xt)

    while !OK
        #Run some computations
        d = rand(length(x))
        #...
        x += d #Update the iterate

        #Update the State and call the Stopping with stop!
        OK = update_and_stop!(stp, x = x, d = d)
    end

    return stp
end
```

This example shows the most basic features of Stopping. It does many checks for you.
In this innocent-looking algorithm, the call to `update_and_stop!` will verifies
unboundedness of `x`, the time spent in the algorithm, the number of iterations (= number of call to `stop!`),
and the domain of `x` (in case some of its components become `NaN` for instance).

## Example II: Do you speak Stopping?

When using a Stopping-compatible algorithm, a.k.a an algorithm that takes a Stopping as an input and return it,
the user is free to explore the results the outputed result and influence the execution of the algorithm.

First, we need to create a Stopping.
```
x = ones(10)
problem = nothing #or your instance
stp = GenericStopping(pb, x, max_time = 10.) #short-cut initializing a `GenericState` and a `StoppingMeta`
@show stp.meta.max_time == 10. #by default the `kwargs...` are passed to the meta.
```
One can also creates separately a state and a meta to form a Stopping:
```
state = GenericState(x)
meta  = StoppingMeta(max_time = 10.)
stp = GenericStopping(pb, meta, state)
```
Once the `Stopping` has been initialized, we can call the algorithm and exploit the output.
```
stp = rand(stp, x)
```
To get the reason why the algorithm stopped we use `status`.
```
status(stp) #or `status(stp, rlist = true)` to have the complete list.
```
The solution as well as problem-related information can be accessed from the state.
```
sol = stp.current_state.x
```
