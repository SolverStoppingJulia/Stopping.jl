## Stopping

A Stopping is an instance (a subtype) of an `AbstractStopping`. Such instances minimally contain:
-  `problem :: Any` an arbitrary instance of a problem;
-  `meta :: AbstractStoppingMeta` contains the used parameters and stopping statuses;
-  `current_state :: AbstractState` current information/state of the problem.

While the `problem` is up to the user, the `meta` and the `current_state` are specific features of Stopping.jl.
The `meta` contains all the parameters relative to the stopping criteria (tolerances, limits ...). We implemented
`StoppingMeta()` which offers a set of default parameters that can be easily modified with keyword arguments. See [StoppingMeta](https://github.com/SolverStoppingJulia/Stopping.jl/blob/master/src/Stopping/StoppingMetamod.jl) for more detailed information. The native instances of `AbstractStopping` (`GenericStopping`, `NLPStoppping`, etc) contains more attributes (`stop_remote`, `main_stp`, `listofstates`, `stopping_user_struct`) that we will developed later on.

The `current_state` contains all the information relative to a problem. We implemented a `GenericState` as an
illustration of the behavior of such object that typically contains:
- `x` the current iterate;
- `d` the current direction;
- `res` the current residual;
- `current_time` the current time;
- `current_score` the current optimality score;
- ... other information relative to the problems.

When running the iterative loop, the `State` is updated and the `Stopping` make a decision based on this information.

## Main Methods

Stopping's main behavior is represented by two functions:
* `start!(:: AbstractStopping)` initializes the time and the tolerance at the starting point and stopping criteria.
* `stop!(:: AbstractStopping)` checks stopping criteria

Stopping uses the information furnished by the State to make a decision. Communication between the two can be done through the following functions:
* `update_and_start!(stp :: AbstractStopping; kwargs...)` updates the states with information furnished as kwargs and then call start!.
* `update_and_stop!(stp :: AbstractStopping; kwargs...)` updates the states with information furnished as kwargs and then call stop!.
* `fill_in!(stp :: AbstractStopping, x :: xtype(stp.current_state))` a function that fills in all the State with all the information required to evaluate the stopping functions correctly. This can reveal useful, for instance, if the user do not trust the information furnished by the algorithm in the State.
* `reinit!(stp :: AbstractStopping)` reinitialize the entries of the Stopping to reuse for another call.

### FAQ: How do I get more information?
As usual in Julia, we can use `?` to get functions' documentation.
```julia
? Stopping.stop!
```
